import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from data_loader import DINDataInput


def calc_auc(raw_arr: List[List[float]]) -> Optional[float]:
    """
    Calculate AUC from list of [n_click, p_click, pred_score]
    Same implementation as DeepCTR-Torch
    """
    if not raw_arr:
        return None
        
    arr = sorted(raw_arr, key=lambda d: d[2])  # sort by prediction score ascending
    
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    
    if tp2 * fp2 > 0.0:
        return 1.0 - auc / (2.0 * tp2 * fp2)
    else:
        return None


def _auc_arr(pos_scores: np.ndarray, neg_scores: np.ndarray) -> List[List[float]]:
    """Convert positive and negative scores into format for calc_auc"""
    score_arr = []
    
    # Add positive samples: [n_click=0, p_click=1, pred_score]
    for s in pos_scores.tolist():
        score_arr.append([0, 1, s])
    
    # Add negative samples: [n_click=1, p_click=0, pred_score]
    for s in neg_scores.tolist():
        score_arr.append([1, 0, s])
    
    return score_arr


def calculate_ndcg_at_k(pos_ranks: List[int], k: int) -> float:
    """
    Calculate NDCG@k given the ranks of positive items
    
    Args:
        pos_ranks: List of ranks (1-based) where positive items appear
        k: Top-k for NDCG calculation
    
    Returns:
        Average NDCG@k score
    """
    ndcg_sum = 0.0
    for rank in pos_ranks:
        if rank <= k:
            # DCG@k = 1 / log2(rank + 1), IDCG@k = 1 / log2(2) = 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        ndcg_sum += ndcg
    
    return ndcg_sum / len(pos_ranks) if pos_ranks else 0.0


def evaluate_model(model, test_data: List[Tuple], batch_size: int,
                   device: torch.device, num_items: int, k: int = 10,
                   calculate_recall: bool = False, calculate_ndcg: bool = False) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
    """
    Evaluate DIN model on test data
    
    Args:
        model: DIN model instance (DINTrain)
        test_data: Test dataset in format [(user_features, pos_item, neg_item), ...]
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        num_items: Total number of items for ranking calculation
        k: Top-k for recall and NDCG calculation
        calculate_recall: Whether to calculate recall@k (can be slow)
        calculate_ndcg: Whether to calculate NDCG@k (can be slow)
    
    Returns:
        Tuple of (GAUC, AUC, Recall@k, NDCG@k)
    """
    model.DINModel.eval()
    
    gauc_sum = 0.0
    all_pos_scores = []
    all_neg_scores = []
    total_samples = 0
    
    with torch.no_grad():
        for _, batch in DINDataInput(test_data, batch_size, mode='test'):
            user_features = batch['user_features'].to(device)
            pos_items = batch['pos_items'].to(device)
            neg_items = batch['neg_items'].to(device)
            
            batch_size_actual = user_features.shape[0]
            
            # Calculate scores for positive items
            pos_scores = model.calculate_preference(user_features, pos_items.unsqueeze(-1))
            pos_scores = torch.sigmoid(pos_scores).squeeze(-1)
            
            # Calculate scores for negative items  
            neg_scores = model.calculate_preference(user_features, neg_items.unsqueeze(-1))
            neg_scores = torch.sigmoid(neg_scores).squeeze(-1)
            
            # Calculate GAUC (group AUC) - simple implementation
            batch_gauc = ((pos_scores > neg_scores).float()).mean()
            gauc_sum += batch_gauc.item() * batch_size_actual
            total_samples += batch_size_actual
            
            # Collect scores for overall AUC
            all_pos_scores.extend(pos_scores.cpu().numpy().tolist())
            all_neg_scores.extend(neg_scores.cpu().numpy().tolist())
    
    # Calculate overall metrics
    test_gauc = gauc_sum / total_samples if total_samples > 0 else 0.0
    
    # Calculate AUC
    if all_pos_scores and all_neg_scores:
        score_arr = _auc_arr(np.array(all_pos_scores), np.array(all_neg_scores))
        auc = calc_auc(score_arr)
    else:
        auc = None

    # --- Recall@k and NDCG@k Calculation (Optional) ---
    if not calculate_recall and not calculate_ndcg:
        model.DINModel.train()
        return test_gauc, auc, None, None

    hits = 0
    pos_ranks = []  # Store ranks of positive items for NDCG calculation
    item_chunk_size = 1024  # Process items in chunks to avoid OOM
    all_item_ids = torch.arange(1, num_items, device=device)

    with torch.no_grad():
        for _, batch in DINDataInput(test_data, batch_size, mode='test'):
            user_features = batch['user_features'].to(device)
            pos_items = batch['pos_items'].to(device)
            current_batch_size = user_features.shape[0]

            # Get scores for all items for the current batch of users
            all_scores = []
            for item_chunk in torch.split(all_item_ids, item_chunk_size):
                current_chunk_size = item_chunk.shape[0]

                # 修复bug: 正确地为每个用户重复物品序列
                # 对于batch中的每个用户，需要对所有chunk中的物品计算分数
                user_features_repeated = user_features.repeat_interleave(current_chunk_size, dim=0)
                # 将item_chunk扩展为与user_features_repeated匹配的形状
                items_repeated = item_chunk.unsqueeze(1).repeat(current_batch_size, 1).view(-1, 1)

                chunk_scores = model.calculate_preference(user_features_repeated, items_repeated)
                chunk_scores = torch.sigmoid(chunk_scores)
                chunk_scores = chunk_scores.view(current_batch_size, current_chunk_size)
                all_scores.append(chunk_scores)
            
            all_scores = torch.cat(all_scores, dim=1)

            # Get top K items
            _, top_k_indices = torch.topk(all_scores, k, dim=1)
            
            # Map indices back to item IDs (indices are 0-based from `all_item_ids`)
            top_k_items = all_item_ids[top_k_indices]

            # Calculate Recall@k
            if calculate_recall:
                hits += (top_k_items == pos_items.unsqueeze(1)).any(dim=1).sum().item()
            
            # Calculate NDCG@k
            if calculate_ndcg:
                # For each user, find the rank of positive item in all_scores
                for i in range(current_batch_size):
                    user_scores = all_scores[i]
                    pos_item = pos_items[i]
                    
                    # Find the rank of positive item (1-based rank)
                    # Higher scores should have lower ranks (rank 1 is best)
                    sorted_indices = torch.argsort(user_scores, descending=True)
                    sorted_items = all_item_ids[sorted_indices]
                    
                    # Find position of positive item
                    pos_rank = ((sorted_items == pos_item).nonzero(as_tuple=True)[0]).item() + 1
                    pos_ranks.append(pos_rank)
            
    recall_at_k = hits / len(test_data) if (calculate_recall and len(test_data) > 0) else None
    ndcg_at_k = calculate_ndcg_at_k(pos_ranks, k) if calculate_ndcg else None
    
    model.DINModel.train()
    return test_gauc, auc, recall_at_k, ndcg_at_k




class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current validation score (higher is better)
            model: Model to potentially save weights from
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                # Deep copy weights to avoid reference mutation
                import copy
                self.best_weights = copy.deepcopy(model.DINModel.state_dict())
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                import copy
                self.best_weights = copy.deepcopy(model.DINModel.state_dict())
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.DINModel.load_state_dict(self.best_weights)
                return True
            return False


if __name__ == "__main__":
    print("DIN Evaluation Module")
    print("This module provides evaluation utilities for DIN model.")
