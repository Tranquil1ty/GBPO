import torch
import numpy as np

def calculate_rewards_batch(candidate_items, target_items, input_seqs, num_rollouts, din_model, device, predict_fn, reward_topk=5, verbose=True, local_rank=0):
    B = target_items.size(0)
    K = num_rollouts
    cands_reshaped = candidate_items.view(B, K)
    
    if din_model:
        try:
            all_scores = []
            for b in range(B):
                user_seq_np = input_seqs[b].cpu().numpy()
                start = b * K
                end = (b + 1) * K
                cands_np = candidate_items[start:end].cpu().numpy()
                scores_k = predict_fn(din_model, user_seq_np, cands_np, device=device)
                all_scores.append(torch.from_numpy(scores_k).to(device))
            
            scores = torch.cat(all_scores)
            
        except Exception as e:
            if local_rank == 0:
                print(f"Warning: DIN prediction failed: {e}. Using binary rewards.")
            target_expanded = target_items.unsqueeze(1).expand(-1, K)
            rewards = (cands_reshaped == target_expanded).float()
            rewards[cands_reshaped < 1] = 0.0
            return rewards.view(B*K)

        scores_matrix = scores.view(B, K)
        topk = min(reward_topk, K)
        thresholds = torch.topk(scores_matrix, k=topk, dim=1).values[:, -1:]
        rewards = (scores_matrix >= thresholds).float()
        rewards[cands_reshaped < 1] = 0.0
        
    else:
        target_expanded = target_items.unsqueeze(1).expand(-1, K)
        rewards = (cands_reshaped == target_expanded).float()
        rewards[cands_reshaped < 1] = 0.0
        
    return rewards.view(B*K)

