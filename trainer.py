import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.amp
import os
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

# Project-specific imports
from RL import (
    compute_token_log_probs_from_codes,
    calculate_advantages, 
    calculate_entropy,
    compute_grpo_loss, 
    compute_gbpo_loss, 
    compute_gupo_loss,
    compute_gspo_loss, 
    compute_gppo_loss,
    compute_dual_ppo_loss,
    compute_sapo_loss,
    compute_dapo_loss,
    compute_online_dpo_loss,
    calculate_rewards_batch as rl_calculate_rewards_batch
)
from model import TIGER

# --- Constants ---
RL_MODES = ["GRPO", "GBPO", "GUPO", "GSPO", "GPPO", "DUAL_PPO", "SAPO", "DAPO", "DPO"]

# Note: 'predict' is imported from 'prediction' which should be in sys.path
# via the DIN/functions directory setup in main.py
try:
    from prediction import predict
except ImportError:
    # If not in path yet, we'll handle it via late binding or assuming it's provided
    predict = None

class Trainer:
    def __init__(self, config: Any, model: TIGER, din_model, device, local_rank, ddp=False):
        self.cfg = config
        self.model = model
        self.din_model = din_model
        self.device = device
        self.local_rank = local_rank
        self.ddp = ddp
        
        # Off-policy reference
        self.ref_model = None 
        
        # State
        self.best_metric = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        self.best_model_state = None  # In-memory best checkpoint
        
    def log(self, msg):
        if self.local_rank == 0:
            print(msg)

    def compute_grad_norms(self, loss):
        """Compute the norm of gradients for a given loss without updating weights."""
        if loss is None or not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return 0.0
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return 0.0
            
        try:
            # We use retain_graph=True because we might call this multiple times for different loss parts
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            norm_sq = 0.0
            for g in grads:
                if g is not None:
                    norm_sq += g.float().pow(2).sum()
            return torch.sqrt(norm_sq).item()
        except Exception:
            return 0.0

    def save_checkpoint(self, path, extra_info={}):
        if self.local_rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            to_save = self.model.module if self.ddp else self.model
            state = {
                'model_state_dict': to_save.state_dict(),
                'config': vars(self.cfg),
                'epoch': self.current_epoch,
                'best_metric': self.best_metric,
                **extra_info
            }
            torch.save(state, path)
            self.log(f"✅ Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            target = self.model.module if self.ddp else self.model
            target.load_state_dict(ckpt['model_state_dict'], strict=False)
            
            self.log(f"✅ Loaded checkpoint from {path}")
            self.log(f"   Checkpoint Epoch: {ckpt.get('epoch', 0)}")
            self.log(f"   Best Metric: {ckpt.get('best_metric', 0.0)}")
            return True
        except Exception as e:
            self.log(f"❌ Failed to load checkpoint: {e}")
            return False

    def create_reference_model(self):
        """Clone current model as reference for Off-Policy GRPO"""
        if self.local_rank == 0: print("Creating Reference Model...")
        # Re-init a fresh model to avoid tying weights, then load state
        self.ref_model = TIGER(
            num_items=self.cfg.NUM_ITEMS,
            max_len=self.cfg.MODEL_MAX_LEN_LOO,
            dropout=self.cfg.DROPOUT,
            t5_model_name_or_path=self.cfg.T5_MODEL_PATH,
            base_path=self.cfg.BASE_PATH,
            dataset=self.cfg.DATASET,
            use_trie=self.cfg.USE_TRIE
        ).to(self.device)
        
        source = self.model.module if self.ddp else self.model
        self.ref_model.load_state_dict(source.state_dict())
        self.ref_model.eval()
        for p in self.ref_model.parameters(): p.requires_grad = False
        if self.local_rank == 0: print("Reference Model Ready.")

    def update_reference_model(self):
        """Update reference model with current model weights"""
        if self.ref_model is None:
            return
        if self.local_rank == 0: self.log("🔄 Updating Reference Model from Current Model...")
        source = self.model.module if self.ddp else self.model
        self.ref_model.load_state_dict(source.state_dict())
        self.ref_model.eval()
        for p in self.ref_model.parameters(): p.requires_grad = False
        
    def calculate_rewards_batch(self, candidate_items, target_items, input_seqs, num_rollouts):
        global predict
        # Ensure 'predict' is available
        if predict is None:
            try:
                from prediction import predict as din_predict
                predict = din_predict
            except ImportError:
                self.log("⚠️ Warning: 'predict' function not found in sys.path. Binary rewards will be used as fallback.")

        return rl_calculate_rewards_batch(
            candidate_items, target_items, input_seqs, num_rollouts, 
            self.din_model, self.device, predict, getattr(self.cfg, "REWARD_TOPK", 5), self.cfg.VERBOSE, self.local_rank
        )

    def train_epoch(self, loader, optimizer, epoch_idx, mode="SFT"):
        self.model.train()
        total_loss = 0
        count = 0
        
        pbar = tqdm(loader, disable=(self.local_rank != 0))
        pbar.set_description(f"Epoch {epoch_idx} [{mode}]")
        
        for batch in pbar:
            seqs, targets = batch
            seqs, targets = seqs.to(self.device), targets.to(self.device)
            
            valid_mask = (targets.squeeze(-1) != self.cfg.PADDING_ITEM_ID)
            
            optimizer.zero_grad()
            loss_val = torch.tensor(0.0, device=self.device, requires_grad=True)
            avg_uniqueness = None
            avg_validity = None
            sft_loss_weighted = 0.0
            rl_loss_weighted = 0.0
            
            if valid_mask.sum() > 0:
                with torch.amp.autocast('cuda', enabled=self.cfg.USE_BF16, dtype=torch.bfloat16):
                    if mode == "SFT":
                        sft_seqs = seqs[valid_mask]
                        sft_targets = targets[valid_mask]
                        att_mask = (sft_seqs != 0)
                        _, sft_loss = self.model(sft_seqs, att_mask, target_item_ids=sft_targets)
                        loss_val = sft_loss.float()
                        sft_loss_weighted = loss_val.item()
                        
                        grad_info = {}
                        if self.cfg.VERBOSE:
                            grad_info['g_sft'] = self.compute_grad_norms(loss_val)
                        
                    elif mode in RL_MODES:
                        sft_seqs = seqs[valid_mask]
                        sft_targets = targets[valid_mask]
                        att_mask = (sft_seqs != 0)
                        _, sft_loss = self.model(sft_seqs, att_mask, target_item_ids=sft_targets)

                        
                        num_grpo = max(1, int(0.3 * valid_mask.sum().item()))
                        valid_indices = torch.where(valid_mask)[0]
                        perm = torch.randperm(len(valid_indices))[:num_grpo]
                        grpo_indices = valid_indices[perm]
                        
                        grpo_seqs = seqs[grpo_indices]
                        grpo_targets = targets[grpo_indices].squeeze(-1)
                        grpo_att_mask = (grpo_seqs != 0)
                        
                        rollout_model = self.ref_model if self.ref_model else (self.model.module if self.ddp else self.model)
                        rollout_bs = self.cfg.ROLLOUT_BATCH_SIZE
                        num_sub_batches = (len(grpo_seqs) + rollout_bs - 1) // rollout_bs
                        
                        all_cands, all_codes, all_old_log_probs, all_old_token_log_probs = [], [], [], []
                        
                        with torch.no_grad():
                            for i in range(num_sub_batches):
                                start_idx = i * rollout_bs
                                end_idx = min((i + 1) * rollout_bs, len(grpo_seqs))
                                sub_cands, sub_codes, sub_old_log_probs, sub_old_token_log_probs = rollout_model.generate_candidates(
                                    grpo_seqs[start_idx:end_idx], grpo_att_mask[start_idx:end_idx], 
                                    self.cfg.NUM_CANDIDATES, strategy=self.cfg.GEN_STRATEGY, temperature=self.cfg.TEMPERATURE,
                                    use_trie=self.cfg.USE_TRIE
                                )
                                all_cands.append(sub_cands)
                                all_codes.append(sub_codes)
                                all_old_log_probs.append(sub_old_log_probs)
                                all_old_token_log_probs.append(sub_old_token_log_probs)
                        
                        self.model.train()
                        cands = torch.cat(all_cands)
                        codes = torch.cat(all_codes)
                        old_log_probs = torch.cat(all_old_log_probs)
                        old_token_log_probs = torch.cat(all_old_token_log_probs)
                        
                        # Calculate Uniqueness & Validity
                        num_grpo_actual = len(grpo_indices)
                        cands_np = cands.view(num_grpo_actual, self.cfg.NUM_CANDIDATES).cpu().numpy()
                        uniqueness_list = [len(set(row)) / self.cfg.NUM_CANDIDATES for row in cands_np]
                        avg_uniqueness = sum(uniqueness_list) / len(uniqueness_list)
                        avg_validity = (cands_np > 0).mean()

                if mode in RL_MODES and valid_mask.sum() > 0:
                    rewards_flat = self.calculate_rewards_batch(cands, grpo_targets, grpo_seqs, self.cfg.NUM_CANDIDATES)
                    rewards_reshaped = rewards_flat.view(len(grpo_indices), self.cfg.NUM_CANDIDATES).float()
                    advantages = calculate_advantages(rewards_reshaped).view(-1)
                    rl_temp = self.cfg.TEMPERATURE if getattr(self.cfg, "GEN_STRATEGY", "rollout") == "rollout" else 1.0

                    with torch.amp.autocast('cuda', enabled=self.cfg.USE_BF16, dtype=torch.bfloat16):
                        # 重要：RL 计算 Logits 时需关闭 Dropout 以保持与 Rollout 一致（但保留梯度）
                        was_training = self.model.training
                        self.model.eval()
                        
                        logits, _ = self.model(
                            input_seq=grpo_seqs,
                            attention_mask=grpo_att_mask,
                            labels=codes,
                            K=self.cfg.NUM_CANDIDATES
                        )
                        
                        if was_training:
                            self.model.train()
                        
                        model_active = self.model.module if self.ddp else self.model
                        
                        # Calculate entropy for monitoring
                        entropy_val = calculate_entropy(logits, model_active, codes=codes)
                        
                        L = max(1, codes.size(1))
                        clip_frac = 0.0

                        # Prepare for gradient calculation if verbose
                        rl_loss_all = None
                        
                        if mode.upper() == "GRPO":
                            rl_loss_all, clip_frac = compute_grpo_loss(
                                logits, codes, advantages, model_active, old_token_log_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "GBPO":
                            rl_old_probs = old_log_probs / L
                            rl_loss_all, clip_frac = compute_gbpo_loss(
                                logits, codes, advantages, model_active, rl_old_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "GUPO":
                            rl_loss_all, clip_frac = compute_gupo_loss(
                                logits, codes, advantages, model_active, old_token_log_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "GSPO":
                            rl_loss_all, clip_frac = compute_gspo_loss(
                                logits, codes, advantages, model_active, old_log_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "GPPO":
                            rl_loss_all, clip_frac = compute_gppo_loss(
                                logits, codes, advantages, model_active, old_token_log_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "DUAL_PPO":
                            rl_loss_all, clip_frac = compute_dual_ppo_loss(
                                logits, codes, advantages, model_active, old_token_log_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "SAPO":
                            rl_loss_all, clip_frac = compute_sapo_loss(
                                logits, codes, advantages, model_active, old_token_log_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "DAPO":
                            rl_loss_all, clip_frac = compute_dapo_loss(
                                logits, codes, advantages, model_active, old_token_log_probs, temperature=rl_temp
                            )
                        elif mode.upper() == "DPO":
                            rl_loss_all = compute_online_dpo_loss(
                                logits, codes, rewards_reshaped, model_active, old_log_probs, 
                                beta=getattr(self.cfg, "DPO_BETA", 0.1), temperature=rl_temp
                            )
                            clip_frac = 0.0
                        
                        rl_loss = rl_loss_all.mean()
                        
                        # --- Graduate calculation for monitoring ---
                        grad_info = {}
                        if self.cfg.VERBOSE:
                            # 1. SFT (Positive only)
                            if sft_loss is not None:
                                grad_info['g_sft'] = self.compute_grad_norms(sft_loss.float())
                            
                            # 2. RL Positive and Negative
                            if rl_loss_all is not None and mode.upper() != "DPO":
                                if rl_loss_all.ndim == 2: # Token-level
                                    adv_expanded = advantages.unsqueeze(1).expand_as(rl_loss_all)
                                    pos_mask = (adv_expanded > 0)
                                    neg_mask = (adv_expanded < 0)
                                else: # Item-level
                                    pos_mask = (advantages > 0)
                                    neg_mask = (advantages < 0)
                                
                                if pos_mask.any():
                                    l_pos = rl_loss_all[pos_mask].mean()
                                    grad_info['g_rl_pos'] = self.compute_grad_norms(l_pos)
                                if neg_mask.any():
                                    l_neg = rl_loss_all[neg_mask].mean()
                                    grad_info['g_rl_neg'] = self.compute_grad_norms(l_neg)
                            elif mode.upper() == "DPO":
                                grad_info['g_rl_dpo'] = self.compute_grad_norms(rl_loss.float())

                        # Set SFT weight based on mode if not explicitly provided
                        sft_weight = getattr(self.cfg, "SFT_LOSS_WEIGHT", None)
                        if sft_weight is None:
                            if mode.upper() in RL_MODES:
                                sft_weight = 0.1
                            else:
                                sft_weight = 0.1  # Default fallback
                        
                        w_sft = sft_weight * sft_loss.float() if sft_loss is not None else torch.tensor(0.0, device=self.device)
                        w_rl = self.cfg.GRPO_LOSS_WEIGHT * rl_loss.float()
                        loss_val = w_sft + w_rl
                        sft_loss_weighted = w_sft.item()
                        rl_loss_weighted = w_rl.item()

                loss_val.backward()
                optimizer.step()
            
            total_loss += loss_val.item()
            count += 1
            
            if self.local_rank == 0:
                postfix = {'loss': f"{loss_val.item():.4f}"}
                if mode in RL_MODES:
                    postfix['sft_l'] = f"{sft_loss_weighted:.4f}"
                    postfix['rl_l'] = f"{rl_loss_weighted:.4f}"
                    if 'clip_frac' in locals():
                        postfix['clip'] = f"{clip_frac:.2f}"
                    if 'entropy_val' in locals():
                        postfix['ent'] = f"{entropy_val.item():.4f}"
                if avg_uniqueness is not None:
                    postfix['uniq'] = f"{avg_uniqueness:.4f}"
                if avg_validity is not None:
                    postfix['valid'] = f"{avg_validity:.4f}"
                
                # Add gradient info to postfix
                if 'grad_info' in locals():
                    for k, v in grad_info.items():
                        postfix[k] = f"{v:.4f}"
                        
                pbar.set_postfix(postfix)
        
        return total_loss / count if count > 0 else 0

    def evaluate(self, eval_data, k_list, beam_size=None):
        self.model.eval()
        
        if self.ddp:
            world_size = dist.get_world_size()
            local_rank = self.local_rank
            eval_data_shard = eval_data[local_rank::world_size]
        else:
            eval_data_shard = eval_data
            
        hits = {k: [] for k in k_list}
        ndcgs = {k: [] for k in k_list}
        validities = []
        
        batch_size = self.cfg.EVAL_BATCH_SIZE
        num_batches = (len(eval_data_shard) + batch_size - 1) // batch_size
        
        iterator = tqdm(range(num_batches), disable=(self.local_rank != 0), desc="Evaluating")
        model_eval = self.model.module if self.ddp else self.model
        
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(eval_data_shard))
            batch_data = eval_data_shard[start_idx:end_idx]
            
            seqs_list = [x[0] for x in batch_data]
            targets_list = [x[1] for x in batch_data]
            
            input_seq_batch = torch.LongTensor(np.array(seqs_list)).to(self.device)
            attention_mask_batch = (input_seq_batch != 0)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.cfg.USE_BF16, dtype=torch.bfloat16):
                    current_beam_size = beam_size if beam_size is not None else self.cfg.EVAL_BEAM_SIZE
                    batch_preds, batch_validity = model_eval.predict(
                        input_seq_batch, 
                        attention_mask_batch, 
                        beam_size=current_beam_size, 
                        topk=max(k_list)
                    )
                validities.append(batch_validity)
            
            batch_preds_np = batch_preds.cpu().numpy()
            targets_np = np.array(targets_list)
            
            valid_mask = (targets_np != 0)
            if np.any(valid_mask):
                v_preds = batch_preds_np[valid_mask]
                v_targets = targets_np[valid_mask]
                
                matches = (v_preds == v_targets[:, None])
                matched_sample_idx, matched_rank_idx = np.where(matches)
                
                ranks = np.full(len(v_targets), -1)
                ranks[matched_sample_idx] = matched_rank_idx
                
                for k in k_list:
                    hit_mask = (ranks >= 0) & (ranks < k)
                    hits[k].extend(hit_mask.astype(float).tolist())
                    
                    ndcg_scores = np.zeros_like(ranks, dtype=float)
                    ndcg_scores[hit_mask] = 1.0 / np.log2(ranks[hit_mask] + 2)
                    ndcgs[k].extend(ndcg_scores.tolist())
                    
        metrics = {}
        local_sum_validity = sum(validities)
        local_count_validity = len(validities)
        
        if self.ddp:
            val_stats = torch.tensor([local_sum_validity, local_count_validity], dtype=torch.float64, device=self.device)
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            metrics['Validity'] = val_stats[0].item() / max(1, val_stats[1].item())
        else:
            metrics['Validity'] = local_sum_validity / max(1, local_count_validity) if local_count_validity > 0 else 0.0

        for k in k_list:
            local_sum_hits = sum(hits[k])
            local_sum_ndcgs = sum(ndcgs[k])
            local_count = len(hits[k])
            
            if self.ddp:
                stats_tensor = torch.tensor([local_sum_hits, local_sum_ndcgs, local_count], dtype=torch.float64, device=self.device)
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                
                total_hits = stats_tensor[0].item()
                total_ndcgs = stats_tensor[1].item()
                total_count = stats_tensor[2].item()
            else:
                total_hits = local_sum_hits
                total_ndcgs = local_sum_ndcgs
                total_count = local_count
            
            if total_count > 0:
                metrics[f'HitRate@{k}'] = total_hits / total_count
                metrics[f'NDCG@{k}'] = total_ndcgs / total_count
            else:
                metrics[f'HitRate@{k}'] = 0.0
                metrics[f'NDCG@{k}'] = 0.0
            
        return metrics

    def run(self, train_loader, valid_data, test_data):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.LEARNING_RATE)
        
        # Mode Switching Logic
        if self.cfg.MODE == "SFT":
            self.log("🚀 Starting SFT Training Mode")
            patience_limit = self.cfg.SFT_PATIENCE
        else:
            self.log(f"🚀 Starting {self.cfg.MODE} Training Mode (On-Policy: {self.cfg.ON_POLICY})")
            # Load SFT Checkpoint first
            if not self.load_checkpoint(self.cfg.PRETRAINED_PATH):
                self.log(f"⚠️ SFT Checkpoint not found, starting from scratch (might be suboptimal for {self.cfg.MODE})")
            
            # Create reference model for Off-Policy OR DPO
            if not self.cfg.ON_POLICY:
                self.create_reference_model()
            else:
                self.log("ℹ️ On-Policy Mode: Sampling from current model.")
                
            patience_limit = self.cfg.GRPO_PATIENCE

        # Determine starting epoch: Resume from ckpt if in RL mode, otherwise start from 0
        start_epoch = self.current_epoch + 1 if (self.cfg.MODE != "SFT" and self.current_epoch > 0) else 0
        
        for epoch in range(start_epoch, self.cfg.EPOCHS):
            self.current_epoch = epoch
            if self.ddp: train_loader.sampler.set_epoch(epoch)

            # Update Reference Model (Off-Policy only, for GRPO/GBPO/GSPO/GPPO/DUAL_PPO/SAPO/DAPO/DPO)
            if self.cfg.MODE.upper() in RL_MODES and not self.cfg.ON_POLICY and epoch > start_epoch and (epoch - start_epoch) % self.cfg.REF_UPDATE_INTERVAL == 0:
                self.update_reference_model()
            
            loss = self.train_epoch(train_loader, optimizer, epoch, mode=self.cfg.MODE)
            self.log(f"Epoch {epoch} Loss: {loss:.4f}")
            
            # SFT 模式下前 15 个 epoch 不进行评估，RL 模式则正常评估
            if self.cfg.MODE == "SFT" and epoch < 15: continue

            # Evaluation
            # Run evaluation on ALL ranks (internal logic handles sharding and aggregation)
            metrics = self.evaluate(valid_data, self.cfg.K_LIST_EVAL)
            key_metric = metrics.get('HitRate@10', 0)

            # Every rank tracks best metric and saves to memory
            is_new_best = False
            if key_metric > self.best_metric:
                self.best_metric = key_metric
                self.patience_counter = 0
                
                # Save best model state to CPU memory
                source = self.model.module if self.ddp else self.model
                self.best_model_state = {k: v.cpu().clone() for k, v in source.state_dict().items()}
                is_new_best = True
            else:
                self.patience_counter += 1
                # Check for sudden performance drop (> 0.025)
                if self.best_metric > 0 and key_metric < self.best_metric - 0.02:
                    if self.local_rank == 0:
                        self.log(f"🛑 Sudden performance drop detected: {key_metric:.4f} vs best {self.best_metric:.4f}. Triggering early stopping.")
                    self.patience_counter = patience_limit

            # Only Rank 0 logs metrics and handles file checkpointing
            if self.local_rank == 0:
                validity = metrics.get('Validity', 0)
                self.log(f"EVAL [Epoch {epoch}]: Hit@10={key_metric:.4f}, NDCG@10={metrics.get('NDCG@10', 0):.4f}, Validity={validity:.4f}")
                
                # Print all metrics sorted
                sorted_keys = sorted(metrics.keys())
                msg = "All Metrics:\n" + "\n".join([f"  {k}: {metrics[k]:.4f}" for k in sorted_keys])
                self.log(msg)
                
                if is_new_best:
                    # Construct save path dynamically based on mode
                    if self.cfg.MODE == "SFT":
                        save_path = self.cfg.PRETRAINED_PATH
                    else:
                        mode_lower = self.cfg.MODE.lower()
                        save_path = self.cfg.PRETRAINED_PATH.replace("sft", mode_lower)
                        if save_path == self.cfg.PRETRAINED_PATH:
                            base_dir = os.path.dirname(self.cfg.PRETRAINED_PATH)
                            filename = os.path.basename(self.cfg.PRETRAINED_PATH)
                            save_path = os.path.join(base_dir, f"{mode_lower}_{filename}")

                    self.save_checkpoint(save_path)
                else:
                    self.log(f"No improvement. Patience: {self.patience_counter}/{patience_limit}")

            # If it's a new best, run test evaluation on all ranks
            if is_new_best:
                test_metrics = self.evaluate(test_data, self.cfg.K_LIST_EVAL)
                if self.local_rank == 0:
                    self.log(f"TEST [Best Valid]: Hit@10={test_metrics.get('HitRate@10',0):.4f}, NDCG@10={test_metrics.get('NDCG@10',0):.4f}, Validity={test_metrics.get('Validity',0):.4f}")
                    
                    sorted_keys = sorted(test_metrics.keys())
                    msg = "All Test Metrics:\n" + "\n".join([f"  {k}: {test_metrics[k]:.4f}" for k in sorted_keys])
                    self.log(msg)
            
            # DDP Sync for Early Stopping
            if self.ddp:
                stop_signal = torch.tensor(1 if self.patience_counter >= patience_limit else 0).to(self.device)
                dist.broadcast(stop_signal, src=0) 
                if stop_signal.item() == 1:
                    if self.local_rank == 0: self.log("🛑 Early stopping triggered.")
                    break
            else:
                if self.patience_counter >= patience_limit:
                    self.log("🛑 Early stopping triggered.")
                    break

        # --- Final Comprehensive Evaluation using Best In-Memory Weights ---
        if self.best_model_state is not None:
            self.log("\n" + "="*30 + " Final Evaluation (Best In-Memory Model) " + "="*30)
            
            # Load best weights back into model
            target = self.model.module if self.ddp else self.model
            target.load_state_dict(self.best_model_state)
            
            # Requested long K_LIST_EVAL
            final_k_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
            final_test_metrics = self.evaluate(test_data, final_k_list, beam_size=256)
            
            if self.local_rank == 0:
                self.log("Final Results on Test Data (All K):")
                sorted_keys = sorted(final_test_metrics.keys())
                for k in sorted_keys:
                    self.log(f"  {k}: {final_test_metrics[k]:.4f}")

