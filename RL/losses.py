import torch
import torch.nn.functional as F
from .utils import compute_token_log_probs_from_codes

def compute_grpo_loss(candidate_logits, global_codes, advantages, model_ref, old_log_probs, clip_range=0.2, temperature=1.0):
    """Token-level GRPO loss."""
    current_log_probs = compute_token_log_probs_from_codes(candidate_logits, global_codes, model_ref, temperature=temperature)
    
    # Force Token-level: advantages (N*K) -> (N*K, 1) to broadcast with current_log_probs (N*K, L)
    adv = advantages.unsqueeze(1)

    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    surr1 = ratio * adv
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surr2 = ratio_clipped * adv
    policy_loss = torch.min(surr1, surr2)
    
    with torch.no_grad():
        # Gradient is 0 if clipped part is chosen: 
        # (adv > 0 and ratio > 1+eps) or (adv < 0 and ratio < 1-eps)
        is_clipped = ((adv > 0) & (ratio > 1.0 + clip_range)) | \
                     ((adv < 0) & (ratio < 1.0 - clip_range))
        clip_frac = is_clipped.float().mean()

    return -policy_loss, clip_frac

def compute_gbpo_loss(candidate_logits, global_codes, advantages, model_ref, old_log_probs, temperature=1.0):
    """Item-level GBPO loss (using mean over tokens)."""
    current_tok_logp = compute_token_log_probs_from_codes(candidate_logits, global_codes, model_ref, temperature=temperature)
    
    # Force Item-level: average over tokens
    current_log_probs = current_tok_logp.mean(dim=1)
    
    # If old_log_probs is sequence-level sum, we should probably use mean as well for consistency with current_log_probs
    # In trainer.py, old_log_probs is .sum(dim=1) for item level if it's token-based, or just sequence logp.
    # We'll assume current_log_probs and old_log_probs are both "mean log prob per token" for GBPO item-level.
    
    log_p_detach = current_log_probs.detach()
    
    # 1. Advantage >= 0
    log_den_pos = torch.max(log_p_detach, old_log_probs)
    log_ratio_pos = current_log_probs - log_den_pos
    
    # 2. Advantage < 0
    safe_log_p_detach = torch.clamp(log_p_detach, max=-1e-7)
    log_one_minus_p = torch.log1p(-torch.exp(safe_log_p_detach))
    log_den_neg = torch.max(old_log_probs, log_one_minus_p)
    log_ratio_neg = current_log_probs - log_den_neg
    
    log_ratio = torch.where(advantages >= 0, log_ratio_pos, log_ratio_neg)
    ratio = torch.exp(log_ratio)
    
    # GBPO always has gradient 1 in log-space (r/sg(r) or r/(1-sg(r)))
    # So zero-gradient clip_frac is 0.
    clip_frac = torch.tensor(0.0, device=advantages.device)
    
    loss_elements = -(ratio * advantages)
    return loss_elements, clip_frac


def compute_gupo_loss(candidate_logits, global_codes, advantages, model_ref, old_token_log_probs, temperature=1.0):
    """Token-level GBPO (GUPO): same GBPO formula applied per-token, no probability averaging over sequence."""
    current_tok_logp = compute_token_log_probs_from_codes(candidate_logits, global_codes, model_ref, temperature=temperature)
    # old_token_log_probs: (N*K, L)
    adv = advantages.unsqueeze(1)

    log_p_detach = current_tok_logp.detach()
    # 1. Advantage >= 0
    log_den_pos = torch.max(log_p_detach, old_token_log_probs)
    log_ratio_pos = current_tok_logp - log_den_pos
    # 2. Advantage < 0
    safe_log_p_detach = torch.clamp(log_p_detach, max=-1e-7)
    log_one_minus_p = torch.log1p(-torch.exp(safe_log_p_detach))
    log_den_neg = torch.max(old_token_log_probs, log_one_minus_p)
    log_ratio_neg = current_tok_logp - log_den_neg

    log_ratio = torch.where(adv >= 0, log_ratio_pos, log_ratio_neg)
    ratio = torch.exp(log_ratio)
    clip_frac = torch.tensor(0.0, device=advantages.device)
    loss_elements = -(ratio * adv)
    return loss_elements, clip_frac

def compute_gspo_loss(candidate_logits, global_codes, advantages,
                      model_ref, old_log_probs, clip_range=0.2, temperature=1.0):
    """
    Item-level GSPO loss (sequence-level PPO-clip with sequence ratio):
      s_i = exp( (1/L) * sum_t log pi/pi_old )
    """
    current_tok_logp = compute_token_log_probs_from_codes(
        candidate_logits, global_codes, model_ref, temperature=temperature
    )
    
    current_seq_logp = current_tok_logp.sum(dim=1)  # (N,)
    L = current_tok_logp.size(1)
    
    # old_log_probs is expected to be sequence-level sum for Item-level
    log_ratio_mean = (current_seq_logp - old_log_probs) / max(1, L)  # (N,)
    s = torch.exp(log_ratio_mean)  # sequence ratio s_i
    
    surr1 = s * advantages
    s_clipped = torch.clamp(s, 1.0 - clip_range, 1.0 + clip_range)
    surr2 = s_clipped * advantages
    
    loss_elements = -torch.min(surr1, surr2)
    
    with torch.no_grad():
        is_clipped = ((advantages > 0) & (s > 1.0 + clip_range)) | \
                     ((advantages < 0) & (s < 1.0 - clip_range))
        clip_frac = is_clipped.float().mean()
        
    return loss_elements, clip_frac

def compute_gppo_loss(candidate_logits, global_codes, advantages, model_ref, old_log_probs, 
                      eps_low=0.2, eps_high=0.28, temperature=1.0):
    """
    Token-level GPPO loss as per the specific formula:
    J = E [ min( r * A, bar{r} * A ) ]
    where bar{r} = clip(r, (1-eps_low)/sg(r) * r, (1+eps_high)/sg(r) * r)
    """
    current_tok_logp = compute_token_log_probs_from_codes(
        candidate_logits, global_codes, model_ref, temperature=temperature
    )
    
    # r_{i,t}(theta) = pi / pi_old
    log_ratio = current_tok_logp - old_log_probs
    r = torch.exp(log_ratio)
    
    # sg(r_{i,t}(theta))
    r_sg = r.detach()
    
    # bar{r}_{i,t}(theta) = clip( r, (1-eps_low)/r_sg * r, (1+eps_high)/r_sg * r )
    # Note: (1-eps)/r_sg * r preserves gradient through the 'r' in the numerator
    low_val  = (1.0 - eps_low)  / r_sg * r   # value == 1-eps_low, grad != 0
    high_val = (1.0 + eps_high) / r_sg * r   # value == 1+eps_high, grad != 0

    r_bar = torch.where(
        r < (1.0 - eps_low),
        low_val,
        torch.where(r > (1.0 + eps_high), high_val, r)
    )
    
    # Sequence-level advantages broadcasted to token-level
    adv = advantages.unsqueeze(1) 
    
    surr1 = r * adv
    surr2 = r_bar * adv
    
    # J_GPPO = min(surr1, surr2)
    policy_loss = torch.min(surr1, surr2)
    
    # Scale loss for negative advantages
    policy_loss = torch.where(adv < 0, 0.01 * policy_loss, policy_loss)
    
    # In GPPO, bar{r} = clip(r, (1-eps)/sg(r)*r, ...). 
    # Even when clipped, the gradient is (1-eps)/sg(r), which is non-zero.
    # Thus, zero-gradient clip_frac is 0.
    clip_frac = torch.tensor(0.0, device=advantages.device)
    
    return -policy_loss, clip_frac

def compute_dual_ppo_loss(candidate_logits, global_codes, advantages, model_ref, old_log_probs, 
                          clip_range=0.2, c=3.0, temperature=1.0):
    """
    Token-level Dual PPO loss.
    L_clip = min(r * A, clip(r, 1-eps, 1+eps) * A)
    L_dual = L_clip if A >= 0 else max(L_clip, c * A)
    """
    current_tok_logp = compute_token_log_probs_from_codes(
        candidate_logits, global_codes, model_ref, temperature=temperature
    )
    
    log_ratio = current_tok_logp - old_log_probs
    ratio = torch.exp(log_ratio)
    
    adv = advantages.unsqueeze(1) # (N*K, 1) broadcast to (N*K, L)
    
    # Standard PPO-Clip (L_clip)
    surr1 = ratio * adv
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surr2 = ratio_clipped * adv
    l_clip = torch.min(surr1, surr2)
    
    # Dual PPO Logic (L_dual)
    # When A < 0, we take max(l_clip, c * A)
    l_dual = torch.where(
        adv >= 0,
        l_clip,
        torch.max(l_clip, c * adv)
    )
    
    with torch.no_grad():
        # Standard PPO clip: gradient is 0 if clipped part is chosen
        is_clipped_std = ((adv > 0) & (ratio > 1.0 + clip_range)) | \
                         ((adv < 0) & (ratio < 1.0 - clip_range))
        
        # Dual PPO additional: if adv < 0 and c*adv > l_clip, 
        # l_dual becomes c*adv (constant), so gradient is 0.
        is_clipped_dual = (adv < 0) & (c * adv > l_clip)
        
        clip_frac = (is_clipped_std | is_clipped_dual).float().mean()
    
    return -l_dual, clip_frac

def compute_sapo_loss(candidate_logits, global_codes, advantages, model_ref, old_log_probs,
                      tau_pos=1.0, tau_neg=1.05, temperature=1.0):
    """
    Token-level SAPO loss.
    f(r) = sigmoid(tau * (r - 1)) * (4 / tau)
    J = E [ f(r) * A ]
    """
    current_tok_logp = compute_token_log_probs_from_codes(
        candidate_logits, global_codes, model_ref, temperature=temperature
    )
    
    log_ratio = current_tok_logp - old_log_probs
    ratio = torch.exp(log_ratio)
    
    adv = advantages.unsqueeze(1) # (N*K, 1) broadcast to (N*K, L)
    
    # Determine tau based on advantage sign
    tau = torch.where(adv > 0, torch.tensor(tau_pos, device=adv.device), torch.tensor(tau_neg, device=adv.device))
    
    # f(r) = sigmoid(tau * (r - 1)) * (4 / tau)
    # Using torch.sigmoid(tau * (ratio - 1))
    f_r = torch.sigmoid(tau * (ratio - 1.0)) * (4.0 / tau)
    
    # J = f(r) * A
    # We want to maximize this, so minimize negative
    loss_elements = -(f_r * adv)
    
    # SAPO uses sigmoid (smooth), so it never has 0 gradient.
    clip_frac = torch.tensor(0.0, device=adv.device)
    
    return loss_elements, clip_frac

def compute_dapo_loss(candidate_logits, global_codes, advantages, model_ref, old_log_probs,
                      eps_low=0.2, eps_high=0.28, temperature=1.0):
    """
    Token-level DAPO loss (Asymmetric PPO-Clip).
    J = E [ min( r * A, clip(r, 1-eps_low, 1+eps_high) * A ) ]
    """
    current_tok_logp = compute_token_log_probs_from_codes(
        candidate_logits, global_codes, model_ref, temperature=temperature
    )
    
    log_ratio = current_tok_logp - old_log_probs
    ratio = torch.exp(log_ratio)
    
    adv = advantages.unsqueeze(1) # (N*K, 1) broadcast to (N*K, L)
    
    # Asymmetric Clipping
    surr1 = ratio * adv
    ratio_clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    surr2 = ratio_clipped * adv
    
    # J = min(surr1, surr2)
    policy_loss = torch.min(surr1, surr2)
    
    with torch.no_grad():
        is_clipped = ((adv > 0) & (ratio > 1.0 + eps_high)) | \
                     ((adv < 0) & (ratio < 1.0 - eps_low))
        clip_frac = is_clipped.float().mean()

    return -policy_loss, clip_frac

def compute_dpo_loss(model_log_probs_w, model_log_probs_l, ref_log_probs_w, ref_log_probs_l, beta=0.1):
    """Standard DPO loss"""
    pi_logratios = model_log_probs_w - model_log_probs_l
    ref_logratios = ref_log_probs_w - ref_log_probs_l
    logits = pi_logratios - ref_logratios
    loss_elements = -F.logsigmoid(beta * logits)
    return loss_elements

def compute_online_dpo_loss(candidate_logits, global_codes, rewards, model_ref, old_log_probs, beta=0.1, temperature=1.0):
    """
    在线从 K 个候选样本中选择 reward 最高（chosen）和最低（rejected）的样本对进行 DPO 训练。
    - candidate_logits: (N*K, L, V)
    - global_codes: (N*K, L)
    - rewards: (N, K)
    - old_log_probs: (N*K, L) 或 (N*K,)
    """
    # 1. 计算当前的 Sequence-level Log Probs
    current_tok_logp = compute_token_log_probs_from_codes(candidate_logits, global_codes, model_ref, temperature=temperature)
    current_seq_logp = current_tok_logp.sum(dim=1) # (N*K,)
    
    # 2. 处理 Reference Log Probs
    is_token = (old_log_probs.ndim == 2)
    ref_seq_logp = old_log_probs.sum(dim=1) if is_token else old_log_probs # (N*K,)
    
    N, K = rewards.shape
    
    current_seq_logp = current_seq_logp.view(N, K)
    ref_seq_logp = ref_seq_logp.view(N, K)
    
    # 3. 找到每组中 reward 最大和最小的索引
    max_indices = torch.argmax(rewards, dim=1) # (N,)
    min_indices = torch.argmin(rewards, dim=1) # (N,)
    
    # 4. 过滤掉 max_reward == min_reward 的组（没有区分度）
    max_rewards = rewards.gather(1, max_indices.unsqueeze(1)).squeeze(1)
    min_rewards = rewards.gather(1, min_indices.unsqueeze(1)).squeeze(1)
    valid_mask = max_rewards > min_rewards
    
    if not valid_mask.any():
        return torch.tensor(0.0, device=candidate_logits.device, requires_grad=True)
    
    # 5. 提取 Chosen 和 Rejected 的概率
    chosen_logps = current_seq_logp[valid_mask, max_indices[valid_mask]]
    rejected_logps = current_seq_logp[valid_mask, min_indices[valid_mask]]
    ref_chosen_logps = ref_seq_logp[valid_mask, max_indices[valid_mask]]
    ref_rejected_logps = ref_seq_logp[valid_mask, min_indices[valid_mask]]
    
    # 6. 调用标准 DPO Loss
    return compute_dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=beta)

