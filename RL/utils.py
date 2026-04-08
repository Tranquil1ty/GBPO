import torch
import torch.nn.functional as F

def calculate_advantages(rewards):
    """Normalize rewards per group (sample) """
    # rewards shape: (B, K)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + 1e-8)

def compute_token_log_probs_from_codes(candidate_logits, global_codes, model_ref, temperature=1.0):
    """Return token-level log-probs for each code layer with vectorized FSA-based Trie masking.
    
    Args:
        candidate_logits: (N, L, V)
        global_codes: (N, L)
    Returns:
        token_log_probs: (N, L)
    """
    logits = candidate_logits.float()
    if temperature != 1.0:
        logits = logits / temperature

    # 1. Apply basic layer masking
    logits = model_ref.apply_layer_mask(logits)

    # 2. Vectorized FSA Trie masking
    trie_mode = getattr(model_ref, "use_trie_rl", "no_trie")
    if trie_mode == "trie_all" and hasattr(model_ref, "fsa_transitions"):
        N, L, _ = logits.shape
        # Initialize current state with the root state, then advance by bos_token_id
        # In TIGER, bos_token_id is always the first token of any sequence.
        current_states = model_ref.fsa_transitions[model_ref.fsa_root_id, model_ref.bos_token_id].expand(N)
        
        # Sequentially apply masks and transition states (L is small, e.g., 4)
        for l in range(L):
            # Apply mask for the current step (N, V)
            step_masks = model_ref.fsa_masks[current_states]
            logits[:, l].masked_fill_(step_masks, float('-inf'))
            
            # Transition to next state based on the actual chosen token
            next_tokens = global_codes[:, l]
            current_states = model_ref.fsa_transitions[current_states, next_tokens]

    log_softmax_probs = F.log_softmax(logits, dim=-1)  # (N, L, V)
    return torch.gather(log_softmax_probs, 2, global_codes.unsqueeze(-1)).squeeze(-1)  # (N, L)


def calculate_entropy(candidate_logits, model_ref, codes=None):
    """Calculate average token-level entropy with vectorized FSA-based Trie masking.
    
    Args:
        candidate_logits: (N, L, V)
        model_ref: used to apply layer masks and check trie config
        codes: (N, L) optional codes for Trie masking
    Returns:
        mean_entropy: scalar
    """
    logits = candidate_logits.float()
    
    # 1. Apply basic layer masking
    logits = model_ref.apply_layer_mask(logits)

    # 2. Vectorized FSA Trie masking
    trie_mode = getattr(model_ref, "use_trie_rl", "no_trie")
    if codes is not None and trie_mode == "trie_all" and hasattr(model_ref, "fsa_transitions"):
        N, L, _ = logits.shape
        current_states = model_ref.fsa_transitions[model_ref.fsa_root_id, model_ref.bos_token_id].expand(N)
        
        for l in range(L):
            step_masks = model_ref.fsa_masks[current_states]
            logits[:, l].masked_fill_(step_masks, float('-inf'))
            next_tokens = codes[:, l]
            current_states = model_ref.fsa_transitions[current_states, next_tokens]
    
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Entropy = -sum(p * log p). Handle log(0) by zeroing out p*log(p) where p=0
    token_entropy = -torch.sum(probs * log_probs.nan_to_num(posinf=0, neginf=0), dim=-1) # (N, L)
    return token_entropy.mean()

