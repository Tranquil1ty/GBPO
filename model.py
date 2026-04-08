import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from transformers import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput

# Project-specific imports
# These assume the 'onerec' root is in sys.path
from t5_flash.modeling_t5 import T5Config, T5ForConditionalGeneration
from trie import Trie, tiger_prefix_allowed_tokens_fn_factory
from RL import compute_token_log_probs_from_codes

class TIGER(nn.Module):
    def _normalize_trie_mode(self, mode):
        """Strictly normalize any input to one of the three modes."""
        if mode is True or mode == "true": return "trie_all"
        if mode is False or mode == "false" or mode == "no_trie" or mode is None: return "no_trie"
        if mode in ["trie_gen", "trie_all"]: return mode
        return "no_trie"

    def __init__(self, num_items, max_len=20, dropout=0.1, t5_model_name_or_path="t5-small", 
                 num_code_layers=4, codebook_size=256, base_path=None, dataset="Beauty", use_trie="no_trie"):
        super(TIGER, self).__init__()
        self.t5_config = T5Config.from_pretrained(t5_model_name_or_path)
        self.num_code_layers = num_code_layers 
        self.codebook_size = codebook_size   
        self.bos_token_id = self.codebook_size * self.num_code_layers
        self.actual_vocab_size = self.codebook_size * self.num_code_layers + 1

        self.t5_config.vocab_size = self.actual_vocab_size
        self.t5_config.decoder_start_token_id = self.bos_token_id
        self.t5_config.eos_token_id = self.bos_token_id
        self.t5_config.pad_token_id = self.bos_token_id
        self.t5_config.use_flash_attention = True
    
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model_name_or_path, config=self.t5_config, ignore_mismatched_sizes=True
        )

        t5_hidden_size = self.t5_config.d_model
        self.semantic_emb = nn.Embedding(self.actual_vocab_size, t5_hidden_size, padding_idx=None)
        self.pos_emb = nn.Embedding(max_len * self.num_code_layers + 1, t5_hidden_size) 
        self.emb_dropout = nn.Dropout(dropout)
        self.t5_model.set_input_embeddings(self.semantic_emb)
        self.t5_model.tie_weights()
        if self.t5_model.get_input_embeddings().weight.data_ptr() != self.t5_model.lm_head.weight.data_ptr():
            self.t5_model.lm_head.weight = self.t5_model.get_input_embeddings().weight
        assert self.t5_model.get_input_embeddings().weight.data_ptr() == self.t5_model.lm_head.weight.data_ptr()

        # Precompute layer masks for performance
        L = self.num_code_layers
        V = self.actual_vocab_size
        idx = torch.arange(V).unsqueeze(0).expand(L, V)
        off = torch.arange(L).unsqueeze(1) * self.codebook_size
        mask = (idx >= off) & (idx < off + self.codebook_size)
        self.register_buffer('layer_masks', mask) # (L, V)

        if base_path is None: raise ValueError("base_path must be provided")
        
        item2code_path = os.path.join(base_path, 'data/processed_data',dataset, 'item2code_rkmeans_4L_llama.npy')
        code2item_path = os.path.join(base_path, 'data/processed_data',dataset, 'code2item_rkmeans_4L_llama.pkl')
        
        item2code_np = np.load(item2code_path)
        with open(code2item_path, 'rb') as f:
            self.code2item_dict = pickle.load(f)
            
        self.register_buffer('item2code', torch.from_numpy(item2code_np).long())

        # Always build Trie for evaluation default use
        sequences = []
        for codes in item2code_np:
            seq = [self.bos_token_id]
            for layer_idx, code in enumerate(codes):
                seq.append(int(code) + layer_idx * self.codebook_size)
            sequences.append(seq)
        
        self.trie = Trie(sequences)
        self.prefix_allowed_tokens_fn = tiger_prefix_allowed_tokens_fn_factory(
            self.trie, self.codebook_size, self.num_code_layers, self.bos_token_id
        )
        # Control RL behavior from outside via this mode: no_trie, trie_gen, trie_all
        self.use_trie_rl = self._normalize_trie_mode(use_trie)

        if self.use_trie_rl == "trie_all":
            # Precompute FSA tables for vectorized masking (only for trie_all)
            fsa_trans, fsa_masks, fsa_root = self.trie.get_fsa_tables(self.actual_vocab_size, torch.device('cpu'))
            self.register_buffer('fsa_transitions', fsa_trans)
            self.register_buffer('fsa_masks', fsa_masks)
            self.register_buffer('fsa_root_id', torch.tensor(fsa_root, dtype=torch.long))

    def get_item_codes(self, item_ids):
        """Encapsulate item to code mapping logic with boundary checks"""
        clamped_ids = torch.clamp(item_ids, 0, self.item2code.size(0) - 1)
        return self.item2code[clamped_ids]

    def apply_layer_mask(self, logits, layer_idx=None):
        if layer_idx is not None:
            return logits.masked_fill(~self.layer_masks[layer_idx].unsqueeze(0), float('-inf'))
        else:
            # logits shape expected to be (..., L, V) or (..., V)
            if logits.ndim == 3: # (B, L, V)
                return logits.masked_fill(~self.layer_masks.unsqueeze(0), float('-inf'))
            else: # (..., V) but which layer? This case usually doesn't happen with the unified logic
                return logits.masked_fill(~self.layer_masks.unsqueeze(0), float('-inf'))

    def encode(self, input_seq, attention_mask):
        batch_size, seq_len = input_seq.size()
        item_codes = self.get_item_codes(input_seq)
        
        offsets = torch.arange(self.num_code_layers, device=input_seq.device) * self.codebook_size
        semantic_ids = (item_codes + offsets).view(batch_size, -1)
        
        extended_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, self.num_code_layers).view(batch_size, -1)
        
        sem_emb_for_encoder = self.semantic_emb(semantic_ids)
        encoder_seq_len = semantic_ids.size(1)
        pos_indices = torch.arange(encoder_seq_len, dtype=torch.long, device=input_seq.device)
        pos_indices = pos_indices.unsqueeze(0).expand(batch_size, encoder_seq_len)
        
        pos_emb_enc = self.pos_emb(pos_indices)
        inputs_embeds = self.emb_dropout(sem_emb_for_encoder + pos_emb_enc)

        encoder_outputs = self.t5_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        return encoder_outputs, extended_attention_mask

    def forward(self, input_seq, attention_mask, target_item_ids=None, labels=None, K=1):
        """Unified forward pass for SFT and GRPO/GBPO."""
        # 1. Encode (Prompt is encoded only once regardless of mode)
        enc_out, enc_mask = self.encode(input_seq, attention_mask)
        
        # 2. Parallel expansion for GRPO/GBPO (K > 1)
        if K > 1:
            B_sub, S_enc, H = enc_out.last_hidden_state.shape
            expanded_hidden = enc_out.last_hidden_state.unsqueeze(1).expand(B_sub, K, S_enc, H).reshape(B_sub * K, S_enc, H)
            expanded_enc_mask = enc_mask.unsqueeze(1).expand(B_sub, K, S_enc).reshape(B_sub * K, S_enc)
            enc_out = BaseModelOutput(last_hidden_state=expanded_hidden)
            enc_mask = expanded_enc_mask

        # 3. Prepare labels and decoder inputs
        if target_item_ids is not None:
            if target_item_ids.ndim > 1: target_item_ids = target_item_ids.squeeze(-1)
            target_codes_raw = self.get_item_codes(target_item_ids)
            offsets = torch.arange(self.num_code_layers, device=target_item_ids.device) * self.codebook_size
            labels = target_codes_raw + offsets
        
        if labels is None:
            raise ValueError("Either target_item_ids or labels must be provided.")

        bos_token_tensor = torch.full((labels.size(0), 1), self.bos_token_id, dtype=torch.long, device=labels.device)
        decoder_input_ids = torch.cat([bos_token_tensor, labels[:, :-1]], dim=1) if self.num_code_layers > 1 else bos_token_tensor

        # 4. T5 Forward
        # In GRPO mode (K > 1), labels are usually not passed to t5_model to avoid internal loss calculation
        outputs = self.t5_model(
            encoder_outputs=enc_out,
            attention_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels if K == 1 else None,
            return_dict=True
        )
        return outputs.logits, outputs.loss

    @torch.no_grad()
    def predict(self, input_seq, attention_mask, beam_size, topk):
        self.eval()
        t5_encoder_output_obj, extended_encoder_attention_mask = self.encode(input_seq, attention_mask)
        num_return_sequences = min(beam_size, max(topk, 200))

        def strict_code_logits_processor(input_ids, scores):
            layer_idx = input_ids.shape[1] - 1
            if 0 <= layer_idx < self.num_code_layers:
                scores = self.apply_layer_mask(scores, layer_idx)
            return scores
        
        gen_config = GenerationConfig(
            max_length=self.num_code_layers + 1,
            min_length=self.num_code_layers + 1,
            num_beams=beam_size,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=self.bos_token_id,
            eos_token_id=self.bos_token_id,
            pad_token_id=self.bos_token_id,
        )

        generated_outputs = self.t5_model.generate(
            encoder_outputs=t5_encoder_output_obj,
            attention_mask=extended_encoder_attention_mask,
            generation_config=gen_config,
            logits_processor=[strict_code_logits_processor],
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        generated_ids = generated_outputs.sequences
        sequences_scores = getattr(generated_outputs, "sequences_scores", None)
        
        batch_size = input_seq.size(0)
        all_pred_items = []
        
        total_generated_sequences = 0
        valid_generated_sequences = 0
        
        # Process each sample in the batch
        for b in range(batch_size):
            start_idx = b * num_return_sequences
            end_idx = (b + 1) * num_return_sequences
            
            sample_gen_ids = generated_ids[start_idx:end_idx]
            sample_scores = sequences_scores[start_idx:end_idx] if sequences_scores is not None else None
            
            final_beams_data = []
            processed_item_ids = set()

            for i in range(sample_gen_ids.size(0)):
                total_generated_sequences += 1
                seq_global_ids = sample_gen_ids[i].tolist()
                score = sample_scores[i].item() if sample_scores is not None else -float(i)

                if len(seq_global_ids) == self.num_code_layers + 1 and seq_global_ids[0] == self.bos_token_id:
                    layer_relative_codes = []
                    for layer_idx in range(self.num_code_layers):
                        layer_relative_codes.append(seq_global_ids[layer_idx + 1] % self.codebook_size)
                    
                    code_tuple = tuple(layer_relative_codes)
                    item_id = self.code2item_dict.get(code_tuple, 0)
                    
                    # Check validity for stats
                    if item_id > 0:
                        valid_generated_sequences += 1
                        
                    if item_id > 0 and item_id not in processed_item_ids:
                        final_beams_data.append((item_id, score))
                        processed_item_ids.add(item_id)

            final_beams_data.sort(key=lambda x: x[1], reverse=True)
            pred_items_list = [x[0] for x in final_beams_data[:topk]]
            pred_items_tensor = torch.tensor(pred_items_list, device=input_seq.device, dtype=torch.long)
            
            if len(pred_items_tensor) < topk:
                padding = torch.zeros(topk - len(pred_items_tensor), dtype=torch.long, device=input_seq.device)
                pred_items_tensor = torch.cat([pred_items_tensor, padding])
            
            all_pred_items.append(pred_items_tensor)
            
        validity_rate = valid_generated_sequences / max(1, total_generated_sequences)
        return torch.stack(all_pred_items), validity_rate # Return stacked tensor (B, TopK) and validity

    @torch.no_grad()
    def generate_candidates(self, input_seq, attention_mask, num_candidates, strategy="rollout", temperature=1.0, use_trie=None):
        """Unified candidate generation using rollout or beam search."""
        self.eval()
        device = input_seq.device
        enc_out, enc_mask = self.encode(input_seq, attention_mask)
        
        # 保存“原始 B”的 encoder hidden 和 mask，避免被 generate() 改写后污染
        enc_hidden_base = enc_out.last_hidden_state
        enc_mask_base = enc_mask

        # Align temperature: transformers generate only uses temperature when sampling
        actual_temp = temperature if strategy == "rollout" else 1.0
        
        gen_config = GenerationConfig(
            max_length=self.num_code_layers + 1,
            min_length=self.num_code_layers + 1,
            num_return_sequences=num_candidates,
            decoder_start_token_id=self.bos_token_id,
            eos_token_id=self.bos_token_id,
            pad_token_id=self.bos_token_id,
            do_sample=(strategy == "rollout"),
            temperature=actual_temp,
            num_beams=num_candidates if strategy == "beam" else 1,
        )

        def strict_code_logits_processor(input_ids, scores):
            layer_idx = input_ids.shape[1] - 1
            if 0 <= layer_idx < self.num_code_layers:
                return self.apply_layer_mask(scores, layer_idx)
            return scores

        # Normalize override if provided, else use default mode
        effective_mode = self._normalize_trie_mode(use_trie) if use_trie is not None else self.use_trie_rl
        should_use_trie = (effective_mode != "no_trie")

        # 用一个新的对象喂给 generate，别让它改到你后面要用的 enc_out
        gen_out = self.t5_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden_base),
            attention_mask=enc_mask_base,
            generation_config=gen_config,
            logits_processor=[strict_code_logits_processor],
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn if should_use_trie else None,
            return_dict_in_generate=True,
        )
        
        codes = gen_out.sequences[:, 1:] # (B*K, L)
        codes_cpu = codes.cpu().numpy()
        sampled_items = torch.tensor([
            self.code2item_dict.get(tuple((codes_cpu[i] % self.codebook_size).tolist()), 0)
            for i in range(codes.size(0))
        ], device=device, dtype=torch.long)

        
        # 使用保存的 base 变量进行手动扩展，确保 B 永远来自原始 input_seq
        B = input_seq.size(0)
        K = num_candidates
        S_enc = enc_hidden_base.size(1)
        H = enc_hidden_base.size(2)
        
        expanded_hidden = enc_hidden_base.unsqueeze(1).expand(B, K, S_enc, H).reshape(B * K, S_enc, H)
        expanded_enc_mask = enc_mask_base.unsqueeze(1).expand(B, K, S_enc).reshape(B * K, S_enc)
        expanded_enc_out = BaseModelOutput(last_hidden_state=expanded_hidden)

        bos_token_tensor = torch.full((codes.size(0), 1), self.bos_token_id, dtype=torch.long, device=device)
        decoder_input_ids = torch.cat([bos_token_tensor, codes[:, :-1]], dim=1) if self.num_code_layers > 1 else bos_token_tensor

        outputs = self.t5_model(
            encoder_outputs=expanded_enc_out,
            attention_mask=expanded_enc_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        logits = outputs.logits
        token_log_probs = compute_token_log_probs_from_codes(logits, codes, self, temperature=actual_temp)
        return sampled_items, codes, token_log_probs.sum(dim=1), token_log_probs

