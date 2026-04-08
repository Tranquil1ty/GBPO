import torch
from typing import Dict, List, Callable

class TrieNode:
    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.is_end_of_sequence: bool = False

class Trie:
    def __init__(self, sequences: List[List[int]] = []):
        self.root = TrieNode()
        if sequences:
            for sequence in sequences:
                self._add_to_trie(sequence)

    def _add_to_trie(self, sequence: List[int]):
        node = self.root
        for token_id in sequence:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.is_end_of_sequence = True

    def get_allowed_next_tokens(self, prefix_sequence: List[int]) -> List[int]:
        node = self.root
        for token_id in prefix_sequence:
            if token_id in node.children:
                node = node.children[token_id]
            else:
                return []
        return list(node.children.keys())

    def get_fsa_tables(self, vocab_size: int, device: torch.device):
        """Convert Trie into state tables for vectorized GPU masking."""
        all_nodes = []
        node_to_id = {}

        def collect(node):
            if id(node) not in node_to_id:
                node_to_id[id(node)] = len(all_nodes)
                all_nodes.append(node)
                for child in node.children.values():
                    collect(child)
        
        collect(self.root)
        num_states = len(all_nodes)
        
        # transition_table: (num_states, vocab_size) -> next_state_id
        # allowed_mask_table: (num_states, vocab_size) -> bool (True for blocked)
        transitions = torch.zeros((num_states, vocab_size), dtype=torch.long, device=device)
        masks = torch.ones((num_states, vocab_size), dtype=torch.bool, device=device)
        
        for node_id, node in enumerate(all_nodes):
            for token_id, child_node in node.children.items():
                if token_id < vocab_size:
                    transitions[node_id, token_id] = node_to_id[id(child_node)]
                    masks[node_id, token_id] = False
        
        return transitions, masks, node_to_id[id(self.root)]

def tiger_prefix_allowed_tokens_fn_factory(trie: Trie, codebook_size: int, num_code_layers: int, bos_token_id: int) -> Callable[[int, torch.Tensor], List[int]]:
    def prefix_allowed_tokens(batch_id: int, sentence_tensor: torch.Tensor) -> List[int]:
        prefix_sequence = sentence_tensor.tolist()
        current_prediction_step = len(prefix_sequence) - 1 

        allowed_global_tokens_from_trie = trie.get_allowed_next_tokens(prefix_sequence)
        
        layer_vocab_start_global_id = current_prediction_step * codebook_size
        layer_vocab_end_global_id = (current_prediction_step + 1) * codebook_size

        if not allowed_global_tokens_from_trie:
            return list(range(layer_vocab_start_global_id, layer_vocab_end_global_id))

        valid_global_tokens_for_layer = [
            tid for tid in allowed_global_tokens_from_trie 
            if layer_vocab_start_global_id <= tid < layer_vocab_end_global_id
        ]
        
        if not valid_global_tokens_for_layer:
            return list(range(layer_vocab_start_global_id, layer_vocab_end_global_id))

        return valid_global_tokens_for_layer

    return prefix_allowed_tokens

