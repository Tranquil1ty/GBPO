# python 2amazon_text_emb.py --root /pub/dengjiaxin03/code/recogpt/git_code/kdd_25_pub/letter_preprocess/preocessed_data --dataset Beauty --plm_checkpoint /pub/dengjiaxin03/code/recogpt/git_code/llama3/Meta-Llama-3.1-8B/Llama-3.1-8B
import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils2 import *
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModel


def load_data(args):

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    return item2feature

def generate_text(item2feature, features):
    item_text_list = []

    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())

        item_text_list.append([int(item), text])

    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)

    item2feature = load_data(args)
    # load item text and clean
    item_text_list = generate_text(item2feature, ['title', 'description'])
    # item_text_list = generate_text(item2feature, ['title'])
    # return: list of (item_ID, cleaned_item_text)
    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)

    if not item_text_list:
        print("No items to process.")
        np.save(os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy"), np.array([]))
        return

    embed_dim_fallback = None
    try:
        if isinstance(model, torch.nn.DataParallel):
            embed_dim_fallback = model.module.config.hidden_size
        else:
            embed_dim_fallback = model.config.hidden_size
    except AttributeError:
        print("Warning: Could not determine model's hidden_size for fallback embeddings for items with no text.")

    # 1. Prepare all sentences from all items and track their origin
    all_sentences_for_encoding = []
    # item_processing_info will store dicts:
    # {"original_item_idx": original_idx, "start_idx_in_flat_list": start_idx, "num_sentences": count}
    item_processing_info = []

    print("Preprocessing item texts and preparing sentences for batch encoding...")
    for item_idx, (_, item_fields_list) in enumerate(tqdm(item_text_list, desc="Preprocessing Items")):
        start_idx_for_this_item_in_flat_list = len(all_sentences_for_encoding)
        num_sentences_for_this_item = 0
        
        processed_item_sentences = []
        if item_fields_list:
            for text_sentence in item_fields_list:
                sentence_to_encode = text_sentence.strip() # Ensure stripped early
                if word_drop_ratio > 0 and sentence_to_encode: # Apply word drop only if sentence is not empty
                    new_sent_parts = []
                    original_sent_parts = sentence_to_encode.split(' ')
                    for wd in original_sent_parts:
                        if random.random() > word_drop_ratio:
                            new_sent_parts.append(wd)
                    sentence_to_encode = ' '.join(new_sent_parts)
                
                if sentence_to_encode: # Only add non-empty sentences after potential word drop
                    processed_item_sentences.append(sentence_to_encode)
        
        if not processed_item_sentences: # Item had no text fields, or all became empty
            all_sentences_for_encoding.append("") # Add a single placeholder sentence
            num_sentences_for_this_item = 1
        else:
            all_sentences_for_encoding.extend(processed_item_sentences)
            num_sentences_for_this_item = len(processed_item_sentences)
        
        item_processing_info.append({
            "original_item_idx": item_idx, 
            "start_idx_in_flat_list": start_idx_for_this_item_in_flat_list, 
            "num_sentences": num_sentences_for_this_item
        })

    if not all_sentences_for_encoding:
        print("Warning: No sentences to encode after processing all items. All items might have been empty.")
        num_items = len(item_text_list)
        if num_items > 0 and embed_dim_fallback is not None:
             final_embeddings_numpy = np.zeros((num_items, embed_dim_fallback), dtype=np.float32)
        else:
             final_embeddings_numpy = np.array([])
        
        output_file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
        np.save(output_file, final_embeddings_numpy)
        print(f"Embeddings (zeros or empty) saved to {output_file}")
        return

    # 2. Process all collected sentences in batches using the model
    # args.batch_size now refers to the batch of sentences fed to the model
    sentence_batch_size = args.batch_size 
    all_sentence_embeddings_from_model_list = []

    print(f"Encoding a total of {len(all_sentences_for_encoding)} sentences in batches of {sentence_batch_size} across available GPUs...")
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for i in tqdm(range(0, len(all_sentences_for_encoding), sentence_batch_size), desc="Encoding Sentences (GPU Batches)"):
            batch_of_individual_sentences = all_sentences_for_encoding[i : i + sentence_batch_size]
            
            if not batch_of_individual_sentences: # Should not happen if outer list is not empty
                continue

            encoded_input = tokenizer(batch_of_individual_sentences, max_length=args.max_sent_len,
                                      truncation=True, return_tensors='pt', padding="longest").to(args.device)
            
            outputs = model(input_ids=encoded_input.input_ids,
                            attention_mask=encoded_input.attention_mask)
            
            last_hidden = outputs.last_hidden_state
            attention_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand_as(last_hidden)
            sum_hidden_states = (last_hidden * attention_mask_expanded).sum(dim=1)
            sum_attention_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9) # Avoid division by zero for empty sentences
            
            mean_pooled_embs_for_batch = (sum_hidden_states / sum_attention_mask).detach().cpu()
            all_sentence_embeddings_from_model_list.append(mean_pooled_embs_for_batch)

    if not all_sentence_embeddings_from_model_list:
        print("Error: Model produced no embeddings, though sentences were provided.")
        num_items = len(item_text_list)
        if num_items > 0 and embed_dim_fallback is not None:
             final_embeddings_numpy = np.zeros((num_items, embed_dim_fallback), dtype=np.float32)
        else:
             final_embeddings_numpy = np.array([])
        # ... (save and return, as above)
        output_file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
        np.save(output_file, final_embeddings_numpy)
        print(f"Embeddings (zeros or empty) saved to {output_file}")
        return
        
    flat_sentence_embeddings_tensor = torch.cat(all_sentence_embeddings_from_model_list, dim=0)

    # 3. Reconstruct item embeddings by averaging their respective sentence embeddings
    final_item_embeddings_list_ordered = []
    print("Reconstructing item embeddings from sentence embeddings...")
    for info in tqdm(item_processing_info, desc="Reconstructing Item Embeddings"):
        item_sentence_start_idx = info["start_idx_in_flat_list"]
        num_sents_for_this_item = info["num_sentences"]
        
        # Retrieve all sentence embeddings for the current item
        # num_sents_for_this_item will be at least 1 (due to placeholder logic)
        item_specific_sentence_embeddings = flat_sentence_embeddings_tensor[item_sentence_start_idx : item_sentence_start_idx + num_sents_for_this_item]
        
        # Average these sentence embeddings to get the final item embedding
        # If num_sents_for_this_item is 1 (e.g. placeholder ""), this is just that embedding
        item_final_embedding = item_specific_sentence_embeddings.mean(dim=0, keepdim=True) # keepdim for torch.cat
        final_item_embeddings_list_ordered.append(item_final_embedding)

    if not final_item_embeddings_list_ordered:
        # This case should ideally be covered if item_text_list was not empty.
        final_embeddings_numpy = np.array([])
        print("Warning: No final item embeddings were reconstructed, item_text_list might have been empty initially.")
    else:
        final_embeddings_numpy = torch.cat(final_item_embeddings_list_ordered, dim=0).numpy()

    print('Final Embeddings shape: ', final_embeddings_numpy.shape)

    output_file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
    np.save(output_file, final_embeddings_numpy)
    print(f"Embeddings saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default="")
    parser.add_argument('--gpu_id', type=str, default="0", help='ID of running GPU(s), e.g., "0" or "0,1,2"') # Modified to accept string for multi-gpu
    parser.add_argument('--plm_name', type=str, default='llama')
    parser.add_argument('--plm_checkpoint', type=str,
                        default='')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for sentences sent to the model for parallel embedding generation.') # Clarified help
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    #args.root = os.path.join(args.root, args.dataset)

    if "," in args.gpu_id: # Multi-GPU case for DataParallel
        print(f"Multiple GPUs specified: {args.gpu_id}. DataParallel will be attempted.")
        # DataParallel determines device distribution. For inputs, usually cuda:0 is the primary.
        # We'll set args.device to cuda:0 if CUDA is available and multiple GPUs are listed.
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0")
            # Set CUDA_VISIBLE_DEVICES for torch.nn.DataParallel to see the right GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
            print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_id}")
            num_visible_gpus = len(args.gpu_id.split(','))
            if torch.cuda.device_count() < num_visible_gpus:
                print(f"Warning: CUDA_VISIBLE_DEVICES is {args.gpu_id}, but PyTorch only sees {torch.cuda.device_count()} CUDA devices. Check GPU availability and drivers.")
            print(f"Primary device for model and data: {args.device}")
        else:
            print("Warning: Multiple GPUs specified, but CUDA is not available. Using CPU.")
            args.device = torch.device("cpu")
    else: # Single GPU or CPU
        try:
            gpu_idx = int(args.gpu_id)
            if gpu_idx < 0 : # CPU
                 args.device = torch.device("cpu")
                 print("Using CPU as per gpu_id < 0.")
            else: # Specific single GPU
                if torch.cuda.is_available():
                    if gpu_idx < torch.cuda.device_count():
                        args.device = torch.device(f"cuda:{gpu_idx}")
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx) # Ensure only this GPU is visible
                        print(f"CUDA_VISIBLE_DEVICES set to: {gpu_idx}")
                    else:
                        print(f"Warning: GPU ID {gpu_idx} requested, but only {torch.cuda.device_count()} GPUs available. Using cuda:0 if available, else CPU.")
                        args.device = torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
                else:
                    print(f"Warning: GPU ID {gpu_idx} requested, but CUDA is not available. Using CPU.")
                    args.device = torch.device("cpu")
        except ValueError:
            print(f"Invalid gpu_id: {args.gpu_id}. Using CPU.")
            args.device = torch.device("cpu")
    
    print(f"Effective device for model and data: {args.device}")


    item_text_list = preprocess_text(args) # This should come after device setup if it uses GPU, but it seems CPU-bound.

    plm_tokenizer, plm_model_original = load_plm(args.plm_checkpoint)
    if plm_tokenizer.pad_token_id is None:
        # Common practice for models like LLaMA if pad_token is not set
        plm_tokenizer.pad_token_id = plm_tokenizer.eos_token_id if plm_tokenizer.eos_token_id is not None else 0 # Fallback to 0 if eos_token_id is also None
        print(f"Tokenizer pad_token_id was None, set to: {plm_tokenizer.pad_token_id}")


    plm_model = plm_model_original 
    
    plm_model.to(args.device)
    print(f"Initial model moved to: {args.device}")

    if str(args.device).startswith("cuda") and "," in args.gpu_id and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Attempting to use torch.nn.DataParallel across GPUs visible via CUDA_VISIBLE_DEVICES ({os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}). PyTorch sees {torch.cuda.device_count()} CUDA device(s).")
        plm_model = torch.nn.DataParallel(plm_model) # device_ids default to all visible GPUs
        print(f"Model wrapped with torch.nn.DataParallel. It will run on primary device {args.device} and scatter to other visible GPUs.")
    elif str(args.device).startswith("cuda"):
        print(f"Using single GPU: {args.device}. Model is already on this device.")
    else:
        print(f"Running on CPU (device: {args.device}). Model is already on this device.")

    plm_model.eval() # Ensure model is in evaluation mode

    generate_item_embedding(args, item_text_list, plm_tokenizer,
                            plm_model, word_drop_ratio=args.word_drop_ratio)


