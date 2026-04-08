import collections
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import os
import argparse
import sys

# Helper functions for collision detection and resolution (adapted for numeric codes)
def check_collision_numeric(all_codes_tuples_list):
    tot_item = len(all_codes_tuples_list)
    tot_unique_codes = len(set(all_codes_tuples_list))
    return tot_item == tot_unique_codes

def get_indices_count_numeric(all_codes_tuples_list):
    indices_count = collections.defaultdict(int)
    for code_tuple in all_codes_tuples_list:
        indices_count[code_tuple] += 1
    return indices_count

def get_collision_item_groups_numeric(all_codes_tuples_list):
    code_tuple_to_ids = collections.defaultdict(list)
    for i, code_tuple in enumerate(all_codes_tuples_list):
        code_tuple_to_ids[code_tuple].append(i)

    collision_item_groups = []
    for _, ids_list in code_tuple_to_ids.items():
        if len(ids_list) > 1:
            collision_item_groups.append(ids_list)
    return collision_item_groups

# constrained_km function (adapted from generate_indices.py)
def constrained_km(data_np, n_clusters=10):
    from k_means_constrained import KMeansConstrained # Ensure this library is installed
    
    if len(data_np) == 0:
        return torch.empty((0, data_np.shape[-1]) if data_np.ndim > 1 else (0,)), []

    actual_n_clusters = min(n_clusters, len(data_np))
    if actual_n_clusters == 0:
         return torch.empty((0, data_np.shape[-1]) if data_np.ndim > 1 else (0,)), []
    if len(data_np) < actual_n_clusters:
        print(f"Warning: len(data_np)={len(data_np)} < actual_n_clusters={actual_n_clusters}. ConstrainedKMeans might fail. Attempting anyway.")
        # KMeansConstrained requires n_samples >= n_clusters. If not, it will raise error.
        # Fallback below will handle this.

    size_min = max(1, min(len(data_np) // (actual_n_clusters * 2), 10)) if actual_n_clusters > 0 else 1
    size_max = max(size_min, actual_n_clusters * 6)

    # Use random_state for reproducibility in KMeansConstrained
    clf = KMeansConstrained(n_clusters=actual_n_clusters, size_min=size_min, size_max=size_max, 
                            max_iter=10, n_init=10, n_jobs=1, verbose=False, random_state=0) # n_jobs=1 for wider compatibility
    try:
        clf.fit(data_np)
        centers = torch.from_numpy(clf.cluster_centers_).float()
        labels = torch.from_numpy(clf.labels_).long().tolist()
    except Exception as e:
        print(f"Constrained KMeans failed: {e}. Falling back to standard KMeans or simple assignment.")
        from sklearn.cluster import KMeans # Fallback
        if actual_n_clusters == 0: return torch.empty((0, data_np.shape[-1]) if data_np.ndim > 1 else (0,)), []
        
        # Ensure n_clusters for KMeans is not more than samples
        kmeans_n_clusters = min(actual_n_clusters, len(data_np))
        if kmeans_n_clusters == 0 : # Still possible if len(data_np) == 0 after initial checks
             return torch.empty((0, data_np.shape[-1]) if data_np.ndim > 1 else (0,)), []

        if kmeans_n_clusters == 1:
             centers_np = np.mean(data_np, axis=0, keepdims=True)
             labels_np = np.zeros(len(data_np), dtype=int)
        else:
            # For sklearn >= 1.2, n_init can be 'auto'. For older versions, use an int like 10.
            # Check scikit-learn version to set n_init appropriately.
            import sklearn
            if hasattr(sklearn, '__version__') and tuple(map(int, sklearn.__version__.split('.')[:2])) >= (1, 2):
                n_init_val = 'auto'
            else:
                n_init_val = 10            
            km = KMeans(n_clusters=kmeans_n_clusters, n_init=n_init_val, max_iter=10, random_state=0)
            km.fit(data_np)
            centers_np = km.cluster_centers_
            labels_np = km.labels_
        centers = torch.from_numpy(centers_np).float()
        labels = torch.from_numpy(labels_np).long().tolist()
    return centers, labels

def generate_mappings_direct(ckpt_file_path, output_dir_path):
    # Attempt to add RQ-VAE module path for imports
    try:
        # Assuming this script (4generate_sid_mappings.py) is in \'.../letter_data/\'
        # and \'RQ-VAE\' is a subdirectory containing \'datasets.py\' and \'models/rqvae.py\'
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        rq_vae_module_parent_dir = os.path.join(current_script_dir, "RQ-VAE")
        if rq_vae_module_parent_dir not in sys.path:
            sys.path.insert(0, rq_vae_module_parent_dir)
            print(f"Added {rq_vae_module_parent_dir} to sys.path for EmbDataset/RQVAE imports")
        
        from datasets import EmbDataset
        from models.rqvae import RQVAE
    except ImportError as e:
        print(f"Error importing EmbDataset or RQVAE: {e}")
        print("Please ensure these modules are in PYTHONPATH, their definitions are included in this script,")
        print(f"or that the path {rq_vae_module_parent_dir} is correct and accessible.")
        raise

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from: {ckpt_file_path}")
    ckpt = torch.load(ckpt_file_path, map_location=torch.device('cpu'), weights_only=False)
    args_ckpt = ckpt["args"]
    state_dict = ckpt["state_dict"]

    if not hasattr(args_ckpt, 'num_emb_list') or not args_ckpt.num_emb_list:
        raise ValueError("Checkpoint args does not contain 'num_emb_list' or it's empty.")
    num_code_layers = len(args_ckpt.num_emb_list)
    print(f"Number of code layers (from ckpt num_emb_list): {num_code_layers}")

    # Define output filenames. Using "4L" as in the original filenames.
    # If these names should be dynamic based on num_code_layers, adjust here.
    # For example: f\"code2item_rkmeans_{num_code_layers}L_llama.pkl\"
    code2item_output_filename = f"code2item_rkmeans_4L_llama.pkl"
    item2code_output_filename = f"item2code_rkmeans_4L_llama.npy"
    
    code2item_output_path = os.path.join(output_dir_path, code2item_output_filename)
    item2code_output_path = os.path.join(output_dir_path, item2code_output_filename)
    print(f"Output code2item path: {code2item_output_path}")
    print(f"Output item2code path: {item2code_output_path}")


    print(f"Loading data from: {args_ckpt.data_path}")
    dataset = EmbDataset(args_ckpt.data_path)
    # Use a default for num_workers if not in args_ckpt, e.g., 0 for main process, or 2/4 if resources allow
    num_workers_val = getattr(args_ckpt, 'num_workers', 0 if os.name == 'nt' else 2) # 0 for Windows, 2 for others as a default
    data_loader = DataLoader(dataset, num_workers=num_workers_val,
                             batch_size=getattr(args_ckpt, 'batch_size', 64), shuffle=False, pin_memory=True)

    print("Initializing model...")
    model = RQVAE(in_dim=dataset.dim, num_emb_list=args_ckpt.num_emb_list, e_dim=args_ckpt.e_dim,
                  layers=args_ckpt.layers, dropout_prob=args_ckpt.dropout_prob, bn=args_ckpt.bn,
                  loss_type=args_ckpt.loss_type, quant_loss_weight=args_ckpt.quant_loss_weight,
                  kmeans_init=args_ckpt.kmeans_init, kmeans_iters=args_ckpt.kmeans_iters,
                  sk_epsilons=args_ckpt.sk_epsilons, sk_iters=args_ckpt.sk_iters)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Preparing labels for constrained_km (if used by model.get_indices)...")
    labels_for_model = {str(i): [] for i in range(num_code_layers)}
    if hasattr(model, 'rq') and hasattr(model.rq, 'vq_layers'):
        model_embs = []
        for vq_layer_idx, vq_layer in enumerate(model.rq.vq_layers):
            if hasattr(vq_layer, 'embedding') and hasattr(vq_layer.embedding, 'weight'):
                model_embs.append(vq_layer.embedding.weight.cpu().detach().numpy())
            else:
                print(f"Warning: VQ layer {vq_layer_idx} does not have 'embedding.weight'. Skipping for label generation.")


        for idx, emb_np in enumerate(model_embs):
            if idx < num_code_layers: 
                # print(f\"Running constrained_km for layer {idx} on emb shape {emb_np.shape}\")
                _, label_list = constrained_km(emb_np, n_clusters=10) 
                labels_for_model[str(idx)] = label_list
            else:
                break 
    else:
        print("Warning: model.rq.vq_layers not found or structured as expected. Skipping label preparation.")

    print("Generating initial codes for all items...")
    all_item_codes_list_of_lists = []
    for batch_data in tqdm(data_loader, desc="Initial code generation"):
        item_tensors = batch_data[0].to(device)
        try:
            batch_indices = model.get_indices(item_tensors, labels=labels_for_model, use_sk=False)
        except TypeError: 
            print("Warning: model.get_indices with 'labels' failed. Trying without 'labels'.")
            batch_indices = model.get_indices(item_tensors, use_sk=False)
        
        batch_indices_np = batch_indices.view(-1, num_code_layers).cpu().numpy()
        for code_vector in batch_indices_np:
            all_item_codes_list_of_lists.append(list(map(int, code_vector))) 

    current_item_codes_tuples = [tuple(codes) for codes in all_item_codes_list_of_lists]
    del all_item_codes_list_of_lists

    print("Starting conflict resolution...")
    if hasattr(model, 'rq') and hasattr(model.rq, 'vq_layers') and model.rq.vq_layers:
        for vq_layer in model.rq.vq_layers[:-1]:
            if hasattr(vq_layer, 'sk_epsilon'): vq_layer.sk_epsilon = 0.0
        
        last_vq_layer = model.rq.vq_layers[-1]
        if hasattr(last_vq_layer, 'sk_epsilon'):
            default_sk_epsilon = getattr(args_ckpt, 'default_sk_epsilon_last_layer', 0.003)
            if last_vq_layer.sk_epsilon == 0.0: 
                 last_vq_layer.sk_epsilon = default_sk_epsilon
    else:
        print("Warning: Could not set sk_epsilon due to model structure issues.")


    tt = 0
    max_iterations = 20
    while tt < max_iterations:
        if check_collision_numeric(current_item_codes_tuples):
            print(f"No collisions found after {tt} iterations.")
            break
        
        collision_groups = get_collision_item_groups_numeric(current_item_codes_tuples)
        if not collision_groups:
            # This case implies check_collision_numeric found collisions, but get_collision_item_groups_numeric did not find groups.
            # This could happen if, for example, all items are unique but the total count doesn't match an expected value elsewhere.
            # However, based on their definitions, if check_collision_numeric is false, get_collision_item_groups_numeric should return non-empty.
            print(f"Iteration {tt+1}: No collision groups returned, though check_collision indicated conflict. This is unexpected. Breaking.")
            break 
        
        print(f"Iteration {tt+1}/{max_iterations}: Found {len(collision_groups)} groups with conflicts.")
        
        for item_ids_in_group in tqdm(collision_groups, desc=f"Resolving conflicts (iter {tt+1})"):
            item_ids_in_group_int = [int(id_val) for id_val in item_ids_in_group]
            
            # EmbDataset usually returns a tuple (data_tensor, other_info)
            # We need to handle if it returns just the tensor or a list/tuple of tensors
            colliding_item_data_batch_output = dataset[item_ids_in_group_int]
            if isinstance(colliding_item_data_batch_output, tuple):
                colliding_item_tensors = colliding_item_data_batch_output[0].to(device)
            else: # Assuming it's the tensor itself
                colliding_item_tensors = colliding_item_data_batch_output.to(device)

            try:
                new_indices_for_group = model.get_indices(colliding_item_tensors, labels=labels_for_model, use_sk=True)
            except TypeError:
                new_indices_for_group = model.get_indices(colliding_item_tensors, use_sk=True)

            new_indices_np = new_indices_for_group.view(-1, num_code_layers).cpu().numpy()

            for i, original_item_id in enumerate(item_ids_in_group_int):
                current_item_codes_tuples[original_item_id] = tuple(map(int, new_indices_np[i]))
        tt += 1
    
    if tt == max_iterations and not check_collision_numeric(current_item_codes_tuples):
        print(f"Warning: Conflict resolution reached max iterations ({max_iterations}) but collisions might still exist.")

    num_total_items = len(dataset)
    item2code = np.zeros((num_total_items + 1, num_code_layers), dtype=np.int32)
    code2item = {}

    print("Populating item2code and code2item mappings...")
    for original_item_id in tqdm(range(num_total_items), desc="Finalizing mappings"):
        # Ensure original_item_id is within bounds of current_item_codes_tuples
        if original_item_id >= len(current_item_codes_tuples):
            print(f"Warning: original_item_id {original_item_id} is out of bounds for current_item_codes_tuples (len {len(current_item_codes_tuples)}). Skipping.")
            continue
            
        item_id_1_indexed = original_item_id + 1
        code_tuple = current_item_codes_tuples[original_item_id]

        item2code[item_id_1_indexed] = np.array(code_tuple, dtype=np.int32)
        
        if code_tuple in code2item and code2item[code_tuple] != item_id_1_indexed:
            print(f"Warning: Duplicate code tuple {code_tuple} for 1-indexed item {item_id_1_indexed}. "
                  f"Previously mapped to item {code2item[code_tuple]}. Overwriting.")
        code2item[code_tuple] = item_id_1_indexed
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"Created output directory: {output_dir_path}")

    print(f"\nSaving code2item mapping to: {code2item_output_path}")
    with open(code2item_output_path, 'wb') as f_pkl:
        pickle.dump(code2item, f_pkl)
    print(f"Saved code2item with {len(code2item)} entries.")

    print(f"Saving item2code mapping to: {item2code_output_path}")
    np.save(item2code_output_path, item2code)
    print(f"Saved item2code with shape {item2code.shape}.")

    print("\n--- SID Mapping Generation Statistics (Direct Method) ---")
    indices_stats = get_indices_count_numeric(current_item_codes_tuples)
    max_conflicts_val = max(indices_stats.values()) if indices_stats else 0
    print(f"Max items sharing a single code tuple: {max_conflicts_val}")

    tot_items_map = len(current_item_codes_tuples)
    tot_unique_codes_map = len(set(current_item_codes_tuples)) # Count unique tuples
    collision_rate_val = ((tot_items_map - tot_unique_codes_map) / tot_items_map) if tot_items_map > 0 else 0.0
    print(f"Collision Rate after resolution: {collision_rate_val:.6f} ({tot_items_map - tot_unique_codes_map} collisions out of {tot_items_map} items)")
    
    print(f"Max original item ID processed (0-indexed): {num_total_items - 1 if num_total_items > 0 else -1}")
    print(f"Max 1-indexed item ID in mappings: {num_total_items}")
    print(f"Number of unique code tuples in code2item dictionary: {len(code2item)}")
    print("--- End of Statistics ---")

if __name__ == "__main__":
    # These paths were originally in 4generate_sid_mappings.py
    default_in_ckpt_file = ""
    default_output_dir = ""

    parser = argparse.ArgumentParser(description="Generate code2item.pkl and item2code.npy directly from RQ-VAE checkpoint.")
    parser.add_argument('--ckpt_path', type=str, default=default_in_ckpt_file,
                        help='Path to the RQ-VAE model checkpoint (.pth).')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Directory to save the output .pkl and .npy files.')
    
    cli_args = parser.parse_args()

    generate_mappings_direct(cli_args.ckpt_path, cli_args.output_dir)