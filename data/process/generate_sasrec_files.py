import json
import numpy as np
import os
import argparse
from tqdm import tqdm


def pad_history(history_ids, max_len, pad_value=0):
    """
    Pads or truncates a history sequence to a specified maximum length.
    If the history is longer than max_len, it takes the last max_len items.
    If shorter, it prepends with pad_value.
    """
    if len(history_ids) > max_len:
        return history_ids[-max_len:]
    elif len(history_ids) < max_len:
        return [pad_value] * (max_len - len(history_ids)) + history_ids
    return history_ids


def generate_sasrec_files(inter_file_path, output_dir, max_seq_len):
    """
    Generates train_x_loo.npy, train_y_loo.npy, train_user_id_loo.npy,
             valid_x_loo.npy, valid_y_loo.npy, valid_user_id_loo.npy,
             test_x_loo.npy, test_y_loo.npy, test_user_id_loo.npy
    for SASRec.py. Item IDs are converted to be 1-indexed.
    SASRec uses 0 for padding.
    """
    print(f"Loading interaction data from: {inter_file_path}")
    try:
        with open(inter_file_path, 'r') as f:
            user_interactions = json.load(f)  # Expected format: {"uid": [item_id1, item_id2, ...]}
    except FileNotFoundError:
        print(f"Error: Input interaction file not found at {inter_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {inter_file_path}")
        return

    train_x_list, train_y_list, train_user_list = [], [], []
    valid_x_list, valid_y_list, valid_user_list = [], [], []
    test_x_list, test_y_list, test_user_list = [], [], []

    print("Processing user interactions...")
    for user_id, original_item_ids in tqdm(user_interactions.items()):
        if not isinstance(original_item_ids, list) or len(original_item_ids) < 3:
            # Need at least 3 items for train/valid/test split
            continue

        # Convert user_id
        try:
            user_id_value = int(user_id)
        except (ValueError, TypeError):
            user_id_value = user_id

        # Convert original 0-indexed item IDs to 1-indexed for SASRec
        item_ids = [item_id for item_id in original_item_ids]

        # --- Generate test data (LOO: last item) ---
        history_test = item_ids[:-1]
        target_test = item_ids[-1]
        if history_test:
            padded_history_test = pad_history(history_test, max_seq_len)
            test_x_list.append(padded_history_test)
            test_y_list.append(target_test)
            test_user_list.append(user_id_value)

        # --- Generate validation data (LOO: second last item) ---
        if len(item_ids) >= 2:
            history_valid = item_ids[:-2]
            target_valid = item_ids[-2]
            if history_valid:
                padded_history_valid = pad_history(history_valid, max_seq_len)
                valid_x_list.append(padded_history_valid)
                valid_y_list.append(target_valid)
                valid_user_list.append(user_id_value)

        # --- Generate training data (sliding window up to the third last) ---
        train_sequence_candidate = item_ids[:-2]
        if len(train_sequence_candidate) >= 2:
            for i in range(1, len(train_sequence_candidate)):
                history_train = train_sequence_candidate[:i]
                target_train = train_sequence_candidate[i]
                padded_history_train = pad_history(history_train, max_seq_len)
                train_x_list.append(padded_history_train)
                train_y_list.append(target_train)
                train_user_list.append(user_id_value)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Convert to numpy arrays
    train_x_np = np.array(train_x_list, dtype=np.int64) if train_x_list else np.zeros((0, max_seq_len), dtype=np.int64)
    train_y_np = np.array(train_y_list, dtype=np.int64)
    train_user_np = np.array(train_user_list, dtype=np.int64)

    valid_x_np = np.array(valid_x_list, dtype=np.int64) if valid_x_list else np.zeros((0, max_seq_len), dtype=np.int64)
    valid_y_np = np.array(valid_y_list, dtype=np.int64)
    valid_user_np = np.array(valid_user_list, dtype=np.int64)

    test_x_np = np.array(test_x_list, dtype=np.int64) if test_x_list else np.zeros((0, max_seq_len), dtype=np.int64)
    test_y_np = np.array(test_y_list, dtype=np.int64)
    test_user_np = np.array(test_user_list, dtype=np.int64)

    # Save .npy files
    train_x_file = os.path.join(output_dir, "train_x_loo.npy")
    train_y_file = os.path.join(output_dir, "train_y_loo.npy")
    train_user_file = os.path.join(output_dir, "train_user_id_loo.npy")

    valid_x_file = os.path.join(output_dir, "valid_x_loo.npy")
    valid_y_file = os.path.join(output_dir, "valid_y_loo.npy")
    valid_user_file = os.path.join(output_dir, "valid_user_id_loo.npy")

    test_x_file = os.path.join(output_dir, "test_x_loo.npy")
    test_y_file = os.path.join(output_dir, "test_y_loo.npy")
    test_user_file = os.path.join(output_dir, "test_user_id_loo.npy")

    np.save(train_x_file, train_x_np)
    print(f"Saved train_x_loo.npy to {train_x_file} with shape {train_x_np.shape}")

    np.save(train_y_file, train_y_np)
    print(f"Saved train_y_loo.npy to {train_y_file} with shape {train_y_np.shape}")

    np.save(train_user_file, train_user_np)
    print(f"Saved train_user_id_loo.npy to {train_user_file} with shape {train_user_np.shape}")

    np.save(valid_x_file, valid_x_np)
    print(f"Saved valid_x_loo.npy to {valid_x_file} with shape {valid_x_np.shape}")

    np.save(valid_y_file, valid_y_np)
    print(f"Saved valid_y_loo.npy to {valid_y_file} with shape {valid_y_np.shape}")

    np.save(valid_user_file, valid_user_np)
    print(f"Saved valid_user_id_loo.npy to {valid_user_file} with shape {valid_user_np.shape}")

    np.save(test_x_file, test_x_np)
    print(f"Saved test_x_loo.npy to {test_x_file} with shape {test_x_np.shape}")

    np.save(test_y_file, test_y_np)
    print(f"Saved test_y_loo.npy to {test_y_file} with shape {test_y_np.shape}")

    np.save(test_user_file, test_user_np)
    print(f"Saved test_user_id_loo.npy to {test_user_file} with shape {test_user_np.shape}")

    # --- Statistics ---
    num_users = len(user_interactions)

    all_item_ids_set = set()
    for x_arr, y_arr in [(train_x_np, train_y_np), (valid_x_np, valid_y_np), (test_x_np, test_y_np)]:
        if x_arr.size > 0:
            all_item_ids_set.update(np.unique(x_arr[x_arr != 0]))
        if y_arr.size > 0:
            all_item_ids_set.update(np.unique(y_arr[y_arr != 0]))

    num_unique_items = len(all_item_ids_set)

    max_id = 0
    for arr in [train_x_np, train_y_np, valid_x_np, valid_y_np, test_x_np, test_y_np]:
        if arr.size > 0:
            max_id = max(max_id, np.max(arr))

    print("\n--- Data Statistics ---")
    print(f"Total users processed from input file: {num_users}")
    print(f"Users skipped due to insufficient interactions (less than 3 items): "
          f"{num_users - (len(set(u for u, items in user_interactions.items() if isinstance(items, list) and len(items) >= 3)))}")

    if max_id > 0:
        print(f"Max item ID found (1-indexed, used for num_items in model): {max_id}")
        print(f"Number of unique item IDs in generated datasets (1-indexed): {num_unique_items}")
        print(f"The 'num_items' parameter for the SASRec model should typically be this maximum ID value ({max_id}).")
        print(f"The item embedding layer in SASRec is usually nn.Embedding(num_items + 1, ..., padding_idx=0),")
        print(f"which means it can handle item IDs from 0 (padding) up to num_items (e.g., {max_id}).")
    else:
        print("Data generation complete, but no valid items were processed or found in generated arrays.")
    print("--- End of Statistics ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for SASRec model from interaction file.")
    parser.add_argument(
        '--inter_file',
        type=str,
        default="",
        help='Path to the input interaction file (e.g., Beauty.inter.json where item IDs are 0-indexed)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="",
        help='Directory to save the output .npy files'
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=20,
        help='Maximum sequence length for history padding/truncation. '
             'This should match the `max_len` parameter of the SASRec model.'
    )

    args = parser.parse_args()
    generate_sasrec_files(args.inter_file, args.output_dir, args.max_seq_len)
