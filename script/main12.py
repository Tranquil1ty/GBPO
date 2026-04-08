import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import random
import pickle
import sys
import argparse
from typing import Dict, List, Callable, Optional, Tuple
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- Path Setup ---
# RL was moved from onerec/script/RL to onerec/RL. 
# We add the parent directory (onerec root) to sys.path to find RL and other packages.
current_dir = os.path.dirname(os.path.abspath(__file__))
onerec_root = os.path.abspath(os.path.join(current_dir, ".."))
din_functions = os.path.join(onerec_root, "DIN/functions")

if onerec_root not in sys.path:
    sys.path.append(onerec_root)
if din_functions not in sys.path:
    sys.path.append(din_functions)

try:
    from prediction import load_model_from_checkpoint
    from model import TIGER
    from trainer import Trainer, RL_MODES
except ImportError as e:
    print(f"Warning: Import error - {e}. Check if RL, model.py, trainer.py and DIN/functions are in the correct location relative to script/.")

class Config:
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k.upper(), v)

        # 动态使用传入的数据集名称
        self.PRETRAINED_PATH = self.PRETRAINED_PATH or os.path.join(self.BASE_PATH, "checkpoints12", self.DATASET, "sft_checkpoints/best_checkpoint.pth")
        self.DATA_DIR_LOO = self.DATA_DIR_LOO or os.path.join(self.BASE_PATH, "data/processed_data", self.DATASET, "processed_loo")
        self.DIN_SAVE_PATH = self.DIN_SAVE_PATH or os.path.join(self.BASE_PATH, "DIN/checkpoint", self.DATASET, "best_model_fixed.pth")
        self.K_LIST_EVAL = [5, 10]
        self.EVAL_BEAM_SIZE = 20

        self.PADDING_ITEM_ID = 0
        self.NUM_ITEMS = 0 
        self.USE_DEEPCTR_REWARDS = True


def load_loo_data(data_dir):
    print(f"Loading LOO data from: {data_dir}")
    files = ['train_x_loo.npy', 'train_y_loo.npy', 'valid_x_loo.npy', 'valid_y_loo.npy', 'test_x_loo.npy', 'test_y_loo.npy']
    paths = [os.path.join(data_dir, f) for f in files]
    
    data = []
    for p in paths:
        if not os.path.exists(p): raise FileNotFoundError(f"{p} not found.")
        data.append(np.load(p))
    
    train_x, train_y, valid_x, valid_y, test_x, test_y = data
    
    # Fix dimensions if needed
    if train_y.ndim > 1: train_y = train_y[:,-1]
    if valid_y.ndim > 1: valid_y = valid_y[:,-1]
    if test_y.ndim > 1: test_y = test_y[:,-1]
    
    all_items = np.concatenate([train_x.flatten(), train_y, valid_x.flatten(), valid_y, test_x.flatten(), test_y])
    num_items = int(np.max(all_items)) if all_items.size > 0 else 0
    return train_x, train_y, valid_x, valid_y, test_x, test_y, num_items

# --- Trainer Class moved to trainer.py ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty", help="Dataset name")
    parser.add_argument("--mode", type=str, default="SFT", choices=["SFT"] + list(RL_MODES))
    parser.add_argument("--base_path", type=str, default="/project-share/zhengpengfei/project/public_data/model/onerec_sid/onerec/amazon-toys_update")
    parser.add_argument("--t5_model_path", type=str, default="/project-share/zhengpengfei/project/public_data/model/google_t5/t5-base")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model_max_len_loo", type=int, default=20)
    
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--grpo_loss_weight", type=float, default=1.0)
    parser.add_argument("--sft_loss_weight", type=float, default=None, help="Weight for SFT loss. If not set, defaults per mode (GRPO/GSPO: 1.0, GBPO: 0.1)")
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--rollout_batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ref_update_interval", type=int, default=1)
    parser.add_argument("--on_policy", action="store_true")
    parser.add_argument("--gen_strategy", type=str, default="rollout", choices=["rollout", "beam"],
                        help="Strategy for generating candidates: rollout (stochastic) or beam (search)")

    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_beam_size", type=int, default=256)
    parser.add_argument("--sft_patience", type=int, default=20)
    parser.add_argument("--grpo_patience", type=int, default=100)
    parser.add_argument("--reward_topk", type=int, default=5, help="Number of reward thresholds (top-k)")
    parser.add_argument("--verbose", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--use_trie", type=str, default="no_trie", choices=["no_trie", "trie_gen", "trie_all"],
                        help="Trie mode: no_trie (none), trie_gen (generation only), trie_all (gen + logit)")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 mixed precision training")

    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--data_dir_loo", type=str, default="")
    parser.add_argument("--din_save_path", type=str, default="")
    
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    # Convert verbose to boolean
    args.verbose = (args.verbose.lower() == "true")
    
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    
    if ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    seed = 42
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    cfg = Config(args)
    
    if local_rank == 0:
        print(f"\n" + "="*30 + " Python Config " + "="*30)
        config_dict = vars(cfg)
        for key in sorted(config_dict.keys()):
            print(f"{key}: {config_dict[key]}")
        print("="*75 + "\n")
    
    # Load Data
    tx, ty, vx, vy, tex, tey, num_items = load_loo_data(cfg.DATA_DIR_LOO)
    cfg.NUM_ITEMS = num_items # Store for later use
    
    train_dataset = TensorDataset(torch.LongTensor(tx), torch.LongTensor(ty).unsqueeze(-1))
    
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=4
    )
    
    # Prep Eval Data (List of tuples)
    valid_data = list(zip(vx, vy))
    test_data = list(zip(tex, tey))
    
    # Initialize Models
    model = TIGER(
        num_items=num_items,
        max_len=cfg.MODEL_MAX_LEN_LOO,
        dropout=cfg.DROPOUT,
        t5_model_name_or_path=cfg.T5_MODEL_PATH,
        base_path=cfg.BASE_PATH,
        dataset=cfg.DATASET,
        use_trie=cfg.USE_TRIE
    ).to(device)
    
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Load DIN Model
    din_model = None
    if cfg.USE_DEEPCTR_REWARDS and cfg.MODE in RL_MODES:
        try:
            din_model, _ = load_model_from_checkpoint(cfg.DIN_SAVE_PATH, device)
            if local_rank == 0: print("✅ DIN Model Loaded")
        except Exception as e:
            if local_rank == 0: print(f"⚠️ Failed to load DIN Model: {e}")

    # Run Trainer
    trainer = Trainer(cfg, model, din_model, device, local_rank, ddp)
    trainer.run(train_loader, valid_data, test_data)
    
    if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()