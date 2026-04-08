import json
import os
import pickle
import random
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DINDataset(Dataset):
    """Dataset class for DIN model training and evaluation"""
    
    def __init__(self, data: List[Tuple], mode: str = 'train'):
        """
        Args:
            data: List of tuples containing (user_features, target_item, label) for train
                  or (user_features, positive_item, negative_item) for test
            mode: 'train' or 'test'
        """
        self.data = data
        self.mode = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            user_features, target_item, label = self.data[idx]
            return {
                'user_features': torch.tensor(user_features, dtype=torch.long),
                'target_item': torch.tensor(target_item, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float32)
            }
        else:  # test mode
            user_features, pos_item, neg_item = self.data[idx]
            return {
                'user_features': torch.tensor(user_features, dtype=torch.long),
                'pos_item': torch.tensor(pos_item, dtype=torch.long),
                'neg_item': torch.tensor(neg_item, dtype=torch.long)
            }


class DINDataInput:
    """Data input iterator for DIN model, similar to DeepCTR-Torch style"""
    
    def __init__(self, data: List[Tuple], batch_size: int, mode: str = 'train'):
        self.batch_size = batch_size
        self.data = data
        self.mode = mode
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        
        batch_data = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        
        if self.mode == 'train':
            user_features_batch = []
            target_items_batch = []
            labels_batch = []
            
            for user_features, target_item, label in batch_data:
                user_features_batch.append(user_features)
                target_items_batch.append(target_item)
                labels_batch.append(label)
            
            return self.i, {
                'user_features': torch.tensor(np.array(user_features_batch), dtype=torch.long),
                'target_items': torch.tensor(np.array(target_items_batch), dtype=torch.long),
                'labels': torch.tensor(np.array(labels_batch), dtype=torch.float32)
            }
        
        else:  # test mode
            user_features_batch = []
            pos_items_batch = []
            neg_items_batch = []
            
            for user_features, pos_item, neg_item in batch_data:
                user_features_batch.append(user_features)
                pos_items_batch.append(pos_item)
                neg_items_batch.append(neg_item)
            
            return self.i, {
                'user_features': torch.tensor(np.array(user_features_batch), dtype=torch.long),
                'pos_items': torch.tensor(np.array(pos_items_batch), dtype=torch.long),
                'neg_items': torch.tensor(np.array(neg_items_batch), dtype=torch.long)
            }


def load_loo_data(data_dir_loo: str) -> Tuple:
    """
    加载 LOO 策略处理后的训练和测试数据, 包括用户ID
    
    Args:
        data_dir_loo: LOO 处理后的数据目录 (e.g., .../processed_loo_strategy)
        
    Returns:
        train_x, train_y, train_user_ids, valid_x, valid_y, test_x, test_y, num_items
    """
    print(f"加载 LOO 数据从: {data_dir_loo}")
    train_x_path = os.path.join(data_dir_loo, 'train_x_loo.npy')
    train_y_path = os.path.join(data_dir_loo, 'train_y_loo.npy')
    train_user_ids_path = os.path.join(data_dir_loo, 'train_user_id_loo.npy')
    valid_x_path = os.path.join(data_dir_loo, 'valid_x_loo.npy')
    valid_y_path = os.path.join(data_dir_loo, 'valid_y_loo.npy')
    test_x_path = os.path.join(data_dir_loo, 'test_x_loo.npy')
    test_y_path = os.path.join(data_dir_loo, 'test_y_loo.npy')

    if not (os.path.exists(train_x_path) and os.path.exists(train_y_path) and \
            os.path.exists(train_user_ids_path) and \
            os.path.exists(valid_x_path) and os.path.exists(valid_y_path) and \
            os.path.exists(test_x_path) and os.path.exists(test_y_path)):
        raise FileNotFoundError(f"一个或多个 LOO 数据文件未在 {data_dir_loo} 中找到。 "
                                f"请先使用 LOO 策略运行数据预处理脚本。")

    train_x = np.load(train_x_path)
    train_y = np.load(train_y_path)
    train_user_ids = np.load(train_user_ids_path)
    valid_x = np.load(valid_x_path)
    valid_y = np.load(valid_y_path)
    test_x = np.load(test_x_path)
    test_y = np.load(test_y_path)

    print(f"训练集 X 形状: {train_x.shape}, Y 形状: {train_y.shape}, User IDs 形状: {train_user_ids.shape}")
    print(f"验证集 X 形状: {valid_x.shape}, Y 形状: {valid_y.shape}")
    print(f"测试集 X 形状: {test_x.shape}, Y 形状: {test_y.shape}")
    
    # 确保train_y和test_y是单个物品ID而不是序列
    if len(train_y.shape) > 1 and train_y.shape[1] > 1:
        print("警告：train_y 是序列，将使用最后一个元素。")
        train_y = train_y[:, -1]
    if len(valid_y.shape) > 1 and valid_y.shape[1] > 1:
        print("警告：valid_y 是序列，将使用最后一个元素。")
        valid_y = valid_y[:, -1]
    if len(test_y.shape) > 1 and test_y.shape[1] > 1:
        print("警告：test_y 是序列，将使用最后一个元素。")
        test_y = test_y[:, -1]
        
    max_item_train_x = np.max(train_x) if train_x.size > 0 else 0
    max_item_train_y = np.max(train_y) if train_y.size > 0 else 0
    max_item_valid_x = np.max(valid_x) if valid_x.size > 0 else 0
    max_item_valid_y = np.max(valid_y) if valid_y.size > 0 else 0
    max_item_test_x = np.max(test_x) if test_x.size > 0 else 0
    max_item_test_y = np.max(test_y) if test_y.size > 0 else 0
    
    max_item_id = max(max_item_train_x, max_item_train_y, max_item_valid_x, max_item_valid_y, max_item_test_x, max_item_test_y)
    num_items = int(max_item_id) + 1  # +1 because we need embedding for item_id=0 (padding) and items 1 to max_item_id

    print(f"物品总数 (包含padding): {num_items} (物品ID范围: 1-{max_item_id})")
    
    all_items = np.concatenate([
        train_x.flatten(), train_y.flatten(),
        valid_x.flatten(), valid_y.flatten(),
        test_x.flatten(), test_y.flatten()
    ])
    unique_items = np.unique(all_items)
    min_actual_item_id = np.min(unique_items[unique_items > 0]) if len(unique_items[unique_items > 0]) > 0 else 1
    max_actual_item_id = np.max(unique_items)

    print(f"数据中实际最小物品ID (非填充): {min_actual_item_id}")
    print(f"数据中实际最大物品ID: {max_actual_item_id}")

    return train_x, train_y, train_user_ids, valid_x, valid_y, test_x, test_y, num_items


def convert_loo_to_din_format(train_x: np.ndarray, train_y: np.ndarray, 
                             valid_x: np.ndarray, valid_y: np.ndarray,
                             test_x: np.ndarray, test_y: np.ndarray,
                             num_items: int) -> Tuple:
    """
    Convert LOO data format to DIN training format
    
    Args:
        train_x, train_y: Training data (user histories and target items)
        valid_x, valid_y: Validation data
        test_x, test_y: Test data
        num_items: Total number of items
    
    Returns:
        train_data, valid_data, test_data in DIN format
        - train_data: List of (user_features, target_item, 1.0) - only positive samples
        - valid_data: List of (user_features, pos_item, neg_item) - for evaluation
        - test_data: List of (user_features, pos_item, neg_item) - for evaluation
    """
    print("Converting LOO data to DIN format...")
    
    # Convert training data - only positive samples for DIN uniform sampled softmax
    train_data = []
    for i in range(len(train_x)):
        user_features = train_x[i]
        target_item = train_y[i]
        
        # Only add positive samples - negative sampling is handled in uniform_sampled_softmax
        train_data.append((user_features, target_item, 1.0))
    
    # Convert validation data (positive-negative pairs)
    valid_data = []
    for i in range(len(valid_x)):
        user_features = valid_x[i]
        pos_item = valid_y[i]
        # Generate random negative item
        neg_item = random.randint(1, num_items - 1)  # Sample from 1 to max_item_id
        while neg_item == pos_item:
            neg_item = random.randint(1, num_items - 1)
        valid_data.append((user_features, pos_item, neg_item))
    
    # Convert test data (positive-negative pairs)
    test_data = []
    for i in range(len(test_x)):
        user_features = test_x[i]
        pos_item = test_y[i]
        # Generate random negative item
        neg_item = random.randint(1, num_items - 1)  # Sample from 1 to max_item_id
        while neg_item == pos_item:
            neg_item = random.randint(1, num_items - 1)
        test_data.append((user_features, pos_item, neg_item))
    
    print(f"Converted - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    return train_data, valid_data, test_data




if __name__ == "__main__":
    # Example usage
    print("DIN Data Loader Module")
    print("This module provides data loading utilities for DIN model training and evaluation.")
