#!/usr/bin/env python3
"""
DIN预测模块 - 提供模型加载和预测功能
主要的训练和评估功能请使用 train_loo.py
"""

import os
from typing import Dict, Tuple, List, Union

import torch
import numpy as np

from DIN_trainer import DINTrain


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[DINTrain, Dict]:
    """
    从检查点加载训练好的DIN模型
    
    Args:
        checkpoint_path: 保存的模型检查点路径
        device: 加载模型的设备
    
    Returns:
        Tuple of (loaded model, config dictionary)
    """
    print(f"从检查点加载模型: {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件未找到: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    # 使用保存的配置初始化模型
    model = DINTrain(
        item_num=config.get('num_items', 12102),
        sample_negative_num=config.get('sample_negative_num', 50),
        emb_dim=config.get('embedding_dim', 64),
        att_hidden_size = config.get('att_hidden_size',[80, 40]),
        fc_hidden_size = config.get('fc_hidden_size', [256, 128, 1]),
        device=device,
        feature_groups=config.get('feature_groups', [8, 6, 4, 2]),
        sum_pooling=config.get('sum_pooling', True),
        optimizer=lambda params: torch.optim.AdamW(params, lr=config.get('learning_rate', 1e-3), weight_decay=0.01)
    )
    
    # 加载模型状态
    model.DINModel.load_state_dict(checkpoint['model_state_dict'])
    model.DINModel.eval()
    
    print(f"模型加载成功!")
    if 'best_gauc' in checkpoint:
        print(f"模型 GAUC: {checkpoint['best_gauc']:.4f}")
    if 'best_auc' in checkpoint:
        print(f"模型 AUC: {checkpoint['best_auc']:.4f}")
    
    return model, config


def predict(model: DINTrain, user_sequence: Union[List[int], np.ndarray, torch.Tensor], 
            candidate_items: Union[List[int], np.ndarray, torch.Tensor], 
            device: torch.device = None) -> np.ndarray:
    """
    预测函数：输入用户交互序列和待预测的n个样本，输出对n个样本的评分
    
    Args:
        model: 训练好的DIN模型
        user_sequence: 用户交互序列，形状为 (seq_len,)，包含用户历史交互的物品ID
        candidate_items: 待预测的候选物品列表，形状为 (n,)，包含n个候选物品ID
        device: 计算设备，如果为None则使用模型的设备
    
    Returns:
        scores: 对候选物品的评分，形状为 (n,)，数值越大表示用户越可能喜欢该物品
    """
    if device is None:
        device = model.device
    
    # 确保模型处于评估模式
    model.DINModel.eval()
    
    # 数据预处理
    if isinstance(user_sequence, (list, np.ndarray)):
        user_sequence = torch.tensor(user_sequence, dtype=torch.long, device=device)
    if isinstance(candidate_items, (list, np.ndarray)):
        candidate_items = torch.tensor(candidate_items, dtype=torch.long, device=device)
    
    # 确保user_sequence是2D张量 (1, seq_len)
    if user_sequence.dim() == 1:
        user_sequence = user_sequence.unsqueeze(0)
    
    # 确保candidate_items是2D张量 (n, 1)
    if candidate_items.dim() == 1:
        candidate_items = candidate_items.unsqueeze(-1)
    
    n_candidates = candidate_items.shape[0]
    
    with torch.no_grad():
        # 重复用户序列以匹配候选物品数量
        user_sequences_repeated = user_sequence.repeat(n_candidates, 1)  # (n, seq_len)
        
        # 计算每个候选物品的评分
        scores = model.calculate_preference(user_sequences_repeated, candidate_items)  # (n, 1)
        
        # 应用sigmoid激活函数得到概率分数
        scores = torch.sigmoid(scores).squeeze(-1)  # (n,)
        
        # 转换为numpy数组
        scores = scores.cpu().numpy()
    
    return scores


def predict_logit(model: DINTrain, user_sequence: Union[List[int], np.ndarray, torch.Tensor], 
                candidate_items: Union[List[int], np.ndarray, torch.Tensor], 
                device: torch.device = None) -> np.ndarray:
    """
    预测函数：输出未经过Sigmoid的Logit
    """
    if device is None:
        device = model.device
    
    model.DINModel.eval()
    
    if isinstance(user_sequence, (list, np.ndarray)):
        user_sequence = torch.tensor(user_sequence, dtype=torch.long, device=device)
    if isinstance(candidate_items, (list, np.ndarray)):
        candidate_items = torch.tensor(candidate_items, dtype=torch.long, device=device)
    
    if user_sequence.dim() == 1:
        user_sequence = user_sequence.unsqueeze(0)
    if candidate_items.dim() == 1:
        candidate_items = candidate_items.unsqueeze(-1)
    
    n_candidates = candidate_items.shape[0]
    
    with torch.no_grad():
        user_sequences_repeated = user_sequence.repeat(n_candidates, 1)
        logits = model.calculate_preference(user_sequences_repeated, candidate_items)  # (n, 1)
        logits = logits.squeeze(-1).cpu().numpy()
    
    return logits


def predict_batch(model: DINTrain, user_sequences: Union[List[List[int]], np.ndarray, torch.Tensor], 
                  candidate_items: Union[List[int], np.ndarray, torch.Tensor], 
                  device: torch.device = None) -> np.ndarray:
    """
    批量预测函数：输入多个用户交互序列和待预测的n个样本，输出对n个样本的评分
    
    Args:
        model: 训练好的DIN模型
        user_sequences: 多个用户交互序列，形状为 (batch_size, seq_len)
        candidate_items: 待预测的候选物品列表，形状为 (n,)
        device: 计算设备，如果为None则使用模型的设备
    
    Returns:
        scores: 对候选物品的评分，形状为 (batch_size, n)
    """
    if device is None:
        device = model.device
    
    # 确保模型处于评估模式
    model.DINModel.eval()
    
    # 数据预处理
    if isinstance(user_sequences, list):
        user_sequences = torch.tensor(user_sequences, dtype=torch.long, device=device)
    elif isinstance(user_sequences, np.ndarray):
        user_sequences = torch.tensor(user_sequences, dtype=torch.long, device=device)
    
    if isinstance(candidate_items, (list, np.ndarray)):
        candidate_items = torch.tensor(candidate_items, dtype=torch.long, device=device)
    
    # 确保user_sequences是2D张量 (batch_size, seq_len)
    if user_sequences.dim() == 1:
        user_sequences = user_sequences.unsqueeze(0)
    
    # 确保candidate_items是2D张量 (n, 1)
    if candidate_items.dim() == 1:
        candidate_items = candidate_items.unsqueeze(-1)
    
    batch_size = user_sequences.shape[0]
    n_candidates = candidate_items.shape[0]
    
    with torch.no_grad():
        # 为每个用户计算所有候选物品的评分
        all_scores = []
        
        for i in range(batch_size):
            user_seq = user_sequences[i:i+1]  # (1, seq_len)
            user_seq_repeated = user_seq.repeat(n_candidates, 1)  # (n, seq_len)
            
            # 计算评分
            scores = model.calculate_preference(user_seq_repeated, candidate_items)  # (n, 1)
            scores = torch.sigmoid(scores).squeeze(-1)  # (n,)
            all_scores.append(scores)
        
        # 堆叠所有用户的评分
        all_scores = torch.stack(all_scores, dim=0)  # (batch_size, n)
        
        # 转换为numpy数组
        all_scores = all_scores.cpu().numpy()
    
    return all_scores


def predict_top_k(model: DINTrain, user_sequence: Union[List[int], np.ndarray, torch.Tensor], 
                  all_items: Union[List[int], np.ndarray, torch.Tensor], 
                  k: int = 10, device: torch.device = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    预测用户最可能喜欢的top-k物品
    
    Args:
        model: 训练好的DIN模型
        user_sequence: 用户交互序列
        all_items: 所有候选物品列表
        k: 返回top-k个物品
        device: 计算设备
    
    Returns:
        top_k_items: top-k个物品ID，形状为 (k,)
        top_k_scores: 对应的评分，形状为 (k,)
    """
    if device is None:
        device = model.device
    
    # 计算所有物品的评分
    all_scores = predict(model, user_sequence, all_items, device)
    
    # 获取top-k
    top_k_indices = np.argsort(all_scores)[-k:][::-1]  # 降序排列
    top_k_items = np.array(all_items)[top_k_indices]
    top_k_scores = all_scores[top_k_indices]
    
    return top_k_items, top_k_scores


if __name__ == '__main__':
    print("DIN预测模块")
    print("主要功能:")
    print("1. load_model_from_checkpoint: 从检查点加载模型")
    print("2. predict: 单个用户对候选物品的评分预测")
    print("3. predict_batch: 批量用户对候选物品的评分预测")
    print("4. predict_top_k: 预测用户最可能喜欢的top-k物品")
    print("\n使用示例:")
    print("model, config = load_model_from_checkpoint('model.pth', device)")
    print("scores = predict(model, user_seq, candidate_items, device)")
    print("top_items, top_scores = predict_top_k(model, user_seq, all_items, k=10)")