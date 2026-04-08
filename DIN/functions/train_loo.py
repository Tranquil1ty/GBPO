#!/usr/bin/env python3
"""
修复版DIN训练脚本 - 解决了注意力机制、学习率调度和过拟合问题
"""

import os
import sys
import time
import random
import argparse  # ✅ 新增

import numpy as np
import torch
import torch.nn as nn

from DIN_trainer import DINTrain
from data_loader import load_loo_data, convert_loo_to_din_format, DINDataInput
from evaluation import evaluate_model, EarlyStopping


def set_random_seeds(seed: int = 42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model: DINTrain, train_data: list, batch_size: int, device: torch.device) -> float:
    """训练一个epoch"""
    model.DINModel.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for _, batch in DINDataInput(train_data, batch_size, mode='train'):
        user_features = batch['user_features'].to(device)
        target_items = batch['target_items'].to(device)
        
        loss = model.update_DIN(user_features, target_items)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """主训练函数"""
    # ✅ 新增 argparse
    parser = argparse.ArgumentParser(description="DIN 训练脚本 (修复版)")
    parser.add_argument("--data_dir_loo", type=str, required=True, help="LOO 数据路径")
    parser.add_argument("--save_path", type=str, required=True, help="模型保存路径")
    args = parser.parse_args()

    print("DIN训练 (修复版) - LOO数据格式 + Sum Pooling模式")
    print("=" * 50)
    print("主要特性:")
    print("1. 添加了正确的softmax注意力权重归一化")
    print("2. 修复了embedding维度处理")
    print("3. 改进了学习率调度策略")
    print("4. 增加了梯度裁剪防止梯度爆炸")
    print("5. 增加了更多负采样和正则化")
    print("6. 修复了Recall@k计算中的维度bug")
    print("7. 支持选择Recall@10或NDCG@10作为模型选择标准")
    print("=" * 50)
    
    # 配置参数
    config = {
        'data_dir_loo': args.data_dir_loo,  # ✅ 改为命令行输入
        'batch_size': 128,
        'test_batch_size': 512,
        'epochs': 100,  # 减少epoch数，依赖早停
        'learning_rate': 5e-4,  # 降低学习率
        'weight_decay': 0.01,  # 增加权重衰减
        'embedding_dim': 64,  # 减小embedding维度，防止过拟合
        'att_hidden_size': [80, 40],  # 稍微增大注意力网络
        'fc_hidden_size': [200, 80, 1],  # 增大FC层
        'dropout_rate': 0.1,  # 增加dropout
        'sample_negative_num': 3000,  # 增加负采样数量
        'patience': 15,  # 增加早停耐心
        'min_delta': 0.0001,
        'save_path': args.save_path,  # ✅ 改为命令行输入
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'feature_groups': [8, 6, 4, 2],
        'sum_pooling': True,
        'save_best_model': True,
        'use_lr_scheduler': True,
        'warmup_steps': 500,  # 减少warmup步数
        'gradient_clip': 5.0,  # 梯度裁剪
        'selection_metric': 'ndcg',  # 👈 使用ndcg作为选择标准
    }
    
    # 设置随机种子
    set_random_seeds(config['seed'])
    
    # 设备配置
    device = torch.device(config['device'])
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(config['save_path'], exist_ok=True)
    
    # 加载和准备数据
    try:
        print("加载LOO数据...")
        train_x, train_y, train_user_ids, valid_x, valid_y, test_x, test_y, num_items = load_loo_data(config['data_dir_loo'])
        
        print("转换为DIN格式...")
        train_data, valid_data, test_data = convert_loo_to_din_format(
            train_x, train_y, valid_x, valid_y, test_x, test_y, num_items
        )
        
        # 打乱训练数据
        random.shuffle(train_data)
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 计算实际的batch数量用于学习率调度
    batches_per_epoch = len(train_data) // config['batch_size']
    actual_warmup_steps = batches_per_epoch * 2  # 2个epoch的warmup
    
    # 初始化模型
    print("初始化DIN模型 (修复版)...")
    model = DINTrain(
        item_num=num_items,
        sample_negative_num=config['sample_negative_num'],
        emb_dim=config['embedding_dim'],
        device=device,
        feature_groups=config['feature_groups'],
        sum_pooling=config['sum_pooling'],
        att_hidden_size=config['att_hidden_size'],
        fc_hidden_size=config['fc_hidden_size'],
        dropout_rate=config['dropout_rate'],
        optimizer=lambda params: torch.optim.AdamW(
            params, 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay'],
            eps=1e-8
        ),
        use_lr_scheduler=config['use_lr_scheduler'],
        warmup_steps=actual_warmup_steps
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.DINModel.parameters()):,}")
    print(f"每个epoch的batch数: {batches_per_epoch}")
    print(f"Warmup步数: {actual_warmup_steps}")
    
    # 初始化早停
    early_stopping = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta'],
        restore_best_weights=True
    )
    
    # 训练循环
    best_gauc = -float('inf')
    best_auc = -float('inf')
    best_metric_value = -float('inf')  # 用于存储选定指标的最佳值
    start_time = time.time()
    
    # 确定使用哪个指标
    selection_metric = config.get('selection_metric', 'ndcg').lower()
    if selection_metric not in ['recall', 'ndcg']:
        raise ValueError(f"selection_metric必须是'recall'或'ndcg'，当前值: {selection_metric}")
    
    metric_name = "Recall@10" if selection_metric == 'recall' else "NDCG@10"
    
    print("\n开始训练...")
    print("-" * 50)
    print(f"注意：使用 {metric_name} 作为模型选择标准")
    
    # 初始评估 - 根据选择计算相应指标
    if selection_metric == 'recall':
        val_gauc, val_auc, val_metric, _ = evaluate_model(
            model, valid_data, config['test_batch_size'], device, num_items, k=10, 
            calculate_recall=True, calculate_ndcg=False
        )
    else:  # ndcg
        val_gauc, val_auc, _, val_metric = evaluate_model(
            model, valid_data, config['test_batch_size'], device, num_items, k=10, 
            calculate_recall=False, calculate_ndcg=True
        )
    
    print(f"初始验证结果 - GAUC: {val_gauc:.4f}, AUC: {val_auc or 0.0:.4f}, {metric_name}: {val_metric or 0.0:.4f}")
    
    no_improve_count = 0
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # 打乱训练数据
        random.shuffle(train_data)
        
        # 训练一个epoch
        avg_loss = train_epoch(model, train_data, config['batch_size'], device)
        
        # 验证 - 根据选择计算相应指标
        if selection_metric == 'recall':
            val_gauc, val_auc, val_metric, _ = evaluate_model(
                model, valid_data, config['test_batch_size'], device, num_items, k=10, 
                calculate_recall=True, calculate_ndcg=False
            )
        else:  # ndcg
            val_gauc, val_auc, _, val_metric = evaluate_model(
                model, valid_data, config['test_batch_size'], device, num_items, k=10, 
                calculate_recall=False, calculate_ndcg=True
            )
        
        epoch_time = time.time() - epoch_start_time
        
        # 获取当前学习率
        current_lr = model.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch + 1:3d}/{config['epochs']} | "
              f"时间: {epoch_time:5.1f}s | "
              f"损失: {avg_loss:.4f} | "
              f"GAUC: {val_gauc:.4f} | "
              f"AUC: {val_auc or 0.0:.4f} | "
              f"{metric_name}: {val_metric or 0.0:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # 保存最佳模型（基于选定的指标）
        if val_metric and val_metric > best_metric_value:
            best_metric_value = val_metric
            best_gauc = val_gauc
            best_auc = val_auc
            no_improve_count = 0
            
            if config['save_best_model']:
                save_config = config.copy()
                save_config['num_items'] = num_items
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.DINModel.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'best_metric_value': best_metric_value,
                    'selection_metric': selection_metric,
                    'metric_name': metric_name,
                    'best_gauc': best_gauc,
                    'best_auc': best_auc,
                    'config': save_config
                }
                
                torch.save(checkpoint, os.path.join(config['save_path'], 'best_model_fixed.pth'))
                print(f"  → 保存新的最佳模型! {metric_name}: {best_metric_value:.4f} (GAUC: {best_gauc:.4f})")
        else:
            no_improve_count += 1
        
        # 早停检查
        if early_stopping(val_metric if val_metric else 0.0, model):
            print(f"  → 早停触发，在第 {epoch + 1} 轮后停止训练")
            break
        
        # 手动降低学习率
        if no_improve_count >= 5 and no_improve_count % 5 == 0:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"  → 手动降低学习率到: {model.optimizer.param_groups[0]['lr']:.6f}")
    
    # 最终测试
    print("\n" + "=" * 50)
    print("最终测试评估（同时计算Recall@10和NDCG@10）...")
    
    if config['save_best_model'] and os.path.exists(os.path.join(config['save_path'], 'best_model_fixed.pth')):
        checkpoint = torch.load(os.path.join(config['save_path'], 'best_model_fixed.pth'), weights_only=False)
        model.DINModel.load_state_dict(checkpoint['model_state_dict'])
        print("已加载最佳模型进行测试")
    
    test_gauc, test_auc, test_recall_10, test_ndcg_10 = evaluate_model(
        model, test_data, config['test_batch_size'], device, num_items, k=10, 
        calculate_recall=True, calculate_ndcg=True
    )
    
    total_time = time.time() - start_time
    
    print(f"\n训练完成!")
    print(f"总训练时间: {total_time:.1f}秒")
    print(f"模型选择标准: {metric_name}")
    print(f"最佳验证 {metric_name}: {best_metric_value:.4f}")
    print(f"最佳验证 GAUC: {best_gauc:.4f}")
    print(f"最佳验证 AUC:  {best_auc or 0.0:.4f}")
    print(f"最终测试 GAUC: {test_gauc:.4f}")
    print(f"最终测试 AUC:  {test_auc or 0.0:.4f}")
    print(f"最终测试 Recall@10: {test_recall_10 or 0.0:.4f}")
    print(f"最终测试 NDCG@10: {test_ndcg_10 or 0.0:.4f}")
    print("=" * 50)
    
    save_config = config.copy()
    save_config['num_items'] = num_items
    
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.DINModel.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'final_test_gauc': test_gauc,
        'final_test_auc': test_auc,
        'final_test_recall_10': test_recall_10,
        'final_test_ndcg_10': test_ndcg_10,
        'best_val_metric_value': best_metric_value,
        'selection_metric': selection_metric,
        'metric_name': metric_name,
        'best_val_gauc': best_gauc,
        'best_val_auc': best_auc,
        'config': save_config,
        'training_time': total_time
    }
    
    torch.save(final_checkpoint, os.path.join(config['save_path'], 'final_model_fixed.pth'))
    print(f"最终模型已保存至: {config['save_path']}/final_model_fixed.pth")
    
    with open(os.path.join(config['save_path'], 'training_results.txt'), 'w') as f:
        f.write(f"DIN训练结果 (修复版) - 使用{metric_name}作为模型选择标准\n")
        f.write("=" * 50 + "\n")
        f.write(f"训练时间: {total_time:.1f}秒\n")
        f.write(f"模型选择标准: {metric_name}\n")
        f.write(f"最佳验证 {metric_name}: {best_metric_value:.4f}\n")
        f.write(f"最佳验证 GAUC: {best_gauc:.4f}\n")
        f.write(f"最佳验证 AUC: {best_auc or 0.0:.4f}\n")
        f.write(f"测试 GAUC: {test_gauc:.4f}\n")
        f.write(f"测试 AUC: {test_auc or 0.0:.4f}\n")
        f.write(f"测试 Recall@10: {test_recall_10 or 0.0:.4f}\n")
        f.write(f"测试 NDCG@10: {test_ndcg_10 or 0.0:.4f}\n")
        f.write("\n配置:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")


if __name__ == '__main__':
    main()
