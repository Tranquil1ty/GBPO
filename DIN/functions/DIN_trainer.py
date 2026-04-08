import torch
import numpy as np
from DIN_Model import DeepInterestNetwork

class DINTrain:
    def __init__(self, item_num=100, sample_negative_num=60, emb_dim=96, device='cpu',
                feature_groups=[20,20,10,10,2,2,2,1,1,1],
                sum_pooling=False,
                att_hidden_size=[64, 16],
                fc_hidden_size=[200, 80, 1],
                dropout_rate=0.1,
                optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01),
                use_lr_scheduler=True,
                warmup_steps=1000):

        self.item_num = item_num
        self.device = device
        self.N = sample_negative_num
        self.use_lr_scheduler = use_lr_scheduler
        self.warmup_steps = warmup_steps

        self.DINModel = DeepInterestNetwork(
            item_num=item_num,
            embedding_dim=emb_dim,
            feature_groups=feature_groups,
            sum_pooling=sum_pooling,
            att_hidden_size=att_hidden_size,
            fc_hidden_size=fc_hidden_size,
            dropout_rate=dropout_rate
        ).to(self.device)

        # optimizer
        self.optimizer = optimizer(self.DINModel.parameters())
        self.batch_num = 0

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=None,
                             decay_rate=0.5, learning_rate_min=1e-6):
        """Learning rate with linear warmup and cosine decay"""
        if not self.use_lr_scheduler:
            return learning_rate_base
            
        if warmup_steps is None:
            warmup_steps = self.warmup_steps
            
        if t < warmup_steps:
            # Linear warmup
            lr = learning_rate_base * (t + 1) / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (t - warmup_steps) / (10000 - warmup_steps)  # assume max 10000 steps
            lr = learning_rate_min + (learning_rate_base - learning_rate_min) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def uniform_sampled_softmax(self, batch_users, batch_labels, N):
        """
        Compute sampled softmax loss with uniform negative sampling
        使用标准的二分类交叉熵形式，更稳定且易于理解
        负样本为除了正样本外的所有物品，随机打乱后取前N个
        高度向量化处理，针对小规模物品目录优化
        """
        batch_size = batch_users.shape[0]
        
        # Prepare samples: first column is positive, rest are negative
        samples = torch.zeros((batch_size, N+1), device=self.device, dtype=torch.long)
        samples[:, 0] = batch_labels.squeeze()  # positive labels
        
        # 创建一个包含所有可能物品的向量
        all_items = torch.arange(1, self.item_num, device=self.device)
        
        # 创建一个布尔掩码，标记每个样本中应该排除的正样本
        # 使用广播机制避免显式repeat
        mask = (all_items.unsqueeze(0) != batch_labels.view(-1, 1))  # [batch_size, item_num-1]
        
        # 使用高效的索引操作，一次性为所有样本生成随机索引
        rand_indices = torch.rand((batch_size, self.item_num-1), device=self.device)
        
        # 将正样本位置的随机值设为1.0（确保它们排在最后）
        rand_indices[~mask] = 2.0
        
        # 对每个样本，获取前N个最小随机值的索引
        # 这等效于随机打乱并取前N个，但避免了显式的打乱操作
        _, top_indices = torch.topk(rand_indices, k=N, dim=1, largest=False)
        
        # 使用高级索引直接获取负样本，完全避免循环
        # 直接使用top_indices从all_items中获取实际的物品ID
        samples[:, 1:] = all_items[top_indices]
        
        # Compute scores for all samples
        batch_users_expanded = batch_users.unsqueeze(1).expand(-1, N+1, -1)
        batch_users_flat = batch_users_expanded.reshape(-1, batch_users.shape[1])
        samples_flat = samples.reshape(-1)
        
        # Get model scores
        scores_flat = self.DINModel(batch_users_flat, samples_flat).squeeze()
        scores = scores_flat.view(batch_size, N+1)
        
        # Apply log correction for uniform sampling
        # log(Q(j|x)) = log(1/(item_num-1)) for negative samples
        log_q = np.log(1.0 / (self.item_num - 1))
        
        # Positive sample score (no correction needed)
        pos_scores = scores[:, 0]
        
        # Negative sample scores with importance sampling correction
        neg_scores = scores[:, 1:] - log_q
        
        # Compute sampled softmax loss
        # Loss = -log(exp(s_pos) / (exp(s_pos) + sum(exp(s_neg))))
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        loss = -torch.log_softmax(all_scores, dim=1)[:, 0].mean()
        
        return loss

    def update_DIN(self, batch_users, batch_labels):
        """Update model with one batch of data"""
        self.batch_num += 1
        
        # Move data to device
        batch_users = batch_users.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Ensure batch_labels has correct shape
        if batch_labels.dim() == 1:
            batch_labels = batch_labels.unsqueeze(1)
        
        # Compute loss
        loss = self.uniform_sampled_softmax(batch_users, batch_labels, self.N)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.DINModel.parameters(), max_norm=5.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Update learning rate
        if self.use_lr_scheduler:
            self.update_learning_rate(self.batch_num)
        
        return loss

    def calculate_preference(self, batch_user, batch_items):
        """Calculate preference scores for user-item pairs"""
        return self.DINModel(batch_user, batch_items)
