import torch.nn as nn
import torch
import torch.nn.functional as F

class DeepInterestNetwork(nn.Module):
    def __init__(self, item_num=100, embedding_dim=96,
                 feature_groups=[20, 20, 10, 10, 2, 2, 2, 1, 1, 1],
                 sum_pooling=False,
                 att_hidden_size=[64, 16],
                 fc_hidden_size=[200, 80, 1],
                 dropout_rate=0.1
                 ):
        super().__init__()
        self.item_num = item_num
        self.embed_dim = embedding_dim
        self.feature_num = sum(feature_groups)
        self.sum_pooling = sum_pooling
        self.item_embedding = EmbeddingLayer(item_num, embedding_dim)
        self.attention_unit = LocalActivationUnit(hidden_size=att_hidden_size, bias=[True, True],
                                                  embedding_dim=embedding_dim, batch_norm=False,
                                                  dropout_rate=dropout_rate)
        if sum_pooling:
            self.fc_layer = FullyConnectedLayer(input_size=2 * embedding_dim,
                                                hidden_size=fc_hidden_size,
                                                bias=[True, True, True],
                                                activation='dice',
                                                sigmoid=False,
                                                dropout_rate=dropout_rate)
        else:
            self.fc_layer = FullyConnectedLayer(input_size=(len(feature_groups) + 1) * embedding_dim,
                                                hidden_size=fc_hidden_size,
                                                bias=[True, True, True],
                                                activation='dice',
                                                sigmoid=False,
                                                dropout_rate=dropout_rate)

            # window matrix for each window's weight sum
            window_matrix = torch.zeros(len(feature_groups), sum(feature_groups))
            start_index = 0
            for i, feature in enumerate(feature_groups):
                window_matrix[i, start_index:start_index + feature] = 1.0
                start_index += feature
            self.register_buffer('window_matrix',window_matrix)



    def forward(self, batch_user, batch_label):
        """
        item_num fill with the absence in batch_user
        """
        batch_size, seq_len = batch_user.shape
        
        # 确保batch_label是正确的形状
        if batch_label.dim() == 1:
            batch_label = batch_label.unsqueeze(1)  # (batch_size, 1)
        
        # Get target item embedding - 不要squeeze，保持3D形状
        target_embedding = self.item_embedding(batch_label)  # (batch_size, 1, embedding_dim)
        
        # Get embeddings for historical behaviors
        history_embedding = self.item_embedding(batch_user)  # (batch_size, seq_len, embedding_dim)
        
        # Expand target embedding to match history sequence length
        target_embedding_expanded = target_embedding.expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
        
        # Calculate attention scores (logits)
        attention_logits = self.attention_unit(history_embedding, target_embedding_expanded)  # (batch_size, seq_len, 1)
        
        # Create mask for padding items (item_id=0)
        mask = (batch_user != 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Apply mask to attention logits (set padding positions to -inf)
        masked_attention_logits = attention_logits.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get normalized attention weights
        attention_weights = F.softmax(masked_attention_logits, dim=1)  # (batch_size, seq_len, 1)
        
        if self.sum_pooling:
            # Weighted sum of historical behaviors using attention weights
            user_interest_embedding = (history_embedding * attention_weights).sum(dim=1)  # (batch_size, embedding_dim)
            
            # Get target embedding in 2D
            target_embedding_2d = target_embedding.squeeze(1)  # (batch_size, embedding_dim)
            
            # Concatenate user interest and target item embedding
            combined_embedding = torch.cat([user_interest_embedding, target_embedding_2d], dim=1)
            
            return self.fc_layer(combined_embedding)
        else:
            # This part handles the feature_groups logic if needed in the future
            weighted_history = history_embedding * attention_weights  # (batch_size, seq_len, embed_dim)
            
            # Apply window matrix to group sequence features
            # window_matrix: (num_groups, seq_len)
            # weighted_history: (batch_size, seq_len, embed_dim)
            # Result: (batch_size, num_groups, embed_dim)
            grouped_interest = torch.einsum('gs,bse->bge', self.window_matrix, weighted_history)
            grouped_interest = grouped_interest.reshape(batch_size, -1)  # (batch_size, num_groups * embed_dim)
            
            # Get target embedding in 2D
            target_embedding_2d = target_embedding.squeeze(1)
            
            # Concatenate grouped interest and target item embedding
            combined_embedding = torch.cat([grouped_interest, target_embedding_2d], dim=-1)
            
            return self.fc_layer(combined_embedding)





class SafeBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SafeBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        # Support (N, C) and (N, L, C). For 3D, transpose to (N, C, L) for BN then back.
        if x.dim() == 2:
            return self.bn(x)
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            x = self.bn(x)
            x = torch.transpose(x, 1, 2)
            return x
        raise ValueError("SafeBatchNorm1d expects a 2D or 3D tensor input")


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=True,
         dropout_rate=0.1, activation='relu', sigmoid=False, dice_dim=2):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(SafeBatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_size[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1], bias=bias[i+1]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=True, dropout_rate=0.1):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4 * embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3,
                                       dropout_rate=dropout_rate)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3,
                                       dropout_rate=dropout_rate)

    def forward(self, user_behavior, queries):
        # user_behavior: (batch_size, seq_len, embedding_dim)
        # queries: (batch_size, seq_len, embedding_dim)
        attention_input = torch.cat([
            queries, 
            user_behavior, 
            queries - user_behavior, 
            queries * user_behavior
        ], dim=-1)  # (batch_size, seq_len, 4*embedding_dim)
        
        attention_output = self.fc2(self.fc1(attention_input))  # (batch_size, seq_len, 1)
        return attention_output


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 3:
            self.alpha = nn.Parameter(torch.zeros((num_features, 1)))
        elif self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((num_features,)))
            
    def forward(self, x):
        if self.dim == 3:
            # x is [batch_size, seq_len, hidden_size]
            x = torch.transpose(x, 1, 2)  # [batch_size, hidden_size, seq_len]
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)  # [batch_size, seq_len, hidden_size]
        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, item_num, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        # item_num+1 to include padding_idx=0
        self.embed = nn.Embedding(item_num+1, embedding_dim, padding_idx=0)
        
        # Initialize embeddings with xavier uniform
        nn.init.xavier_uniform_(self.embed.weight.data[1:])  # Don't initialize padding embedding
        
    def forward(self, x):
        return self.embed(x)
