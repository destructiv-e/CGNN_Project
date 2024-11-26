import torch
import torch.nn as nn
import torch.nn.functional as F


class SGFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SGFormer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1)

    def forward(self, X, reg_norm_adj_matrix):
        reg_norm_adj_matrix = torch.tensor(reg_norm_adj_matrix, dtype=torch.float32)
        for layer in self.layers:
            X = F.relu(layer(X))
            X = self.attention(X, X, X, attn_mask=reg_norm_adj_matrix)[0]

        return X


class CGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(CGNN, self).__init__()
        self.encoder = SGFormer(input_dim, hidden_dim, num_layers)
        # другие компоненты модели, такие как ODE Solver и Decoder

    def forward(self, x, edge_index, reg_norm_adj_matrix):
        initial_representation = self.encoder(x, reg_norm_adj_matrix)
        # продолжение с использованием ODE Solver и Decoder
        return initial_representation

# import torch
# import torch.nn as nn
#
#
# class SimpleEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(SimpleEncoder, self).__init__()
#         self.fc = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return self.relu(self.fc(x))
#
#
# class CGNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(CGNN, self).__init__()
#         self.encoder = SimpleEncoder(input_dim, hidden_dim)
#         # Добавить другие компоненты модели, такие как ODE Solver и Decoder
#
#     def forward(self, x, edge_index):
#         initial_representation = self.encoder(x)
#         # Здесь будет продолжение с использованием ODE Solver и Decoder
#         return initial_representation
