import torch
import torch.nn as nn
import torch.nn.functional as F
from .ode_solver import solve_ode


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj_matrix):
        """
        :param x: Признаки узлов (размерность: [num_nodes, input_dim]).
        :param adj_matrix: Нормализованная матрица смежности (размерность: [num_nodes, num_nodes]).
        :return: Скрытые представления узлов (размерность: [num_nodes, output_dim]).
        """
        # Первый слой GCN
        x = self.conv1(x)
        x = torch.mm(adj_matrix, x)  # Умножение на матрицу смежности
        x = F.relu(x)  # Применяем активацию

        # Второй слой GCN
        x = self.conv2(x)
        x = torch.mm(adj_matrix, x)  # Умножение на матрицу смежности

        return x


class CGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, alpha=0.5):
        super(CGNN, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, hidden_dim)
        self.output_dim = output_dim
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))  # Обучаемый параметр
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, reg_norm_adj_matrix):
        # Кодирование начальных признаков
        initial_representation = self.encoder(x, reg_norm_adj_matrix)

        # Решение ODE
        t = torch.tensor([0.0, 1.0])  # Временные точки (можно настроить)
        h = solve_ode(reg_norm_adj_matrix, initial_representation.detach().numpy(), t, self.alpha)

        # Декодирование конечных представлений
        final_representation = h[-1]  # Берем последнее состояние
        output = self.decoder(final_representation)

        # Применяем softmax для получения вероятностей классов
        output = F.softmax(output, dim=1)

        return output


# class CGNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super(CGNN, self).__init__()
#         self.encoder = GCNEncoder(input_dim, hidden_dim, hidden_dim)
#         self.output_dim = output_dim
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#     def forward(self, x, edge_index, reg_norm_adj_matrix):
#         # Кодирование начальных признаков
#         initial_representation = self.encoder(x, reg_norm_adj_matrix)
#
#         # Решение ODE
#         t = torch.tensor([0.0, 1.0])  # Временные точки (можно настроить)
#         h = solve_ode(reg_norm_adj_matrix, initial_representation.detach().numpy(), t)
#
#         # Декодирование конечных представлений
#         final_representation = h[-1]  # Берем последнее состояние
#         output = self.decoder(final_representation)
#
#         # Применяем softmax для получения вероятностей классов
#         output = F.softmax(output, dim=1)
#
#         return output
#
