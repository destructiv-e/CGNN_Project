import torch
from torch_geometric.data import Data
import numpy as np

def create_adjacency_matrix(edge_index, num_nodes):
    """
    Создает матрицу смежности из списка ребер.

    :param edge_index: Список ребер в формате (2, num_edges), где каждый элемент - индекс узла.
    :param num_nodes: Количество узлов в графе.
    :return: Матрица смежности в формате numpy.ndarray.
    """
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edge_index.T:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1  # Если граф неориентированный
    return adj_matrix

def normalize_adjacency_matrix(adj_matrix):
    """
    Нормализует матрицу смежности.

    :param adj_matrix: Матрица смежности в формате numpy.ndarray.
    :return: Нормализованная матрица смежности в формате numpy.ndarray.
    """
    num_nodes = adj_matrix.shape[0]
    I = np.eye(num_nodes)
    adj_matrix_with_self_loops = adj_matrix + I  # Добавляем петли к каждому узлу

    degree = np.sum(adj_matrix_with_self_loops, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
    degree_inv_sqrt = np.diag(degree_inv_sqrt)

    return degree_inv_sqrt @ adj_matrix_with_self_loops @ degree_inv_sqrt

def regularize_normalized_adjacency_matrix(norm_adj_matrix, a):
    """
    Регуляризует нормализованную матрицу смежности.

    :param norm_adj_matrix: Нормализованная матрица смежности в формате numpy.ndarray.
    :param a: Параметр регуляризации, принадлежащий интервалу (0, 1).
    :return: Регуляризованная матрица смежности в формате numpy.ndarray.
    """
    num_nodes = norm_adj_matrix.shape[0]
    I = np.eye(num_nodes)
    return (a / 2) * (I + norm_adj_matrix)

# Пример использования
if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float)
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    num_nodes = data.num_nodes
    adj_matrix = create_adjacency_matrix(data.edge_index, num_nodes)

    # Нормализация матрицы смежности
    norm_adj_matrix = normalize_adjacency_matrix(adj_matrix)

    # Регуляризация нормализованной матрицы смежности
    a = 0.5  # Пример значения параметра регуляризации
    reg_norm_adj_matrix = regularize_normalized_adjacency_matrix(norm_adj_matrix, a)

    node_features = data.x
    node_labels = data.y

    print("Матрица смежности:")
    print(adj_matrix)

    print("Нормализованная матрица смежности:")
    print(norm_adj_matrix)

    print("Регуляризованная нормализованная матрица смежности:")
    print(reg_norm_adj_matrix)

    print("Матрица признаков узлов:")
    print(node_features)

    print("Матрица меток узлов:")
    print(node_labels)