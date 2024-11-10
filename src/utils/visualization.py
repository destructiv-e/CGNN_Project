import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_graph(adj_matrix, title="Graph"):
    """
    Визуализирует граф, используя матрицу смежности.

    :param adj_matrix: Матрица смежности в формате numpy.ndarray.
    :param title: Заголовок графика.
    """
    # Создаем граф из матрицы смежности
    G = nx.from_numpy_array(adj_matrix)

    # Визуализируем граф
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Расположение узлов
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold', edge_color='gray')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1, 2],
                               [1, 2, 0]], dtype=torch.long)
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float)
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    num_nodes = data.num_nodes
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edge_index.T:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1

    visualize_graph(adj_matrix, title="Original Adjacency Matrix")