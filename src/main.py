import torch
from torch_geometric.data import Data
from src.data.preprocessing import create_adjacency_matrix, normalize_adjacency_matrix, \
    regularize_normalized_adjacency_matrix
from src.utils.visualization import visualize_graph
from src.models.cgnn import CGNN

def main():
    edge_index = torch.tensor([[0, 1, 2],
                                    [1, 2, 0]], dtype=torch.long)
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float)
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    num_nodes = data.num_nodes
    adj_matrix = create_adjacency_matrix(data.edge_index, num_nodes)
    norm_adj_matrix = normalize_adjacency_matrix(adj_matrix)
    a = 0.5
    reg_norm_adj_matrix = regularize_normalized_adjacency_matrix(norm_adj_matrix, a)

    visualize_graph(adj_matrix, data.x)

    print("Матрица смежности:")
    print(adj_matrix)
    print("Нормализованная матрица смежности:")
    print(norm_adj_matrix)
    print("Регуляризованная нормализованная матрица смежности:")
    print(reg_norm_adj_matrix)

    print("Начальные признаки узлов:")
    print(data.x)

    input_dim = data.x.shape[1]
    hidden_dim = 8
    output_dim = 2
    num_layers = 2
    model = CGNN(input_dim, hidden_dim, output_dim, num_layers)

    initial_representation = model(data.x, data.edge_index, reg_norm_adj_matrix)

    print("Начальное представление узлов после нейронного кодировщика:")
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    print(initial_representation)

if __name__ == "__main__":
    main()
