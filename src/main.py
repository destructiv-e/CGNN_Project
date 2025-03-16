import torch
from torch_geometric.data import Data
from src.data.preprocessing import create_adjacency_matrix, normalize_adjacency_matrix, \
    regularize_normalized_adjacency_matrix
from src.utils.visualization import visualize_graph
from src.models.cgnn import CGNN
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    # Создаем тестовый граф
    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 2, 5, 6, 4, 5],
                               [1, 2, 3, 1, 0, 3, 4, 6, 5, 5, 6]],
                              dtype=torch.long)

    # Добавляем новые признаки для узлов
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0],
                      [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
                      [6.0, 7.0], [7.0, 8.0]],  # Новые узлы 5 и 6
                     dtype=torch.float)

    # Добавляем новые метки для узлов
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)  # Новые метки для узлов 5 и 6

    # Создаем объект Data
    data = Data(x=x, edge_index=edge_index, y=y)



    # Создаем PDF-файл для сохранения результатов
    with PdfPages('output.pdf') as pdf:
        # Выводим начальные данные
        print("Начальные данные:")
        print("Список ребер (edge_index):")
        print(data.edge_index)
        print("Признаки узлов (x):")
        print(data.x)
        print("Метки узлов (y):")
        print(data.y)

        # Создаем таблицу для начальных данных
        initial_data = {
            "Edge Index": [str(data.edge_index.numpy())],
            "Node Features (x)": [str(data.x.numpy())],
            "Node Labels (y)": [str(data.y.numpy())]
        }
        df_initial = pd.DataFrame(initial_data)
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.table(cellText=df_initial.values, colLabels=df_initial.columns, loc='center')
        plt.title("Начальные данные")
        pdf.savefig()
        plt.close()

        # Создаем матрицу смежности
        num_nodes = data.num_nodes
        adj_matrix = create_adjacency_matrix(data.edge_index, num_nodes)
        print("\nМатрица смежности:")
        print(adj_matrix)

        # Создаем таблицу для матрицы смежности
        df_adj = pd.DataFrame(adj_matrix)
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.table(cellText=df_adj.values, colLabels=df_adj.columns, loc='center')
        plt.title("Матрица смежности")
        pdf.savefig()
        plt.close()

        # Нормализуем матрицу смежности
        norm_adj_matrix = normalize_adjacency_matrix(adj_matrix)
        print("\nНормализованная матрица смежности:")
        print(norm_adj_matrix)

        # Создаем таблицу для нормализованной матрицы смежности
        df_norm_adj = pd.DataFrame(norm_adj_matrix)
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.table(cellText=df_norm_adj.values, colLabels=df_norm_adj.columns, loc='center')
        plt.title("Нормализованная матрица смежности")
        pdf.savefig()
        plt.close()

        # Регуляризуем нормализованную матрицу смежности
        a = 0.5
        reg_norm_adj_matrix = regularize_normalized_adjacency_matrix(norm_adj_matrix, a)
        print("\nРегуляризованная нормализованная матрица смежности:")
        print(reg_norm_adj_matrix)

        # Создаем таблицу для регуляризованной нормализованной матрицы смежности
        df_reg_norm_adj = pd.DataFrame(reg_norm_adj_matrix)
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.table(cellText=df_reg_norm_adj.values, colLabels=df_reg_norm_adj.columns, loc='center')
        plt.title("Регуляризованная нормализованная матрица смежности")
        pdf.savefig()
        plt.close()

        # Визуализируем граф
        visualize_graph(adj_matrix, title="Original Adjacency Matrix")

        # Инициализируем модель CGNN с GCNEncoder
        input_dim = data.x.shape[1]
        hidden_dim = 8  # Увеличено до 8
        output_dim = 2
        num_layers = 2  # Уменьшено до 2

        # Преобразуем матрицу смежности в тензор
        reg_norm_adj_matrix_tensor = torch.tensor(reg_norm_adj_matrix, dtype=torch.float32)

        # Инициализируем модель
        model = CGNN(input_dim, hidden_dim, output_dim, num_layers)

        # Определяем функцию потерь и оптимизатор
        criterion = torch.nn.CrossEntropyLoss()  # Функция потерь для классификации
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Оптимизатор Adam

        # Цикл обучения
        num_epochs = 2000  # Количество эпох
        losses = []

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            # Прямой проход
            output = model(data.x, data.edge_index, reg_norm_adj_matrix_tensor)

            # Вычисление потерь
            loss = criterion(output, y)
            losses.append(loss.item())

            # Обратный проход
            loss.backward()
            optimizer.step()

            # Вывод лосса
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Alpha: {model.alpha.item():.4f}")

        # for epoch in range(num_epochs):
        #     # Переводим модель в режим обучения
        #     model.train()
        #
        #     # Прямой проход: получаем предсказания
        #     output = model(data.x, data.edge_index, reg_norm_adj_matrix_tensor)
        #
        #     # Вычисляем функцию потерь
        #     loss = criterion(output, y)
        #     losses.append(loss.item())
        #
        #     # Обратный проход: вычисляем градиенты
        #     optimizer.zero_grad()  # Обнуляем градиенты
        #     loss.backward()  # Вычисляем градиенты
        #     optimizer.step()  # Обновляем параметры
        #
        #     # Выводим loss каждые 10 эпох
        #     if (epoch + 1) % 10 == 0:
        #         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Создаем таблицу для потерь
        df_losses = pd.DataFrame(losses, columns=["Loss"])
        plt.figure(figsize=(10, 4))
        plt.plot(df_losses.index, df_losses['Loss'], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        pdf.savefig()
        plt.close()

        # Переводим модель в режим оценки
        model.eval()

        # Получаем итоговый результат после ODE Solver и декодера
        output = model(data.x, data.edge_index, reg_norm_adj_matrix_tensor)
        print("\nИтоговый результат (вероятности классов):")
        print(output)

        # Преобразуем вероятности в предсказанные классы
        predicted_classes = torch.argmax(output, dim=1)
        print("\nПредсказанные классы:")
        print(predicted_classes)

        # Оцениваем точность модели
        accuracy = (predicted_classes == y).float().mean()
        print("\nТочность модели (accuracy):")
        print(accuracy.item())

        # Создаем таблицу для итоговых результатов
        final_results = {
            "Predicted Classes": [str(predicted_classes.numpy())],
            "Accuracy": [accuracy.item()]
        }
        df_final = pd.DataFrame(final_results)
        plt.figure(figsize=(10, 4))
        plt.axis('off')
        plt.table(cellText=df_final.values, colLabels=df_final.columns, loc='center')
        plt.title("Итоговые результаты")
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    main()