import torch
from torchdiffeq import odeint
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Нейронная сеть, которая определяет динамику системы (правая часть ODE).
    """

    def __init__(self, adj_matrix, encoder_output, alpha):
        super(ODEFunc, self).__init__()
        self.adj_matrix = adj_matrix  # Матрица смежности
        self.encoder_output = encoder_output  # Начальные представления от кодировщика
        self.alpha = alpha  # Обучаемый параметр для регулирования динамики

    def forward(self, t, h):
        """
        Вычисляет производную dh/dt.
        :param t: Время (не используется, но требуется для интерфейса ODE Solver).
        :param h: Текущие представления узлов.
        :return: Производная dh/dt.
        """
        # Вычисляем alpha * (A - I)H(t) + E
        return self.alpha * (self.adj_matrix - torch.eye(self.adj_matrix.shape[0])) @ h + self.encoder_output


def solve_ode(adj_matrix, encoder_output, t, alpha):
    """
    Решает ODE для заданной матрицы смежности и начальных представлений.
    :param adj_matrix: Матрица смежности.
    :param encoder_output: Начальные представления узлов (E).
    :param t: Временные точки для решения ODE.
    :param alpha: Обучаемый параметр для регулирования динамики.
    :return: Решение ODE (H(t)).
    """
    # Преобразуем матрицу смежности и начальные представления в тензоры
    adj_matrix = adj_matrix.clone().detach().float()
    encoder_output = torch.tensor(encoder_output, dtype=torch.float32)

    # Создаем объект ODEFunc
    ode_func = ODEFunc(adj_matrix, encoder_output, alpha)

    # Начальные условия (H(0) = E)
    h0 = encoder_output

    # Решаем ODE
    h = odeint(ode_func, h0, t, method='dopri5')  # Используем метод Dormand-Prince (DOPRI5)
    return h


# class ODEFunc(nn.Module):
#     """
#     Нейронная сеть, которая определяет динамику системы (правая часть ODE).
#     """
#
#     def __init__(self, adj_matrix, encoder_output):
#         super(ODEFunc, self).__init__()
#         self.adj_matrix = adj_matrix  # Матрица смежности
#         self.encoder_output = encoder_output  # Начальные представления от кодировщика
#
#     def forward(self, t, h):
#         """
#         Вычисляет производную dh/dt.
#         :param t: Время (не используется, но требуется для интерфейса ODE Solver).
#         :param h: Текущие представления узлов.
#         :return: Производная dh/dt.
#         """
#         # Вычисляем (A - I)H(t) + E
#         return (self.adj_matrix - torch.eye(self.adj_matrix.shape[0])) @ h + self.encoder_output
#
#
# def solve_ode(adj_matrix, encoder_output, t):
#     """
#     Решает ODE для заданной матрицы смежности и начальных представлений.
#     :param adj_matrix: Матрица смежности.
#     :param encoder_output: Начальные представления узлов (E).
#     :param t: Временные точки для решения ODE.
#     :return: Решение ODE (H(t)).
#     """
#     # Преобразуем матрицу смежности и начальные представления в тензоры
#     adj_matrix = adj_matrix.clone().detach().float()
#     encoder_output = torch.tensor(encoder_output, dtype=torch.float32)
#
#     # Создаем объект ODEFunc
#     ode_func = ODEFunc(adj_matrix, encoder_output)
#
#     # Начальные условия (H(0) = E)
#     h0 = encoder_output
#
#     # Решаем ODE
#     h = odeint(ode_func, h0, t, method='dopri5')  # Используем метод Dormand-Prince (DOPRI5)
#     return h
