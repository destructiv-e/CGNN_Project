# Continuous Graph Neural Network (CGNN)

## Описание

Этот проект реализует модель Continuous Graph Neural Network (CGNN), описанную в статье [Continuous Graph Neural Network](https://arxiv.org/abs/2003.04407) авторов Louis-Pascal A. C. Xhonneux, Meng Qu и Jian Tang. CGNN использует непрерывные дифференциальные уравнения для моделирования динамики графов, что позволяет более гибко управлять процессом обучения и может привести к более стабильным и точным моделям.

## Структура проекта

```plaintext
cgnn_project/
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cgnn.py
│   │   └── ode_solver.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_cgnn.py
│   └── test_data.py
│
├── configs/
│   ├── config.yaml
│   └── hyperparameters.json
│
├── logs/
│
├── README.md
├── requirements.txt
└── .gitignore

```
