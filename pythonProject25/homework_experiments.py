"""
Задание 3: Эксперименты
- Гиперпараметры: lr, batch_size, optimizers
- Feature engineering: полиномиальные, взаимодействия, статистические признаки
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from homework_model_modification import LinearRegressionModel

# Создание папки для графиков
os.makedirs("plots", exist_ok=True)

# --- Исследование гиперпараметров ---
def hyperparameter_search(learning_rates, batch_sizes, optimizers):
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

    results = {}
    for opt_name in optimizers:
        for lr in learning_rates:
            for bs in batch_sizes:
                model = LinearRegressionModel(X.shape[1])
                if opt_name == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                elif opt_name == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                else:
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

                criterion = torch.nn.MSELoss()
                loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=bs, shuffle=True)

                for epoch in range(50):
                    for xb, yb in loader:
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                key = f"{opt_name}_lr{lr}_bs{bs}"
                results[key] = val_loss
                print(f"{key}: {val_loss:.4f}")

    # Визуализация результатов
    names = list(results.keys())
    values = list(results.values())
    plt.figure(figsize=(10,5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("Validation Loss")
    plt.tight_layout()
    plt.savefig("plots/hyperparam_search.png")
    plt.close()

# --- Feature Engineering ---
def feature_engineering():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Полиномиальные признаки
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Статистические признаки: среднее и дисперсия по строкам
    mean_feat = X.mean(axis=1, keepdims=True)
    var_feat = X.var(axis=1, keepdims=True)
    X_stats = np.hstack([X, mean_feat, var_feat])

    for name, X_new in [('original', X), ('poly', X_poly), ('stats', X_stats)]:
        X_train, X_val, y_train, y_val = train_test_split(X_new, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = LinearRegressionModel(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        print(f"{name} features: val_loss={val_loss:.4f}")

if __name__ == '__main__':
    hyperparameter_search(
        learning_rates=[0.001, 0.01, 0.1],
        batch_sizes=[16, 32, 64],
        optimizers=['SGD', 'Adam', 'RMSprop']
    )
    feature_engineering()