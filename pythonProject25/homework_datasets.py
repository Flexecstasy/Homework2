"""
Задание 2: Работа с датасетами
- CustomDataset для CSV
- Эксперименты на кастомных датасетах
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from homework_model_modification import LinearRegressionModel, LogisticRegressionModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

os.makedirs("data", exist_ok=True)

class CSVDataset(Dataset):
    def __init__(self, file_path, target_column):
        self.df = pd.read_csv(file_path)
        self.target_column = target_column

        # Обработка целевой переменной
        self.y = self.df[target_column].values

        # Определяем признаки
        self.X = self.df.drop(columns=[target_column])

        # Кодируем категориальные признаки
        for col in self.X.select_dtypes(include=['object', 'category']).columns:
            self.X[col] = LabelEncoder().fit_transform(self.X[col])

        self.X = self.X.values.astype(np.float32)
        self.y = self.y.astype(np.float32)

        # Нормализация
        self.X = StandardScaler().fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def experiment_regression():
    logging.info("Регрессия: California Housing")
    df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
    df = df.dropna()
    df.to_csv("data/california.csv", index=False)

    dataset = CSVDataset("data/california.csv", target_column="median_house_value")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearRegressionModel(dataset.X.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(201):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        logging.info(f"Epoch {epoch}: loss={total_loss:.4f}")

    plt.plot(losses)
    plt.title("California Housing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("plots/california_loss.png")
    plt.close()


def experiment_classification():
    logging.info("Классификация: Titanic")
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df = df.dropna()
    df.to_csv("data/titanic.csv", index=False)

    dataset = CSVDataset("data/titanic.csv", target_column="Survived")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LogisticRegressionModel(dataset.X.shape[1], 2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(201):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            y_batch = y_batch.long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        logging.info(f"Epoch {epoch}: loss={total_loss:.4f}")

    plt.plot(losses)
    plt.title("Titanic Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("plots/titanic_loss.png")
    plt.close()

if __name__ == '__main__':
    experiment_regression()
    experiment_classification()