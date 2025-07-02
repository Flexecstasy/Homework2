"""
Задание 1: Модификация моделей
- LinearRegression с L1, L2 регуляризацией и early stopping на diabetes
- LogisticRegression с мультиклассовой поддержкой и метриками на breast_cancer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Модель линейной регрессии ---
class LinearRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear_model(l1_lambda=0.0, l2_lambda=0.01, patience=10):
    # Загрузка diabetes датасета
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = LinearRegressionModel(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)

        # Добавление регуляризации
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Валидация
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor)
            val_losses.append(val_loss.item())

        logging.info(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

        # Early stopping
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), "models/best_linear_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

    # Визуализация
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Linear Regression Loss")
    plt.savefig(os.path.join(PLOT_DIR, "linear_loss.png"))
    plt.close()

# --- Модель логистической регрессии ---
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_logistic_model():
    breast = datasets.load_breast_cancer()
    X, y = breast.data, breast.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = LogisticRegressionModel(X.shape[1], 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

        logging.info(f"Epoch {epoch}: loss={loss.item():.4f}")

    # Оценка метрик
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).numpy()
        y_true = y_test_tensor.numpy()
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        roc_auc = roc_auc_score(y_true, preds)
        logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
        plt.close()

if __name__ == '__main__':
    train_linear_model()
    train_logistic_model()