import pandas as pd
import torch
from torch.utils.data import DataLoader
from homework_datasets import CSVDataset


# Вспомогательная функция: создаёт dummy CSV-файл с числовыми и категориальными данными
def create_dummy_csv(path):
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': ['A', 'B', 'A', 'C'],
        'feature3': [0, 1, 1, 0],
        'target': [10.0, 20.0, 30.0, 40.0]
    })
    df.to_csv(path, index=False)


# Тест: проверка длины и формы элементов датасета
def test_dataset_structure(tmp_path):
    csv_file = tmp_path / "test.csv"
    create_dummy_csv(str(csv_file))

    dataset = CSVDataset(str(csv_file), target_column="target")

    assert len(dataset) == 4  # количество строк

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == 3  # 3 признака после кодирования
    assert y.ndim == 0 or y.shape == torch.Size([])  # скаляр


# Тест: проверка batch'ей DataLoader'а
def test_dataloader_batches(tmp_path):
    csv_file = tmp_path / "test.csv"
    create_dummy_csv(str(csv_file))

    dataset = CSVDataset(str(csv_file), target_column="target")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 2  # 4 элемента по 2 в каждом батче

    for X_batch, y_batch in batches:
        assert X_batch.shape[0] == 2  # batch size
        assert X_batch.shape[1] == 3  # num features
        assert y_batch.shape[0] == 2


# Тест: проверка, что нормализация даёт среднее ~0
def test_feature_normalization(tmp_path):
    csv_file = tmp_path / "test.csv"
    create_dummy_csv(str(csv_file))

    dataset = CSVDataset(str(csv_file), target_column="target")

    features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    means = features.mean(dim=0)
    assert torch.all(torch.abs(means) < 1e-5)  # близко к 0
