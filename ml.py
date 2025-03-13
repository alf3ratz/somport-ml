import random

num_frames = 392  # Количество кадров в видео
labels = [random.uniform(0.1, 1.0) for _ in range(num_frames)]

with open('train_labels.txt', 'w') as f:
    for label in labels:
        f.write(f'{label}\n')

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Определим класс для нашего датасета
class FishDataset(Dataset):
    def __init__(self, video_path, labels_path, transform=None):
        self.video_path = video_path
        self.labels_path = labels_path
        self.transform = transform

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Файл меток {labels_path} не найден.")

        try:
            self.labels = np.loadtxt(labels_path)
        except Exception as e:
            raise ValueError(f"Ошибка при чтении файла меток {labels_path}: {e}")

        if not np.all((self.labels >= 0.1) & (self.labels <= 1.0)):
            raise ValueError("Метки должны находиться в диапазоне от 0.1 до 1.0.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видео {video_path} не найдено.")

        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"Не удалось открыть видео {video_path}.")

        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        if len(self.labels) != self.length:
            raise ValueError(
                f"Количество кадров в видео ({self.length}) не совпадает с количеством меток ({len(self.labels)})")

        print(f"Количество кадров в видео: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"Индекс {idx} выходит за пределы допустимого диапазона (0-{self.length - 1})")

        self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.video.read()

        if not ret:
            raise ValueError(f"Не удалось прочитать кадр {idx} из видео {self.video_path}")

        label = self.labels[idx]

        if self.transform:
            frame = self.transform(frame)

        return frame, torch.tensor(label, dtype=torch.float32)


# Улучшенная сверточная сеть
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Приводит к размеру (batch, channels, 1, 1)

        self.fc1 = nn.Linear(32, 128)  # Теперь 32 нейрона после Global Pooling
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.global_pool(x)  # Выход (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # Превращаем в (batch, 32)

        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Преобразования изображений
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
])

# Путь к файлам
train_video_path = 'train_video.mp4'
train_labels_path = 'train_labels.txt'

try:
    train_dataset = FishDataset(video_path=train_video_path, labels_path=train_labels_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
    print("Датасет и DataLoader успешно инициализированы.")
except Exception as e:
    print(f"Ошибка при инициализации датасета: {e}")
    train_loader = None

if train_loader is None:
    print("Не удалось инициализировать DataLoader. Программа завершена.")
else:
    model = SimpleCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (frames, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(frames)

            outputs = outputs.squeeze(1)  # Убираем лишнее измерение
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'fish_model.pth')
    print("Модель успешно сохранена.")
