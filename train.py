import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits

from mlp import SimpleMLP

def main():
    # 1) load_digits
    digits = load_digits()
    X = digits.images  # shape: (1797,8,8)
    y = digits.target  # shape: (1797,)

    # 2) "1,2"만 추출
    mask = np.isin(y, [1,2])
    X = X[mask]
    y = y[mask]

    # 3) 라벨 맵핑: 원본 1->0, 2->1
    y_new = np.where(y==1, 0, 1).astype(np.int64)

    # 4) Flatten + 정규화
    X = X.reshape(-1,64).astype(np.float32)
    X /= 16.0  # digits는 0..16 범위

    # 5) train/val 분할
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train = X[train_idx]
    y_train = y_new[train_idx]
    X_val   = X[val_idx]
    y_val   = y_new[val_idx]

    # 6) Tensor 변환
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    # 7) 모델 생성 (16,16,16)
    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 8) 학습
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t)
                preds = val_out.argmax(dim=1)
                acc = (preds == y_val_t).float().mean().item()
            print(f"Epoch[{epoch+1}/{epochs}] loss={loss.item():.4f} "
                  f"val_loss={val_loss.item():.4f} acc={acc:.2f}")

    # 9) 모델 저장
    torch.save(model.state_dict(), "model.pt")
    print("1 vs 2 분류 모델 학습 완료 (16,16,16). model.pt 저장.")

if __name__ == "__main__":
    main()
