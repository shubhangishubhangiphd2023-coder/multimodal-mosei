import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.mosei_dataset import MOSEIDataset
from models.fusion import End2EndMulT

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = MOSEIDataset("data/features/train.h5")
val_ds   = MOSEIDataset("data/features/val.h5")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8)

model = End2EndMulT().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(20):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["vision"].to(device)
        )
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss {loss.item():.4f}")
