import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DoomWorldModelDataset
from model import DoomWorldModel
from tqdm import tqdm
import os

# ---------------------------
# DEVICE
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# DATA
# ---------------------------

dataset = DoomWorldModelDataset("doom_dataset.npz", sequence_length=5)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# ---------------------------
# MODEL
# ---------------------------

model = DoomWorldModel(action_dim=3).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# TRAIN LOOP
# ---------------------------

epochs = 10
save_path = "doom_world_model.pth"

for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader)

    for frame_seq, action, next_frame in loop:

        frame_seq = frame_seq.to(device)
        action = action.to(device)
        next_frame = next_frame.to(device)

        optimizer.zero_grad()

        output = model(frame_seq, action)

        loss = criterion(output, next_frame)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)

print("Training complete.")
