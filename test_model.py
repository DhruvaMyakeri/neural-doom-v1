import torch
from torch.utils.data import DataLoader
from dataset import DoomWorldModelDataset
from model import DoomWorldModel

# ---------------------------
# DEVICE
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# LOAD DATASET
# ---------------------------

dataset = DoomWorldModelDataset("doom_dataset.npz", sequence_length=5)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Take one batch
frame_seq, action, next_frame = next(iter(dataloader))

frame_seq = frame_seq.to(device)
action = action.to(device)
next_frame = next_frame.to(device)

print("Frame seq shape:", frame_seq.shape)
print("Action shape:", action.shape)
print("Target shape:", next_frame.shape)

# ---------------------------
# INIT MODEL
# ---------------------------

model = DoomWorldModel(action_dim=3).to(device)

# Forward pass
with torch.no_grad():
    output = model(frame_seq, action)

print("Output shape:", output.shape)
