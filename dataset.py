import numpy as np
import torch
from torch.utils.data import Dataset


class DoomWorldModelDataset(Dataset):
    def __init__(self, npz_path, sequence_length=5):
        data = np.load(npz_path)

        self.frames = data["frames"]      # (N, 64, 64, 3)
        self.actions = data["actions"]    # (N, 3)

        self.seq_len = sequence_length
        self.length = len(self.frames) - self.seq_len - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get frame sequence
        frame_seq = self.frames[idx : idx + self.seq_len]
        next_frame = self.frames[idx + self.seq_len]

        # Action at final timestep
        action = self.actions[idx + self.seq_len - 1]

        # Convert to torch tensors
        frame_seq = torch.tensor(frame_seq, dtype=torch.float32).permute(0, 3, 1, 2)
        next_frame = torch.tensor(next_frame, dtype=torch.float32).permute(2, 0, 1)
        action = torch.tensor(action, dtype=torch.float32)

        return frame_seq, action, next_frame
