import torch
import torch.nn as nn


# ---------------------------
# CNN ENCODER
# ---------------------------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16 → 8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 8 → 4
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # 256 channels × 4 × 4
        self.fc = nn.Linear(256 * 4 * 4, 1024)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ---------------------------
# DECODER
# ---------------------------

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(1024, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4 → 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 32 → 64
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = self.deconv(x)
        return x


# ---------------------------
# WORLD MODEL
# ---------------------------

class DoomWorldModel(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()

        self.encoder = Encoder()

        self.action_embed = nn.Linear(action_dim, 128)

        self.lstm = nn.LSTM(
            input_size=1024 + 128,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )

        self.decoder_input = nn.Linear(512, 1024)
        self.decoder = Decoder()

    def forward(self, frame_seq, action):
        """
        frame_seq: (B, T, 3, 64, 64)
        action:    (B, action_dim)
        """

        B, T, C, H, W = frame_seq.shape

        # Encode each frame independently
        frame_seq = frame_seq.view(B * T, C, H, W)
        latent = self.encoder(frame_seq)          # (B*T, 1024)
        latent = latent.view(B, T, -1)            # (B, T, 1024)

        # Embed action
        action_embed = self.action_embed(action)  # (B, 128)

        # Repeat action across sequence length
        action_embed = action_embed.unsqueeze(1).repeat(1, T, 1)

        # Concatenate latent + action
        lstm_input = torch.cat([latent, action_embed], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Take last timestep output
        final_hidden = lstm_out[:, -1, :]         # (B, 512)

        # Map to decoder input
        decoder_latent = self.decoder_input(final_hidden)

        # Decode image
        predicted_frame = self.decoder(decoder_latent)

        return predicted_frame
