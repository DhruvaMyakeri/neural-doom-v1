import torch
import numpy as np
import cv2
import keyboard
from dataset import DoomWorldModelDataset
from model import DoomWorldModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DoomWorldModelDataset("doom_dataset.npz", sequence_length=5)
idx = np.random.randint(0, len(dataset))
frame_seq, _, _ = dataset[idx]
frame_seq = frame_seq.unsqueeze(0).to(device)

model = DoomWorldModel(action_dim=3).to(device)
model.load_state_dict(torch.load("doom_world_model.pth", map_location=device))
model.eval()

print("Hold A / D / W. Press Q to quit.")

cv2.namedWindow("Neural Doom")

while True:

    action_vec = [0, 0, 0]

    if keyboard.is_pressed('a'):
        action_vec[0] = 1
    if keyboard.is_pressed('d'):
        action_vec[1] = 1
    if keyboard.is_pressed('w'):
        action_vec[2] = 1
    if keyboard.is_pressed('q'):
        break

    action_tensor = torch.tensor(action_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_frame = model(frame_seq, action_tensor)

    pred_img = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    display = (pred_img * 255).astype(np.uint8)
    display = cv2.resize(display, (512, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Neural Doom", display)
    cv2.waitKey(1)

    pred_frame = pred_frame.unsqueeze(1)
    frame_seq = torch.cat([frame_seq[:, 1:], pred_frame], dim=1)

cv2.destroyAllWindows()
