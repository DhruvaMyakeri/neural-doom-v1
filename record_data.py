import os
import cv2
import numpy as np
import vizdoom as vzd
from tqdm import tqdm

# ---------------------------
# INIT GAME
# ---------------------------

game = vzd.DoomGame()

# In ViZDoom 1.3.0, vzd.scenarios_path points directly to the folder 
# containing basic.cfg, deadly_corridor.cfg, etc.
scenario_path = os.path.join(vzd.scenarios_path, "basic.cfg")

try:
    game.load_config(scenario_path)
except Exception as e:
    print(f"Error loading config: {e}")
    print(f"Attempted path: {scenario_path}")
    exit()

# Set window to False for faster data collection
game.set_window_visible(False)
game.init()

n_buttons = game.get_available_buttons_size()
print(f"Successfully initialized ViZDoom version: {vzd.__version__}")
print("Number of buttons detected:", n_buttons)

# ---------------------------
# STORAGE
# ---------------------------

frames = []
actions = []

# 100k steps is ~5GB of RAM. If you hit memory errors, lower this to 50000.
num_steps = 100000 

# ---------------------------
# RECORD LOOP
# ---------------------------

print(f"Starting data collection for {num_steps} steps...")

for _ in tqdm(range(num_steps)):

    if game.is_episode_finished():
        game.new_episode()

    state = game.get_state()

    if state is None:
        continue

    # Get raw frame (C, H, W)
    frame = state.screen_buffer

    # Convert to (H, W, C) for OpenCV/Standard processing
    frame = np.transpose(frame, (1, 2, 0))

    # Resize to 64x64 for efficient AI training
    frame = cv2.resize(frame, (64, 64))

    # Normalize to 0-1 and cast to float32
    frame = frame.astype(np.float32) / 255.0

    # Random multi-binary action (simulating random exploration)
    action = np.random.randint(0, 2, size=n_buttons).tolist()

    frames.append(frame)
    actions.append(action)

    game.make_action(action)

game.close()

# ---------------------------
# SAVE DATASET
# ---------------------------

print("\nProcessing arrays for saving...")
frames = np.array(frames, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)

# Save as a compressed numpy file to save disk space
save_path = "doom_dataset.npz"
np.savez_compressed(
    save_path,
    frames=frames,
    actions=actions
)

print(f"\nDataset saved successfully to {save_path}")
print("Final Frames shape:", frames.shape)
print("Final Actions shape:", actions.shape)