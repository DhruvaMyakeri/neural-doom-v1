import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# LOAD DATA
# ---------------------------

data = np.load("doom_dataset.npz")

frames = data["frames"]
actions = data["actions"]

# ---------------------------
# BASIC INFO
# ---------------------------

print("Frames shape:", frames.shape)
print("Actions shape:", actions.shape)

print("Frame dtype:", frames.dtype)
print("Action dtype:", actions.dtype)

print("Frame min:", frames.min())
print("Frame max:", frames.max())

print("Number of unique action vectors:", len(np.unique(actions, axis=0)))

# ---------------------------
# VISUAL CHECK
# ---------------------------

# Convert first frame back to 0-255 uint8
img = (frames[0] * 255).astype(np.uint8)

# Upscale for display only
big = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

cv2.imshow("Upscaled Doom Frame", big)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------
# MOTION CHECK
# ---------------------------

diff = np.mean(np.abs(frames[1] - frames[0]))
print("Mean pixel difference between frame 0 and 1:", diff)

# ---------------------------
# PLOT A FEW FRAMES INLINE
# ---------------------------

plt.figure(figsize=(10, 4))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(frames[i])
    plt.axis("off")
    plt.title(f"Frame {i}")

plt.show()
