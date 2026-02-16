# Neural Doom: V1 🧠🔫

A neural world model that learns to simulate _Doom_ (1993) purely from pixels.

**This is NOT a game engine.** This is a neural network "hallucinating" the game dynamics in real-time. By training on gameplay sequences, the model learns to approximate the transition function of the environment without ever seeing the underlying source code.

---

## 🌟 Project Overview

This project implements a **Generative World Model** inspired by the concept of "Neural Simulation." Instead of using the actual Doom engine to render frames, we train a **CNN-LSTM** neural network to predict the _next frame_ given a history of frames and a specific action.

During the "Rollout" phase, the real game engine is completely disconnected. The user provides inputs (A/D/W keys), and the AI generates the visual response on the fly based on what it learned about Doom's physics and geometry.

### Current Features (Version 1.0)

- **Resolution:** 64x64 RGB (Optimized for neural stability)
- **Architecture:** Convolutional Encoder + LSTM Temporal Memory + Transposed Conv Decoder
- **Action Space:** 3-Button Multi-binary (Move Left, Move Right, Attack)
- **Inference:** Real-time interactive simulation using the `keyboard` library.

---

## 🧠 Architecture & Theory

The model operates in three distinct stages to process space and time:

1.  **Visual Encoder (CNN):** A 4-layer Convolutional Neural Network that compresses a sequence of 5 frames into a high-dimensional latent vector (1024-dim), capturing spatial features like walls and weapon positions.
2.  **Temporal Core (LSTM):** A Long Short-Term Memory network that tracks motion and "physics" state over time. It combines the visual latent with a 128-dim action embedding.
3.  **Visual Decoder (Deconv):** A Transposed Convolutional network that "renders" the hidden state of the LSTM back into a 64x64 RGB image.

**Loss Function:** Mean Squared Error (MSE) between the predicted frame and the ground truth frame.

---

## 🛠️ Installation & Environment

### Prerequisites

- **Python 3.10** (Required for ViZDoom binary compatibility)
- **NVIDIA GPU** with CUDA 12.1+ support

### Setup

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/neural-doom-v1.git](https://github.com/YOUR_USERNAME/neural-doom-v1.git)
    cd neural-doom-v1
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**

    ```bash
    # Install PyTorch with CUDA support
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    # Install additional libraries
    pip install numpy==1.26.4 opencv-python tqdm gymnasium vizdoom keyboard
    ```

---

## 🚀 How to Run

### 1. Data Collection

Collect 100,000 transitions of raw gameplay data using a random exploration policy.

```bash
python record_data.py
2. Training
Train the CNN-LSTM "Brain" to understand Doom's visual transitions.

Bash

python train.py
3. Neural Rollout (The Simulator)
Run the interactive simulation where the AI generates the world in response to your keys.

Bash

python rollout.py
A: Move Left

D: Move Right

W: Attack

Q: Quit

📝 Files in this Repo
record_data.py: Connects to ViZDoom and saves frames/actions to .npz.

dataset.py: Handles sliding-window sequence creation for temporal training.

model.py: Defines the Encoder, Decoder, and LSTM architecture.

train.py: The optimization loop.

rollout.py: The interactive real-time neural simulator.

inspect_dataset.py: Utility to verify data integrity and resolution.

🔮 Roadmap
V2: Expand action space to 7 buttons (Forward, Backward, Turning).

V2: Implement Perceptual Loss (L1 + SSIM) for sharper, less blurry frames.

V3: Transition to Latent World Modeling (training on VAE latents rather than raw pixels).

Author: [Dhruva Myaekri]

Credits: Built with ViZDoom. Inspired by World Models (Ha & Schmidhuber, 2018).
```
