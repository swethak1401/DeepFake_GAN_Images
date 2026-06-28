# DeepFake GAN — Synthetic Face Image Generation

A deep learning project that trains a **Generative Adversarial Network (GAN)** in **PyTorch** to synthesize photorealistic, "deepfake" human face images from random noise vectors. The model learns the underlying distribution of a facial-image dataset and generates entirely new, non-existent faces.

> **Note:** This project is built for educational and research purposes — exploring deep generative modeling and computer vision. Synthetic faces should not be used to deceive, impersonate, or misrepresent real individuals.

---

## Overview

GANs consist of two neural networks trained adversarially:

- **Generator (G):** maps a random latent vector `z` to a synthetic image, learning to produce faces that look real.
- **Discriminator (D):** classifies images as *real* (from the dataset) or *fake* (from the generator).

The two networks compete: the generator improves at fooling the discriminator, while the discriminator improves at telling real from fake. At convergence, the generator produces realistic face images.

This implementation follows a **Deep Convolutional GAN (DCGAN)**-style architecture using convolutional and transposed-convolutional layers.

---

## Features

- End-to-end GAN training pipeline implemented in **PyTorch** + **torchvision**.
- Convolutional Generator and Discriminator networks.
- Adversarial training loop with Binary Cross-Entropy loss and the Adam optimizer.
- GPU-accelerated training (developed on Kaggle GPU runtime).
- Visualization of generated faces and training progress with `matplotlib` / `torchvision.utils`.

---

## Tech Stack

| Category        | Tools |
|-----------------|-------|
| Language        | Python 3.10 |
| Deep Learning   | PyTorch, torchvision |
| Data / Imaging  | NumPy, Pillow (PIL) |
| Visualization   | Matplotlib |
| Environment     | Jupyter Notebook (Kaggle, GPU) |

---

## Dataset

The model is trained on a **facial-image dataset** loaded via `torchvision` (the notebook was developed on a Kaggle dataset).

> **TODO:** Add the exact dataset name and link here (e.g., the Kaggle dataset URL). The raw dataset is intentionally **not** committed to the repo — see [`.gitignore`](.gitignore). Download it separately and place it under a local `data/` directory.

Images are resized, center-cropped, and normalized before training.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/swethak1401/DeepFake_GAN_Images.git
cd DeepFake_GAN_Images
```

### 2. Set up the environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add the dataset
Download the face dataset and place it so the structure looks like:
```
data/
└── faces/
    ├── img_0001.jpg
    ├── img_0002.jpg
    └── ...
```

### 4. Run the notebook
```bash
jupyter notebook ml-da-gan-deepfake.ipynb
```
Run the cells top to bottom to load the data, build the networks, train the GAN, and generate face images.

---

## Requirements

Minimum dependencies (also see `requirements.txt`):
```
torch
torchvision
numpy
matplotlib
pillow
```
> A CUDA-capable GPU is strongly recommended for reasonable training times.

---

## Results

After training, the generator produces synthetic human faces from random latent vectors.

> **TODO:** Add sample output images here, e.g.:
> ```markdown
> ![Generated faces](assets/generated_samples.png)
> ```
> Place a few curated samples in an `assets/` folder (the `generated/` and `samples/` folders are git-ignored by default).

---

## Project Structure
```
DeepFake_GAN_Images/
├── ml-da-gan-deepfake.ipynb   # Main notebook: data, models, training, generation
├── README.md
├── .gitignore
└── requirements.txt           # (optional) pinned dependencies
```

---

## How It Works (Training Loop)

1. Sample a batch of real face images from the dataset.
2. Sample random latent vectors `z` and generate fake images with **G**.
3. Train **D** to correctly classify real vs. fake images.
4. Train **G** to produce images that **D** classifies as real.
5. Repeat across epochs, periodically saving/visualizing generated samples.

---

## Ethical Use

This project demonstrates generative modeling techniques. The generated faces are **synthetic and fictional**. Do not use this software to create deceptive media, impersonate real people, or generate harmful content.

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Author

**Swetha K** — [GitHub](https://github.com/swethak1401)
