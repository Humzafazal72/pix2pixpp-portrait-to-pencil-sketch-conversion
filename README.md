# pix2pix++: Portrait to Pencil Sketch Conversion

<p align="center">
  <img src="assets/demo_result.png" alt="Portrait to Pencil Sketch Demo" width="600"/>
</p>

A PyTorch implementation of **pix2pix++**, a GAN-based framework for converting portrait images into realistic pencil sketches. This project enhances the original pix2pix architecture by integrating a U-Net++ generator to capture richer contextual information and produce fine-grained sketch details.

---

## 🖋️ Overview

This repository contains:

- An improved **pix2pix++** model architecture for image-to-image translation.
- A dataset of portrait/sketch pairs created using a mobile app and preprocessed for training.
- Training and evaluation scripts to replicate and extend our results.

This work was part of a research study aiming to generate pencil sketches that preserve the artistic quality and structure of input portraits.

---

## 📂 Dataset

- Total Pairs: **5,058**
  - Training: 4,299 pairs
  - Testing: 759 pairs
- Sketches generated using *Photo Sketch Maker* (mobile app) and manually enhanced (brightness & contrast).


---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```
git clone https://github.com/Humzafazal72/pix2pixpp-portrait-to-pencil-sketch-conversion.git
cd pix2pixpp-portrait-to-pencil-sketch-conversion
```

### 2️⃣ Install Dependencies
We recommend using a virtual environment:

```
conda create -n pix2pixpp python=3.9
conda activate pix2pixpp
pip install -r requirements.txt
```
