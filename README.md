# audio-classification-task1
# Audio Classification Project

A comprehensive audio classification pipeline built using **PyTorch**, integrating modern deep‑learning techniques such as **transfer learning**, **mel‑spectrogram feature extraction**, **data augmentation**, **SpecAugment**, **early stopping**, and **ResNet‑18** fine‑tuning.

This project is based on the **FreeCodeCamp PyTorch Tutorial** by `omaratef3221`, with several enhancements inspired by the following resources:

* [https://github.com/omaratef3221/pytorch_tutorials](https://github.com/omaratef3221/pytorch_tutorials)
* Transfer learning with ResNet‑18: [https://ngaif.com/guide-to-transfer-learning-with-pytorch/](https://ngaif.com/guide-to-transfer-learning-with-pytorch/)
* SpecAugment and audio augmentation: [https://docs.pytorch.org/audio/main/tutorials/audio_feature_augmentation_tutorial.html](https://docs.pytorch.org/audio/main/tutorials/audio_feature_augmentation_tutorial.html)
* Early stopping reference: [https://pythonguides.com/pytorch-early-stopping/](https://pythonguides.com/pytorch-early-stopping/)

This README explains the project structure, dataset pipeline, model design, training loop, techniques used, and reasons for choosing them.

##  Project Overview

This project performs **multi‑class audio classification** using deep learning. It converts raw audio into **log‑mel‑spectrograms**, applies augmentations, and trains a **ResNet‑18** model (pretrained on ImageNet) to classify environmental sounds.

The workflow includes:

1. Dataset loading & preprocessing
2. Mel‑spectrogram extraction
3. Log1p compression 
4. SpecAugment techniques
5. Model architecture (ResNet‑18 transfer learning)
6. Training/validation loops
7. Early stopping & best‑model saving
8. Evaluation & inference

---

## 📁 Folder Structure (Recommended)

```
project/
│── data/                    # Raw audio files (.wav)
│── models/                  # Saved model weights
│── utils/                   # Helper functions
│── audio_classificationTASK1.ipynb
│── README.md                # This file
```

---

## Features & Techniques Used

### **1. Audio Preprocessing (Mel‑Spectrograms)**

* Converts audio to **80‑dim mel‑spectrograms** using librosa
* Parameters used:

  * n_fft = 1024
  * hop_length = 256
  * win_length = 1024

**Reason:** Mel‑spectrograms resemble how humans perceive sound and are the standard input for audio deep‑learning models.

---

### **2. Log1p Compression**

```
log_mel = torch.log1p(mel)
```

**Reason:** Reduces dynamic range of spectrograms and stabilizes training.

---

### **3. SpecAugment (Time & Frequency Masking)**

Added using PyTorch's official API:

* `TimeMasking`
* `FrequencyMasking`

**Reason:** Improves model robustness against noise & overfitting by simulating real‑world variations.

---

### **4. Transfer Learning with ResNet‑18**

* Loaded pretrained ResNet‑18
* Modified first Conv2D layer to accept 1‑channel inputs
* Replaced final FC layer with `nn.Linear(num_features, num_classes)`

**Reason:** ResNet‑18 learns high‑level features efficiently and performs extremely well on spectrogram images.

---

### **5. Early Stopping**

Monitors **validation loss**, stops when no improvement after `patience` epochs.

**Reason:** Prevents overfitting and reduces unnecessary training time.

---

### **6. Custom Dataset Class**

Handles:

* Loading audio
* Computing mel‑spectrograms
* Padding/cropping to fixed frame length
* Applying SpecAugment

**Reason:** Provides complete control over preprocessing, making the pipeline reusable.

---

### **7. Training Loop**

Includes:

* Learning rate scheduling
* Loss monitoring
* Model saving

**Reason:** Smooth and stable optimization.

--------

##  Training

* Optimizer: AdamW
* Loss: CrossEntropyLoss
* Batch size : 16
* Epochs: 25

---

## 📝 References

1. FreeCodeCamp PyTorch tutorial (base code): [https://github.com/omaratef3221/pytorch_tutorials](https://github.com/omaratef3221/pytorch_tutorials)
2. Transfer Learning with ResNet‑18: [https://ngaif.com/guide-to-transfer-learning-with-pytorch/](https://ngaif.com/guide-to-transfer-learning-with-pytorch/)
3. SpecAugment documentation:

   * [https://docs.pytorch.org/audio/main/tutorials/audio_feature_augmentation_tutorial.html](https://docs.pytorch.org/audio/main/tutorials/audio_feature_augmentation_tutorial.html)
4. Early Stopping guide:

   * [https://pythonguides.com/pytorch-early-stopping/](https://pythonguides.com/pytorch-early-stopping/)
