# Multimodal Person Re-Identification Pipeline using Fine-Tuned CLIP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains a comprehensive **Multimodal Person Re-Identification (ReID) system** capable of locating and tracking specific individuals across surveillance videos. Built upon a foundational Vision-Language Model (CLIP) and robust tracking algorithms, it implements three advanced search mechanisms:

1. **Image Query:** Locate a person using a source visual image.
2. **Text Query:** Locate a person using a natural language description (e.g., "A man wearing a red shirt and blue jeans").
3. **Hybrid Query:** Combines both an image and a text description, applying dynamic weighting to maximize accuracy.

The system incorporates **YOLOv8** for human detection and **DeepSORT** for temporal object tracking. The core ReID inference is presented through an interactive **Streamlit web application**.

---

## 1. System Pipeline Details

The end-to-end inference and tracking pipeline executes the following sequenced architecture:

1. **Video Input & Detection:** Surveillance video frames are processed frame-by-frame. A custom-trained **YOLOv8** model detects humans, returning bounding box coordinates for each subject.
2. **Object Tracking:** The bounding boxes are fed into **DeepSORT**, which assigns and tracks unique IDs across consecutive frames, converting individual detections into continuous "tracklets."
3. **Feature Extraction:** For each tracked individual, the detected image crop is passed through the fine-tuned dual-stream **CLIP Encoders**. Simultaneously, the user's text or image query is passed through the corresponding encoder to generate feature embeddings in a unified visual-linguistic space.
4. **Scoring & Matching:** The system calculates the **Cosine Similarity** between the user's query embedding and the tracklet embeddings. 
5. **Output Generation:** High-confidence matches are annotated over the final output video, dynamically highlighting the sequence where the targeted individual is present.

---

## 2. Model Training and Testing Methodologies

The system relies on OpenAI's **CLIP (ViT-B/16)** model, which has been specifically fine-tuned for the Person Re-Identification task. We developed two distinct specialized pipelines for text-based and image-based queries:

### A. Image-Only Pipeline (CLIP-ReID Framework)

**Methodology & Architecture:**
This pipeline operates using **Prompt Learning** inspired by the **CLIP-ReID** methodology. Rather than fully updating and potentially degrading the pre-trained visual backbone of CLIP, the visual encoder remains frozen or operates at a significantly low learning rate. Instead, it trains domain-specific learnable text prompts (16 learnable context tokens) alongside a bottleneck layer (`BatchNorm1d(512)`) and a classification head. This strategy maps person images to a highly discriminative embedding space.

**Training Process:**
* **Dataset:** Evaluated and trained on **Market-1501**.
* **Hyperparameters:** Trained for 30 epochs using an Adam optimizer with `CosineAnnealingLR`.
* **Loss Functions:** Trained utilizing a multi-loss strategy combining **ID Loss (CrossEntropy)**, **Triplet Loss**, and **Image-Text Contrastive (ITC) Loss**. 
* **Sampling:** Uses a `RandomIdentitySampler` to draw 4 instances per identity per batch ensuring rich triplet mining.
* **Testing / Results:** Achieved **94.48% Rank-1** accuracy and **87.16% mAP**.

### B. Text-Only Pipeline (IRRA Framework)

**Methodology & Architecture:**
The text-to-image module incorporates the **Cross-Modal Implicit Relation Reasoning and Aligning (IRRA)** framework. Unlike the image pipeline, it fine-tunes *both* the image and text transformers fully end-to-end to maximize cross-modal alignment.

**Training Process:**
* **Dataset:** Evaluated and trained on **CUHK-PEDES**, split strictly by person identity (preventing train-test data leakage).
* **Hyperparameters:** Trained for 30 epochs (original IRRA parameter: 60) with an effective batch size of 64 (batch size 32 with 2 gradient accumulation steps). Uses a backbone learning rate of `1e-5`, a new-module learning rate of `1e-4`, weight decay of `4e-4`, and a 5-epoch warmup.
* **Loss Components:**
  1. **Similarity Distribution Matching (SDM) Loss:** Replaces standard InfoNCE. It constructs soft identity labels using person IDs and minimizes the KL-divergence between the predicted and actual label similarity distributions (`sigma=0.01`). Correctly handles multiple positives per identity.
  2. **Masked Language Modeling (MLM):** Masks random text tokens (`15%`) and applies a single cross-modal attention module (attending text queries to image patches) to predict the masked vocabulary. This mandates fine-grained, localized alignment between image regions and text descriptions.
  3. **ID Loss:** Combined with SDM and MLM equally.
* **Testing / Results:** Achieved **65.22% Rank-1** and **62.93% mAP** on the identity-level CUHK-PEDES split, compared to a CLIP (ViT-B/16) zero-shot baseline of **23.55% Rank-1** and **21.26% mAP**.

---

## Repository Structure

```text
person-reid-clip/
├── app.py                      # Main Streamlit web application interface
├── reid_main.py                # Core ReID logic, Video processing, YOLO/DeepSORT integration
├── clip_inference.py           # Feature extraction using the fine-tuned CLIP models
├── clip_training/              # Training scripts for fine-tuning
│   ├── clip_train_image.py     # Image-mode fine-tuning (Prompt Learning via CLIP-ReID)
│   └── clip_train_text.py      # Text-mode fine-tuning (IRRA + SDM + MLM loss formulation)
├── clip_models/                # Directory for fine-tuned CLIP weights
│   ├── best_model_image.pth    
│   └── best_model_text.pth     
├── models_path/                # Directory for tracking weights
│   └── yolo_finetuned_best.pt  
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

---

## Setup and Installation

**1. Clone the Repository:**
```bash
git clone https://github.com/meteroid27/person-reid-clip.git
cd person-reid-clip
```

**2. Install Dependencies:**
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```
*Note: For maximum performance, install the GPU-compiled PyTorch directly from [pytorch.org](https://pytorch.org/).*

**3. Download Model Weights:**
Ensure you download the fine-tuned model `.pth` and `.pt` weights into their respective directories:
* Place the text and image encoder weights (`best_model_image.pth`, `best_model_text.pth`) inside the `clip_models/` folder.
* Place the YOLO fine-tuned weights (`yolo_finetuned_best.pt`) inside the `models_path/` folder.

---

## Usage

Launch the interactive Streamlit web application via the terminal:
```bash
streamlit run app.py
```
From the interactive browser window, you can:
1. Upload your input surveillance video(s).
2. Input a text query, upload a target image, or do both for Hybrid mode.
3. Configure the **Hybrid Weight (Alpha)** slider to balance visual vs. text representation importance dynamically.
4. Run the pipeline to output the traced individual.

---

## References

* **CLIP:** Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* [Repository](https://github.com/openai/CLIP)
* **YOLOv8:** Jocher, G., et al. (2023). Ultralytics. [Repository](https://github.com/ultralytics/ultralytics)
* **DeepSORT:** Wojke, N., et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric.* ICIP.
* **CLIP-ReID (Image-Query Framework):** Li, S., et al. (2023). *CLIP-Driven Fine-grained Text-Image Person Re-identification.* IEEE TIP. [Paper](https://arxiv.org/abs/2211.13977)
* **IRRA (Text-Query Framework):** Jiang, Z., & Ye, M. (2023). *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval.* CVPR. [Paper](https://arxiv.org/abs/2303.12501)
