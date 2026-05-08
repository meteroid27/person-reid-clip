# Multimodal Person Re-Identification Pipeline using Fine-Tuned CLIP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains a comprehensive **Multimodal Person Re-Identification (ReID) system** that tracks and locates specific individuals across surveillance videos. Leveraging a foundational Vision-Language Model (CLIP) and advanced tracking algorithms, it supports three distinct search mechanisms:

1. **Image Query:** Provide an image of the target person.
2. **Text Query:** Provide a natural language description (e.g., "A man in a red shirt and blue jeans carrying a black backpack").
3. **Hybrid Query:** Combine both an image and a text description, dynamically weighted for maximum accuracy.

The pipeline incorporates **YOLOv8** for robust human detection and **DeepSORT** for temporal object tracking, wrapping the underlying ReID engine into an interactive **Streamlit web application**.

---

## Model Architecture and Theory

The core of this system relies on OpenAI's **CLIP** model, systematically fine-tuned to project both visual and textual features into a deeply aligned discriminative space. We utilize two specialized fine-tuning paradigms:

### 1. Text-Query Pipeline (IRRA Framework)
* **Methodology:** Uses the **Cross-Modal Implicit Relation Reasoning and Aligning (IRRA)** framework. It applies Masked Language Modeling (MLM) and Similarity Distribution Matching (SDM) loss. This allows the model to learn fine-grained attribute-level alignments between natural language descriptions and visual crops.
* **Training Dataset:** **CUHK-PEDES** (Person Search with Natural Language Description).

### 2. Image-Query Pipeline (CLIP-ReID Framework)
* **Methodology:** Rather than heavily fine-tuning and destroying CLIP's pre-trained visual backbone, this pipeline utilizes **Prompt Learning** based on the **CLIP-ReID** methodology. It freezes the core encoders and trains domain-specific learnable text prompts and visual adapters to map person images to a highly discriminative ReID embedding space.
* **Training Dataset:** **Market-1501**.

### System Pipeline

\\\	ext
[ Input Video ]     [ Query: Image / Text / Both ]
       |                          |
       v                          |
+---------------+                 |
|    YOLOv8     |                 |
|  (Detection)  |                 |
+-------+-------+                 |
        |                         |
        v                         |
+---------------+                 v
|   DeepSORT    |       +-------------------+
|  (Tracking)   |       |   CLIP Encoders   |
+-------+-------+       | (IRRA/CLIP-ReID)  |
        |               +---------+---------+
        v                         |
[ Target Tracklets ]              |
        |                         |
        +----------+--------------+
                   |
                   v
         +-------------------+
         | Cosine Similarity |
         |   Track Scoring   |
         +---------+---------+
                   |
                   v
          [ Final Output Video ]
\\\

---

## Repository Structure

\\\	ext
person-reid-clip/
|
+-- app.py                      # Main Streamlit web application interface
+-- reid_main.py                # Core ReID logic, Video processing, YOLO/DeepSORT integration
+-- clip_inference.py           # Feature extraction using the fine-tuned CLIP models
|
+-- clip_training/              # Training scripts for fine-tuning
|   +-- clip_train_image.py     # Image-mode fine-tuning (Prompt Learning via CLIP-ReID)
|   +-- clip_train_text.py      # Text-mode fine-tuning (IRRA + SDM + MLM loss formulation)
|
+-- clip_models/                # Directory for fine-tuned CLIP weights
|   +-- .gitkeep                # (Weights downloaded separately)
|
+-- models_path/                # Directory for YOLO detection weights
|   +-- .gitkeep                # (Weights downloaded separately)
|
+-- requirements.txt            # Python dependencies
+-- README.md                   # Project documentation
+-- LICENSE                     # MIT License
\\\

---

## Setup and Installation

**1. Clone the Repository:**
\\\ash
git clone https://github.com/meteroid27/person-reid-clip.git
cd person-reid-clip
\\\

**2. Install Dependencies:**
Ensure you have Python 3.10+ installed.
\\\ash
pip install -r requirements.txt
\\\
*Note: Depending on your hardware, you may want to install the GPU-compiled version of PyTorch directly from the [PyTorch website](https://pytorch.org/).*

**3. Download Model Weights:**
Because of their size, the fine-tuned \.pth\ and \.pt\ model weights are hosted on GitHub Releases.
* Download the weights from the **[GitHub Releases Page](https://github.com/meteroid27/person-reid-clip/releases)**.
* Place the text and image encoder weights inside the \clip_models/\ folder.
* Place the YOLO fine-tuned weights inside the \models_path/\ folder.

Your folder structure should now look like this:
\\\	ext
clip_models/
   +-- best_model_image.pth
   +-- best_model_text.pth
models_path/
   +-- yolo_finetuned_best.pt
\\\

---

## Usage

To launch the interactive web application, run:
\\\ash
streamlit run app.py
\\\
This will open the interface in your default web browser where you can:
1. Upload your input surveillance video(s).
2. Input a query text, upload a query image, or both.
3. Adjust the **Hybrid Weight (Alpha)** slider if using both (determines the balance between image vs text feature importance).
4. Run the pipeline to process, detect, and track the matching individual.

---

## References

**Foundational Models & Trackers:**
* **CLIP:** Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* [[Repository]](https://github.com/openai/CLIP)
* **YOLOv8:** Jocher, G., et al. (2023). Ultralytics. [[Repository]](https://github.com/ultralytics/ultralytics)
* **DeepSORT:** Wojke, N., et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric.* ICIP.

**ReID Frameworks:**
* **IRRA (Text-Query Framework):** Jiang, Z., & Ye, M. (2023). *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval.* CVPR. [[Paper]](https://arxiv.org/abs/2303.12501)
* **CLIP-ReID (Image-Query Framework):** Li, S., et al. (2023). *CLIP-Driven Fine-grained Text-Image Person Re-identification.* IEEE TIP. [[Paper]](https://arxiv.org/abs/2211.13977)

**Datasets:**
* **CUHK-PEDES** (Text-Query Dataset) - Li, S., et al. CVPR 2017.
* **Market-1501** (Image-Query Dataset) - Zheng, L., et al. ICCV 2015.

---

## Author

**Prasish Timalsina**  
Final Year — B.E. Electronics and Computer Engineering  
Tribhuvan University, Institute of Engineering