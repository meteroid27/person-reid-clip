# Person-Based Video Segmentation and ReID using CLIP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project implements a **multimodal person re-identification (ReID) system** capable of locating and tracking a specific individual across one or more surveillance videos using:

- **Query Image:** Image-based search using CLIP-ReID (Prompt Learning).
- **Query Text:** Natural language description search using the IRRA framework.
- **Hybrid Query:** Both combined (hybrid fusion) for maximum robustness.

The system is powered by **CLIP**. The text-query pipeline is fine-tuned using the **IRRA** framework on the **CUHK-PEDES** dataset. The image-query pipeline utilizes **prompt learning** based on **CLIP-ReID** and is trained on the **Market-1501** dataset. These are integrated with **YOLOv8** detection and **DeepSORT** tracking, and deployed as an interactive **Streamlit web application**.

---

## Model Training & Testing Details

Our system utilizes two specialized pipelines for text and image queries, both fine-tuned from OpenAI's foundational CLIP model.

### 1. Text-Only Model (IRRA)
- **Framework:** IRRA (Cross-Modal Implicit Relation Reasoning and Aligning).
- **Dataset:** Trained and tested on **CUHK-PEDES** (Person Search with Natural Language Description).
- **Methodology:** The text pipeline fine-tunes the CLIP text encoder. It uses Implicit Relation Reasoning and Similarity Distribution Matching (SDM) loss along with Masked Language Modeling (MLM) to deeply align textual attribute descriptions with visual embeddings. 

### 2. Image-Only Model (CLIP-ReID)
- **Framework:** CLIP-ReID (Prompt Learning).
- **Dataset:** Trained and tested on the **Market-1501** dataset.
- **Methodology:** Instead of fully fine-tuning the entire visual backbone (which can destroy CLIP's rich pre-trained representations), this pipeline freezes the core visual encoder and utilizes **Prompt Learning**. Learnable text prompts and domain-specific adapters are trained to project Market-1501 visual features into a highly discriminative ReID embedding space.

---

## System Architecture

\\\	ext
Input Video(s)  +  Query (Image / Text / Both)
        |
        V
+-----------------+
|   YOLOv8        |  <-  Fine-tuned person detector
|   (Detection)   |
+--------+--------+
         |  Bounding boxes per frame
         V
+-----------------+
|   DeepSORT      |  <-  Multi-object tracker
|   (Tracking)    |      assigns consistent track IDs
+--------+--------+
         |  Per-person crops (track segments)
         V
+------------------------------------------+
|                                          |
|  Query Image --> Image Encoder --> feat  |
|  Query Text  --> Text Encoder  --> feat  |
|                                          |
|  Hybrid:  feat = a.img + (1-a).text      |
+------------------+-----------------------+
                   |  Cosine similarity vs all tracks
                   V
          +-----------------+
          |  Track Scoring  |
          +--------+--------+
         +---------+---------+
         V                   V
+-----------------+ +-----------------+
| Annotated Video | |  Cropped Target |
| (full + bbox)   | |  (person only)  |
+-----------------+ +-----------------+
\\\

---

## Features

| Feature | Description |
|---|---|
| **Image Query** | Search by uploading a photo of the target person |
| **Text Query** | Search using natural language (e.g. *"red jacket, black jeans"*) |
| **Hybrid Query** | Fuse image + text with tunable alpha weight for best accuracy |
| **Multi-Video** | Search across multiple videos simultaneously with one query |
| **Dual Output** | Annotated full-frame video + cropped target-only video |
| **In-App Playback** | Watch output directly in the browser |
| **Occlusion Handling** | Track survives up to 7 seconds of occlusion without ID switch |

---

## Project Structure

\\\	ext
person-reid-clip/
|
+-- app.py                      # Streamlit web application
+-- reid_main.py                # Core ReID pipeline
+-- clip_inference.py           # CLIP model architecture & inference
|
+-- clip_training/              # Model training scripts
|   +-- clip_train_image.py     # Image-mode fine-tuning (CLIP-ReID prompt learning)
|   +-- clip_train_text.py      # Text-mode fine-tuning (IRRA + SDM + MLM)
|
+-- clip_models/                # [Download from Releases]
|   +-- best_model_image.pth    #   -> image encoder checkpoint
|   +-- best_model_text.pth     #   -> text encoder checkpoint
|
+-- models_path/                # [Download from Releases]
|   +-- yolo_finetuned_best.pt
|
+-- requirements.txt
+-- LICENSE
+-- README.md
\\\

> **Model weights are not included in this repository due to file size.**  
> Download all model files from the [**GitHub Releases**](https://github.com/meteroid27/person-reid-clip/releases/tag/v1.0) page and place them in \clip_models/\ and \models_path/\ as shown above.

---

## References

1. **CLIP** — Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
   [[OpenAI CLIP Repository]](https://github.com/openai/CLIP)

2. **IRRA (Text Query Pipeline)** — Jiang, Z., & Ye, M. (2023). *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval.* CVPR 2023.
   [[Paper]](https://arxiv.org/abs/2303.12501) | [[Code]](https://github.com/anosorae/IRRA)

3. **CLIP-ReID (Image Query Pipeline)** — Li, S., et al. (2023). *CLIP-Driven Fine-grained Text-Image Person Re-identification.* IEEE Transactions on Image Processing.
   [[Paper]](https://arxiv.org/abs/2211.13977) | [[Code]](https://github.com/Syliz517/CLIP-ReID)

4. **CUHK-PEDES Dataset** — Li, S., et al. (2017). *Person Search with Natural Language Description.* CVPR 2017.
   [[Dataset Repository]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)

5. **Market-1501 Dataset** — Zheng, L., et al. (2015). *Scalable Person Re-identification: A Benchmark.* ICCV 2015.
   [[Dataset Information]](https://zheng-zhe.com/market1501.html)

---

## License

MIT License — see [LICENSE](LICENSE) for details.