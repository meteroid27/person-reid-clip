# Person-Based Video Segmentation and ReID using CLIP


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![CLIP](https://img.shields.io/badge/Model-IRRA%20CLIP-green)](https://arxiv.org/abs/2303.12501)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project implements a **multimodal person re-identification (ReID) system** capable of locating and tracking a specific individual across one or more surveillance videos using:

- A **query image** (image-based search)
- A **natural language description** (text-based search)
- **Both combined** (hybrid fusion for maximum robustness)

The system is powered by **CLIP** fine-tuned using the **IRRA** framework (CVPR 2023) on the CUHK-PEDES person description dataset, integrated with **YOLOv8** detection and **DeepSORT** tracking, and deployed as an interactive **Streamlit web application**.

---

## System Architecture

```
Input Video(s)  +  Query (Image / Text / Both)
        │
        ▼
┌─────────────────┐
│   YOLOv8        │  ←  Fine-tuned person detector
│   (Detection)   │
└────────┬────────┘
         │  Bounding boxes per frame
         ▼
┌─────────────────┐
│   DeepSORT      │  ←  Multi-object tracker
│   (Tracking)    │      assigns consistent track IDs
└────────┬────────┘
         │  Per-person crops (track segments)
         ▼
┌──────────────────────────────────────────┐
│                                          │
│                                          │
│  Query Image ──► Image Encoder ──► feat  │
│  Query Text  ──► Text Encoder  ──► feat  │
│                                          │
│  Hybrid:  feat = α·img + (1-α)·text      │
└───────────────────┬──────────────────────┘
                    │  Cosine similarity vs all tracks
                    ▼
┌─────────────────┐
│  Track Scoring  │  ←  Top-50% avg similarity
│  & Selection    │      + per-frame quality filter
└────────┬────────┘
         │
         ▼
┌────────────────────┐   ┌───────────────────────┐
│  Annotated Video   │   │  Cropped Target Video  │
│  (full frame+bbox) │   │  (target person only)  │
└────────────────────┘   └───────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **Image Query** | Search by uploading a photo of the target person |
| **Text Query** | Search using natural language (e.g. *"red jacket, black jeans"*) |
| **Hybrid Query** | Fuse image + text with tunable alpha weight for best accuracy |
| **Multi-Video** | Search across multiple videos simultaneously with one query |
| **Dual Output** | Annotated full-frame video + cropped target-only video |
| **In-App Playback** | Watch output directly in the browser — no download needed |
| **Occlusion Handling** | Track survives up to 7 seconds of occlusion without ID switch |

---

## Project Structure

```
person-reid-clip/
│
├── app.py                      # Streamlit web application
├── reid_main.py                # Core ReID pipeline
├── clip_inference.py           # CLIP model architecture & inference
│
├── clip_training/              # Model training scripts
│   ├── clip_train_image.py     # Image-mode fine-tuning (IRRA)
│   └── clip_train_text.py      # Text-mode fine-tuning (IRRA + SDM + MLM)
│
├── clip_models/                # [NOT included] Fine-tuned CLIP weights
│   ├── best_model_image.pth    #   → image encoder checkpoint
│   └── best_model_text.pth     #   → text encoder checkpoint
│
├── models_path/                # [NOT included] YOLO weights
│   └── yolo_finetuned_best.pt
│
├── requirements.txt
├── LICENSE
└── README.md
```

> **Model weights are not included in this repository due to file size.**  
> Download all model files from the [**GitHub Releases**](https://github.com/meteroid27/person-reid-clip/releases/tag/v1.0) page and place them as shown below:
>
> ```
> clip_models/
> ├── best_model_image.pth    ← download from Releases
> └── best_model_text.pth     ← download from Releases
>
> models_path/
> └── yolo_finetuned_best.pt  ← download from Releases
> ```

---

## Installation

### Requirements
- Python 3.10+
- CUDA GPU (recommended) or CPU

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/meteroid27/person-reid-clip.git
cd person-reid-clip

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights
#    clip_models/best_model_image.pth
#    clip_models/best_model_text.pth
#    models_path/yolo_finetuned_best.pt

# 4. Launch the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Usage

### Image Mode
1. Select **Image** as the CLIP query mode in the sidebar
2. Upload one or more input videos
3. Upload a clear photo of the target person
4. Click **▶ Run ReID Pipeline**

### Text Mode
1. Select **Text** as the CLIP query mode
2. Upload videos
3. Enter a description: *"A person wearing a blue shirt and dark jeans"*
4. Click **▶ Run ReID Pipeline**

### Hybrid Mode *(recommended — best results)*
1. Select **Both** as the CLIP query mode
2. Upload videos + query image + text description
3. Adjust the **alpha** slider (`1.0` = image only, `0.0` = text only, `0.5` = equal)
4. Click **▶ Run ReID Pipeline**

### Multi-Video Search
Upload multiple videos in the sidebar. The pipeline processes all of them in sequence with the same query, producing separate outputs for each video.

---

## Training

The CLIP models are fine-tuned using the **IRRA** framework on the **CUHK-PEDES** dataset.

```bash
# Fine-tune text model (SDM + MLM losses)
python clip_training/clip_train_text.py

# Fine-tune image model
python clip_training/clip_train_image.py
```

**Hyperparameters (following IRRA CVPR 2023):**

| Parameter | Value |
|---|---|
| Base model | `openai/clip-vit-base-patch16` |
| Dataset | CUHK-PEDES (40,206 images, 13,003 persons) |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 4e-4 |
| Epochs | 60 |
| Warmup | 5 epochs |
| Batch size | 64 |
| Losses | SDM + MLM (text) / InfoNCE (image) |

---

## Configuration

Key tuning parameters in `reid_main.py`:

| Constant | Default | Effect |
|---|---|---|
| `CLIP_SIM_THRESHOLD` | `0.20` | Minimum track score to be accepted |
| `CLIP_RELATIVE_SCORE_MARGIN` | `0.10` | Margin to include re-identified track fragments |
| `CLIP_IMAGE_ABS_FRAME_THRESHOLD` | `0.15` | Per-frame floor for image/hybrid mode |
| `CLIP_TEXT_ABS_FRAME_THRESHOLD` | `0.138` | Per-frame floor for text mode |
| `DETECTION_CONFIDENCE` | `0.30` | YOLO confidence (lower = detects smaller persons) |

---

## References

1. **IRRA** — Jiang, Z. & Ye, M. (2023). *Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval.* CVPR 2023.  
   [[Paper]](https://arxiv.org/abs/2303.12501) | [[Code]](https://github.com/anosorae/IRRA)

2. **CLIP** — Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.  
   [[Paper]](https://arxiv.org/abs/2103.00020) | [[Code]](https://github.com/openai/CLIP)

3. **ViT** — Dosovitskiy, A. et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.  
   [[Paper]](https://arxiv.org/abs/2010.11929)

4. **SDM Loss** — Jiang, Z. & Ye, M. (2023). *Similarity Distribution Matching.* (Introduced in IRRA, CVPR 2023.)  
   [[Paper]](https://arxiv.org/abs/2303.12501)

5. **YOLOv8** — Jocher, G. et al., Ultralytics (2023).  
   [[Docs]](https://docs.ultralytics.com) | [[Repo]](https://github.com/ultralytics/ultralytics)

6. **DeepSORT** — Wojke, N., Bewley, A. & Paulus, D. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric.* ICIP 2017.  
   [[Paper]](https://arxiv.org/abs/1703.07402) | [[Code]](https://github.com/nwojke/deep_sort)

7. **CUHK-PEDES Dataset** — Li, S. et al. (2017). *Person Search with Natural Language Description.* CVPR 2017.  
   [[Paper]](https://arxiv.org/abs/1702.05729) | [[Dataset]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)

8. **SORT** — Bewley, A. et al. (2016). *Simple Online and Realtime Tracking.* ICIP 2016.  
   [[Paper]](https://arxiv.org/abs/1602.00763)

---

## Author

**Pukar Timalsina**  
Final Year — B.E. Electronics and Computer Engineering  
Tribhuvan University, Institute of Engineering

---

## License

MIT License — see [LICENSE](LICENSE) for details.
