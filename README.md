# 🎯 Yifer Face Classification — Colab GPU Pipeline

> **Automatically sort group photos into two classes based on whether a specific person (Yifer) appears in them — using ArcFace embeddings, YOLOv8n face detection, and GPU-accelerated inference on Google Colab.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![DeepFace](https://img.shields.io/badge/DeepFace-ArcFace-orange)
![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8n-green?logo=ultralytics)
![GPU](https://img.shields.io/badge/GPU-T4%20%2F%20A100-76B900?logo=nvidia)

---

## 📖 Overview

This notebook implements a **face-identity-based photo classifier** that:

1. Takes a folder of reference photos of **one specific person** (Yifer)
2. Extracts their **ArcFace facial embeddings** and builds a **L2-normalized mean centroid**
3. Scans every photo in your **group photos** folder
4. Classifies each photo into:
   - **`class_1/`** — Yifer is present in the photo
   - **`class_2/`** — Yifer is NOT present
5. Saves all results directly to **Google Drive** with a visual report

The pipeline is fully GPU-accelerated (NVIDIA T4 / A100 on Colab) and processes images from Google Drive through a local SSD cache for maximum speed.

---

## 🏗️ Architecture

```
Reference Photos (yife/)
        │
        ▼
  YOLOv8n Detection
  + Multi-Detector Fallback
        │
        ▼
  ArcFace Embeddings
  + L2-Normalization
        │
        ▼
  Mean Centroid + Individual Refs
        │
        ▼
Group Photos (group_photos/)
        │
        ▼
  Cosine Distance Comparison
  (threshold = 0.65)
        │
        ├── ≤ threshold ──▶ class_1/  (Yifer present)
        └──  > threshold ──▶ class_2/  (Yifer absent)
```

---

## ✨ Key Features

| Feature | Detail |
|---------|--------|
| 🎯 **Face Recognition Model** | ArcFace (512-D embeddings, state-of-the-art accuracy) |
| 👁️ **Primary Detector** | YOLOv8n (fast & accurate) |
| 🔄 **Multi-Detector Fallback** | Auto-tries `yolov8n → retinaface → mtcnn → opencv → ssd` |
| 📐 **Matching Strategy** | L2-normalized cosine distance vs. both mean centroid and all individual embeddings |
| ⚡ **Performance** | Drive → SSD cache = 10-20x faster repeated runs |
| 🔍 **Debug Step** | Inspect actual cosine distances before classifying all photos |
| 📊 **Visual Report** | Pie chart + bar chart + distance histogram, saved to Drive |
| 🔧 **Tunable Threshold** | Optional re-tune cell to re-run with a different threshold without re-extracting embeddings |
| 📁 **Auto Folder Creation** | `class_1/` and `class_2/` created on Drive automatically if missing |

---

## 📋 Notebook Steps

| Step | Cell ID | Description |
|------|---------|-------------|
| **1** | `s1` | ✅ Verify GPU (fails fast if no GPU detected) |
| **2** | `s2a` / `s2b` | 📦 Install libraries + import all dependencies |
| **3** | `s3` | 📂 Mount Google Drive |
| **4** | `s4` | ⚙️ Configuration — set all paths, model, and threshold |
| **5** | `s5` | 🔧 Define helper functions (`preprocess`, `l2_normalize`, `show_faces`) |
| **6** | `s6` | 🔧 Preprocess & cache photos to local `/content/` SSD |
| **7** | `s7` | 🔍 **Detector Diagnosis** — test all detectors, auto-select best |
| **8** | `s8` / `grid` | 🧠 Extract Yifer embeddings + display reference photo grid |
| **9** | `s9` | 🔍 **Debug** — inspect actual cosine distances on sample photos |
| **10** | `s10fn` / `s10run` | 🔍 **Classify** all group photos |
| **11** | `s11` | 📁 Copy results to Google Drive (`class_1/` & `class_2/`) |
| **12** | `s12grids` / `s12charts` / `s12summary` | 📊 Visual grids + charts + final summary table |
| *(opt)* | `retune` / `apply` | 🔄 Re-run classification with a different threshold |

---

## 🚀 Quick Start

### 1. Prerequisites

- A **Google account** with Google Drive
- Google Colab with **GPU runtime enabled**
  ```
  Runtime → Change runtime type → Hardware accelerator → GPU (T4)
  ```

### 2. Prepare Your Google Drive

Upload the following folders to your Google Drive **before** opening the notebook:

```
MyDrive/
└── practice/
    ├── yife/              ← Reference photos of Yifer (10-50 photos recommended)
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── ...
    └── group_photos/      ← Group photos to classify
        ├── group1.jpg
        ├── group2.jpg
        └── ...
```

> **Tips for reference photos:**
> - Use clear, front-facing photos where Yifer's face is visible
> - Variety helps (different angles, lighting, expressions)
> - Minimum ~5 photos; 15-40 is ideal

### 3. Open the Notebook

Upload `face_classification_colab.ipynb` to Colab, or open it directly from your Drive.

### 4. Run All Cells

```
Runtime → Run all  (Ctrl+F9)
```

Then follow the step-by-step progress in the output.

---

## ⚙️ Configuration Reference

All user-facing settings live in **Step 4**:

```python
# ===== EDIT THESE IF NEEDED =====
BASE_DIR    = '/content/drive/MyDrive/practice'
YIFER_DIR   = os.path.join(BASE_DIR, 'yife')          # Reference photos
GROUP_DIR   = os.path.join(BASE_DIR, 'group_photos')  # Photos to classify
CLASS1_DIR  = os.path.join(BASE_DIR, 'class_1')       # Output: Yifer found
CLASS2_DIR  = os.path.join(BASE_DIR, 'class_2')       # Output: Yifer not found

MODEL_NAME  = 'ArcFace'   # Face recognition model
THRESHOLD   = 0.65        # Cosine distance threshold (0.0–1.0)

DETECTOR_FALLBACK_ORDER = ['yolov8n', 'retinaface', 'mtcnn', 'opencv', 'ssd']
```

### Threshold Guide

| Threshold | Effect |
|-----------|--------|
| `0.55` | 🔒 Stricter — fewer false positives, may miss some Yifer photos |
| `0.65` | ✅ **Default** — balanced for most use cases |
| `0.70` | 🔓 More lenient — catches more Yifer, may include some false positives |
| `0.75` | ⚠️ Very lenient — high recall, lower precision |

> Run **Step 9 (Debug)** first to see what distances your photos actually produce before committing to a threshold.

---

## 📦 Dependencies

Installed automatically in Step 2:

| Package | Purpose |
|---------|---------|
| `deepface` | ArcFace embeddings + face extraction |
| `tf-keras` | TensorFlow/Keras backend for DeepFace |
| `retina-face` | RetinaFace detector support |
| `mtcnn` | MTCNN detector support |
| `ultralytics` | **YOLOv8n** detector support |
| `tqdm` | Progress bars |
| `scikit-learn` | Utility functions |
| `Pillow` | Image loading and preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization and charts |
| `scipy` | Cosine distance computation |

---

## 🔬 How It Works

### Embedding Extraction
Each reference photo goes through:
1. **EXIF-aware preprocessing** — correct rotation, resize to ≤ 1024px
2. **YOLOv8n face detection** — with fallback to RetinaFace, MTCNN, etc.
3. **ArcFace representation** — 512-dimensional face embedding vector
4. **L2-normalization** — unit-length vectors for stable cosine similarity

### Identity Matching
For each group photo face, cosine distance is computed against:
- The **mean centroid** of all Yifer embeddings (robust to outlier reference photos)
- **Every individual** reference embedding (catches edge cases the centroid misses)

The **minimum** of both distances is used as the final score.

```
distance = min(
    cosine(face_emb, yifer_mean_centroid),
    min(cosine(face_emb, ref) for ref in all_yifer_refs)
)

if distance ≤ THRESHOLD:  →  Class 1 (Yifer present)
else:                      →  Class 2 (Yifer absent)
```

---

## 🛠️ Known Issues & Fixes Applied

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `Invalid model_name passed - face_detector/yolov8` | Wrong detector name | Changed `yolov8` → **`yolov8n`** |
| `KerasTensor cannot be used as input` (RetinaFace) | Keras v3 incompatibility | Set `TF_USE_LEGACY_KERAS=1` before DeepFace import |
| `module 'mediapipe' has no attribute 'solutions'` | Broken in newer Colab | Removed mediapipe from detector list |
| `NameError: name 'time' is not defined` | Missing import | `import time` added to Step 2 |
| Bounding boxes drawn in wrong position | Scale mismatch between display size and native pixels | BB coords now drawn on the **native pixel array** |
| All photos classified as Class 2 | Threshold too low + no L2 normalization | Raised threshold to 0.65 + added L2-normalize step |
| `class_1/` or `class_2/` not found on Drive | Folders not created | `os.makedirs(..., exist_ok=True)` added in Step 11 |

---

## 📊 Output Files

After a successful run, your Google Drive `practice/` folder will contain:

```
MyDrive/practice/
├── class_1/                        ← Photos containing Yifer
│   ├── group_photo_001.jpg
│   └── ...
├── class_2/                        ← Photos NOT containing Yifer
│   ├── group_photo_002.jpg
│   └── ...
└── classification_report.png       ← Visual summary chart
```

---

## 🔄 Re-running & Resuming

- **Re-run classification only** (without re-extracting embeddings): Just re-run Steps 10-12
- **Change threshold without reclassifying everything**: Use the **Optional Re-tune** cells at the bottom
- **Add more reference photos**: Re-run Steps 6 and 8 only (the new photos will be included)
- **SSD cache**: The preprocessed images in `/content/preprocessed/` persist until the Colab session ends. Re-running Step 6 reuses the cache automatically.

---

## 📐 Technical Notes

- **ArcFace cosine default threshold:** `0.68` (per DeepFace docs). This notebook uses `0.65` for slightly stricter matching.
- **L2 normalization:** Ensures cosine distance returns values in `[0, 1]` range, making the threshold intuitive.
- **Detector auto-selection (Step 7):** Tests each detector on 3 sample photos and picks the one with the highest detection count. The selected detector is "locked in" for Steps 8–10 to guarantee consistency.
- **Multi-detector fallback:** If the locked detector fails on a specific photo (e.g., unusual angle), the pipeline automatically tries the next detectors before giving up.

---

## 🤝 Contributing / Adapting

To adapt this notebook for a **different person**:
1. Replace the contents of `yife/` with reference photos of the new person
2. Optionally rename the directory and update `YIFER_DIR` in Step 4
3. Re-run from Step 6 onwards

To classify for **multiple people**, run the full pipeline once per person with different output folders.

---

## 📄 License

This project is for personal/educational use. The underlying models (ArcFace, YOLOv8) are subject to their respective open-source licenses.
