# MVTec AD Anomaly Detection Demo – ViT-Core / Swin Transformer + FAISS

## 1. Overview

This project implements an industrial surface defect detection system on the **MVTec AD** dataset, inspired by the paper **ViT-Core: Lightweight Anomaly Detection Model using Transformer-based Feature Extractor**. The workflow is divided into two main stages:

1. **Training and evaluation on Kaggle**: processing the MVTec AD dataset, fine-tuning the model for each product category, building a FAISS Memory Bank, and saving the required artifacts.
2. **Local deployment on a personal computer**: downloading the trained model, FAISS index, and metrics from Hugging Face, running a Gradio interface, testing input images, visualizing heatmaps, and classifying products as **GOOD** or **DEFECTIVE** using category-specific thresholds.

The project focuses on **industrial anomaly detection**, where the model learns the feature distribution of normal images and detects regions that deviate from normal patterns.

---

## 2. References and Dataset

### 2.1. Reference Paper

- **Title:** ViT-Core: Lightweight Anomaly Detection Model using Transformer-based Feature Extractor
- **Publication:** IEEE Access, 2025
- **DOI:** https://doi.org/10.1109/ACCESS.2025.3618462
- **Reference page:** https://www.researchgate.net/publication/396267323_ViT-Core_Lightweight_Anomaly_Detection_Model_using_Transformer-based_Feature_Extractor

The main idea referenced from the paper is the use of a **Transformer-based feature extractor**, particularly a Swin Transformer-style backbone, for extracting visual representations in anomaly detection tasks. Instead of relying only on traditional CNN-based features, the project explores Transformer-based feature extraction combined with a memory-bank-based anomaly scoring approach.

### 2.2. Dataset

- **MVTec AD official dataset:** https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- **Kaggle mirror used for experimentation:** https://www.kaggle.com/datasets/ipythonx/mvtec-ad
- **Hugging Face:** https://huggingface.co/Manh2005/base-version/tree/main

The MVTec AD dataset contains multiple industrial object and texture categories. Each category follows a structure similar to:

```text
<category>/
├── train/
│   └── good/
├── test/
│   ├── good/
│   └── <defect_type>/
└── ground_truth/
    └── <defect_type>/
```

In this project, `train/good` images are used to build the normal feature representation, while `test/good` and `test/<defect_type>` images are used for evaluating image-level classification and visualizing anomalous regions.

---

## 3. Implemented Categories

The current local demo focuses on the following MVTec AD categories:

```text
carpet
grid
leather
tile
wood
bottle
cable
capsule
hazelnut
toothbrush
zipper
```

Each category requires three artifact files:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

For example, for the `toothbrush` category:

```text
mvtec_anomaly_detection/
└── toothbrush/
    ├── vit_core_swin_toothbrush.pth
    ├── memory_bank_toothbrush.index
    └── metrics_toothbrush.json
```

---

## 4. Work Completed on Kaggle

### 4.1. Dataset Preparation

The MVTec AD dataset was used in the Kaggle environment through a path similar to:

```text
/kaggle/input/datasets/ipythonx/mvtec-ad
```

The training notebook reads data category by category, for example:

```text
/kaggle/input/datasets/ipythonx/mvtec-ad/toothbrush/train/good
/kaggle/input/datasets/ipythonx/mvtec-ad/toothbrush/test/good
/kaggle/input/datasets/ipythonx/mvtec-ad/toothbrush/test/defective
```

### 4.2. Fine-tuning with Cut-Paste Augmentation

The model was fine-tuned using a self-supervised strategy based on **Cut-Paste augmentation**. Synthetic anomalous images were created by cutting a region from a normal image and pasting it into another location. This allows the model to learn differences between normal and abnormal patterns without requiring real defect labels during training.

The overall pipeline for each category is:

```text
train/good images
→ generate synthetic defects using Cut-Paste
→ fine-tune the Swin Transformer backbone
→ extract visual features
→ build a FAISS Memory Bank
→ evaluate on test/good and test/defect images
```

### 4.3. Feature Extractor

The model uses the following backbone:

```python
swin_base_patch4_window7_224
```

In the current local version, the feature extractor uses the following block:

```python
self.backbone.layers[2].blocks[3]
```

The extracted feature map is converted into the format:

```text
B, C, H, W
```

Then the feature map is reshaped into patch-level feature vectors and compared against the FAISS Memory Bank.

### 4.4. FAISS Memory Bank Construction

After fine-tuning, features are extracted from the `train/good` images. These normal features are stored in a FAISS index, which acts as the Memory Bank for the corresponding category.

During inference, a new image is passed through the feature extractor. Its patch-level features are compared with the normal feature vectors in the FAISS Memory Bank. A higher distance indicates a higher anomaly score.

### 4.5. Model Evaluation

On Kaggle, the model was evaluated using the following metrics:

```text
Image AUROC: image-level normal/defective classification performance
Pixel AUROC: pixel-level anomaly localization performance
FPS: inference speed
```

The experimental results showed that several categories such as `carpet`, `leather`, `bottle`, and `toothbrush` achieved high Image AUROC scores, indicating that the model can distinguish normal and defective images relatively well on MVTec AD.

### 4.6. Saving Training Artifacts

After training and evaluation, three files were saved for each category:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

Their meanings are:

- `.pth`: fine-tuned model weights.
- `.index`: FAISS Memory Bank built from normal image features.
- `.json`: evaluation metrics and the category-specific classification threshold.

These artifacts were uploaded to the Hugging Face repository:

```text
Manh2005/base-version
```

---

## 5. Moving the Model from Kaggle to Local Machine

After the artifacts were uploaded to Hugging Face, the project was moved to a local computer using the following workflow:

```text
Hugging Face repository
→ download.py downloads model/index/metrics to local storage
→ app.py loads the model by category
→ Gradio receives an input image
→ the model computes the anomaly score
→ the score is compared with the category-specific threshold
→ the system displays the result and heatmap
```

### 5.1. Downloading Artifacts Locally

The `download.py` file uses `snapshot_download` to download the complete Hugging Face repository:

```python
from huggingface_hub import snapshot_download

repo_id = "Manh2005/base-version"
TARGET_DIR = r"E:\dataScience\Year_3_Documents\Project_NCKH\NCKH"

snapshot_download(
    repo_id=repo_id,
    local_dir=TARGET_DIR,
    repo_type="model",
    local_dir_use_symlinks=False
)
```

After downloading, the local directory should follow this structure:

```text
E:\dataScience\Year_3_Documents\Project_NCKH\NCKH
└── mvtec_anomaly_detection
    ├── carpet
    ├── grid
    ├── leather
    ├── tile
    ├── wood
    ├── bottle
    ├── cable
    ├── capsule
    ├── hazelnut
    ├── toothbrush
    └── zipper
```

Each category folder must contain:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

---

## 6. Running the Local Gradio Demo

### 6.1. Installing Dependencies

It is recommended to create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install torch torchvision timm faiss-cpu opencv-python pillow scipy gradio huggingface_hub numpy
```

If the machine has an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch version from the official PyTorch website.

### 6.2. Launching the App

After downloading all required model, index, and metrics files, run:

```bash
python app.py
```

or, if using the corrected version:

```bash
python app_fixed.py
```

Gradio will start a local interface, usually at:

```text
http://127.0.0.1:7860
```

On the interface, the user selects a category, uploads an image, and clicks **KIỂM TRA LỖI** to run the inspection.

---

## 7. Local Inference Logic

The local inference process in `app.py` is:

```text
Input image
→ Resize and CenterCrop to 224x224
→ Normalize using ImageNet mean/std
→ Extract features using Swin Transformer
→ Compare features with the FAISS Memory Bank
→ Compute the anomaly score
→ Resize the anomaly score map into a heatmap
→ Compare the image score with the category-specific threshold
→ Return GOOD or DEFECTIVE
```

The classification rule is:

```python
if image_score > best_threshold:
    status = "DEFECTIVE"
else:
    status = "GOOD"
```

In other words:

```text
score <= threshold  → normal product
score > threshold   → defective product
```

---

## 8. Category-Specific Thresholds

Initially, the app used a fixed fallback threshold:

```python
best_threshold = 200.0
```

This caused a major issue: actual anomaly scores were usually only a few dozen, so a threshold of 200 made almost every image appear as **normal**.

To fix this, the project uses **category-specific thresholds**. Each category has its own `Best_Threshold` stored in:

```text
metrics_<category>.json
```

Example:

```json
{
    "Image_AUROC": 0.9972,
    "Pixel_AUROC": 0.9884,
    "FPS": 1.78,
    "Best_Threshold": 25.0
}
```

### 8.1. Updating Local Thresholds

The `update_thresholds_local.py` script writes category-specific thresholds into each metrics file:

```python
CATEGORY_THRESHOLDS = {
    "carpet": 14.0,
    "grid": 18.0,
    "leather": 20.0,
    "tile": 19.0,
    "wood": 20.0,
    "bottle": 24.0,
    "cable": 20.0,
    "capsule": 15.0,
    "hazelnut": 22.0,
    "toothbrush": 25.0,
    "zipper": 20.0,
}
```

Run:

```bash
python update_thresholds_local.py
```

Then restart the app:

```bash
python app_fixed.py
```

The app must be restarted after changing thresholds to avoid using cached values.

### 8.2. Choosing Thresholds

Thresholds should be chosen based on the actual score distribution of each category:

```text
If defective images are still classified as GOOD  → decrease the threshold
If good images are classified as DEFECTIVE       → increase the threshold
```

Example:

```text
toothbrush good score: 12–25
toothbrush defect score: 26-40
→ a reasonable threshold is around 25
```

A single global threshold should not be used for all categories because each product type has a different anomaly score distribution.

---

## 9. Issues Fixed During Kaggle-to-Local Migration

### 9.1. Incorrect Model Directory

At first, the app searched for the model in the same directory as `app.py`, while the actual artifacts were stored in:

```text
E:\dataScience\Year_3_Documents\Project_NCKH\NCKH\mvtec_anomaly_detection
```

The `base_dir` was corrected to point to the directory that directly contains the category folders.

### 9.2. Missing Model, Index, or Metrics Files

The MVTec AD dataset does not include:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

These files must be generated after training on Kaggle or downloaded from the Hugging Face repository where the artifacts were uploaded.

## 10. Recommended Project Structure

```text
NCKH/
├── app_fixed.py
├── download.py
├── update_thresholds_local.py
├── README.md
└── mvtec_anomaly_detection/
    ├── carpet/
    │   ├── vit_core_swin_carpet.pth
    │   ├── memory_bank_carpet.index
    │   └── metrics_carpet.json
    ├── grid/
    ├── leather/
    ├── tile/
    ├── wood/
    ├── bottle/
    ├── cable/
    ├── capsule/
    ├── hazelnut/
    ├── toothbrush/
    │   ├── vit_core_swin_toothbrush.pth
    │   ├── memory_bank_toothbrush.index
    │   └── metrics_toothbrush.json
    └── zipper/
```

---

## 11. Demo Output

When the user uploads an image, the system returns:

1. An overlay image combining the original image and the heatmap.
2. A heatmap showing suspected anomalous regions.
3. An AI-generated inspection result.

Example defective result:

```text
🔴 DEFECT DETECTED
Category: toothbrush
Analysis score: 45.82
Category threshold: 30.00
```

Example good result:

```text
🟢 NORMAL PRODUCT
Category: toothbrush
Analysis score: 18.24
Category threshold: 30.00
```

---

## 13. Future Improvements

- Automatically compute thresholds from the `train/good` score distribution or a separate validation set.
- Save optimal thresholds based on Youden's J statistic or F1-score.
- Add support for all 15 MVTec AD categories.
- Support batch image upload and batch inspection.
- Export inspection results to CSV.
- Package the system as a desktop application or Docker container.
- Optimize inference speed for both CPU and GPU deployment.

---


## 15. Note

This project is an experimental research and technical demonstration. For real industrial deployment, additional validation, robust threshold calibration, real production data testing, and inference optimization are required.
