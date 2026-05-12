import os
import json
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import faiss
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import gradio as gr

# ==========================================
# CẤU HÌNH CHUNG
# ==========================================
# Thư mục chứa trực tiếp các category: carpet, grid, leather, ..., toothbrush, zipper
# Có thể sửa trực tiếp tại đây hoặc đặt biến môi trường MVTEC_MODEL_DIR.
BASE_DIR = os.environ.get(
    "MVTEC_MODEL_DIR",
    r"E:\dataScience\Year_3_Documents\Project_NCKH\NCKH\mvtec_anomaly_detection",
)

# Ngưỡng dự phòng theo từng category.
# Nếu metrics_<category>.json có Best_Threshold thì app sẽ ưu tiên dùng file metrics.
# Nếu metrics không có hoặc lỗi đọc file, app mới dùng bảng này.
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

# ==========================================
# CƠ CHẾ CACHE MODEL / INDEX
# ==========================================
# Chỉ cache model và FAISS index. Threshold sẽ được đọc lại từ metrics mỗi lần chạy,
# để sau khi bạn chạy update_thresholds_local.py thì không bị giữ ngưỡng cũ trong cache.
LOADED_MODELS_CACHE = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. KIẾN TRÚC MẠNG VÀ HÀM SUY LUẬN
# ==========================================
class ViTCoreExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=2,
        )
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            module.extracted_feature = output

        # Dùng 1 block giống bản base để khớp với memory_bank.index đã tạo trước đó.
        self.backbone.layers[2].blocks[3].register_forward_hook(hook_fn)

    def forward_features(self, x):
        _ = self.backbone.forward_features(x)
        feat = self.backbone.layers[2].blocks[3].extracted_feature

        # Swin output: B, H, W, C -> đổi về B, C, H, W
        feat = feat.permute(0, 3, 1, 2)
        return feat


def calculate_anomaly_scores_full(test_features, memory_bank_index, k=9):
    B, C, H, W_dim = test_features.shape

    test_patches = (
        test_features
        .view(B, C, H * W_dim)
        .permute(0, 2, 1)
        .reshape(-1, C)
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    print("===== DIMENSION CHECK =====")
    print("Feature shape:", test_features.shape)
    print("test_patches.shape:", test_patches.shape)
    print("Feature dim from app.py:", test_patches.shape[1])
    print("FAISS index dim:", memory_bank_index.d)
    print("FAISS index ntotal:", memory_bank_index.ntotal)
    print("===========================")

    if test_patches.shape[1] != memory_bank_index.d:
        raise ValueError(
            f"Feature dimension không khớp: app.py tạo ra dim={test_patches.shape[1]}, "
            f"nhưng FAISS index cần dim={memory_bank_index.d}. "
            f"Hãy sửa ViTCoreExtractor cho giống lúc tạo memory_bank.index."
        )

    if memory_bank_index.ntotal <= 0:
        raise ValueError("FAISS index rỗng, không có vector nào trong memory bank.")

    k = min(k, memory_bank_index.ntotal)

    distances, _ = memory_bank_index.search(test_patches, k)
    distances = np.maximum(distances, 0)
    distances = np.sqrt(distances)

    distances_tensor = torch.tensor(distances)
    softmax_weights = F.softmax(distances_tensor, dim=1)

    base_scores = distances_tensor[:, 0]
    W = 1.0 - softmax_weights[:, 0]

    anomaly_scores_flat = base_scores * W
    anomaly_scores = anomaly_scores_flat.view(B, H * W_dim)

    image_scores = anomaly_scores.max(dim=1)[0].numpy()
    patch_scores = anomaly_scores.view(B, H, W_dim).numpy()

    return image_scores, patch_scores


# ==========================================
# 2. LOAD MODEL / INDEX / THRESHOLD
# ==========================================
def get_category_dir(base_dir, category):
    return os.path.join(base_dir, category)


def load_threshold(category, base_dir):
    """
    Ưu tiên đọc Best_Threshold trong metrics_<category>.json.
    Nếu không có thì dùng CATEGORY_THRESHOLDS.
    Không fallback về 200.0 nữa vì ngưỡng đó làm hầu hết ảnh bị báo TỐT.
    """
    category_dir = get_category_dir(base_dir, category)
    metrics_path = os.path.join(category_dir, f"metrics_{category}.json")

    threshold = CATEGORY_THRESHOLDS.get(category)
    source = "CATEGORY_THRESHOLDS"

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            # Hỗ trợ vài tên key phổ biến để tránh lỗi do ghi khác tên.
            for key in ("Best_Threshold", "best_threshold", "threshold", "Threshold"):
                if key in metrics_data:
                    threshold = float(metrics_data[key])
                    source = f"{os.path.basename(metrics_path)}::{key}"
                    break

        except Exception as e:
            print(f"Không đọc được metrics file cho {category}: {e}")

    if threshold is None:
        raise ValueError(
            f"Chưa có threshold cho category '{category}'. "
            f"Hãy thêm vào CATEGORY_THRESHOLDS hoặc tạo {metrics_path} với key Best_Threshold."
        )

    print(f"Threshold for {category}: {threshold} | source: {source}")
    return float(threshold), source


def get_or_load_model(category, base_dir):
    global LOADED_MODELS_CACHE

    if category in LOADED_MODELS_CACHE:
        return LOADED_MODELS_CACHE[category]

    category_dir = get_category_dir(base_dir, category)
    model_path = os.path.join(category_dir, f"vit_core_swin_{category}.pth")
    index_path = os.path.join(category_dir, f"memory_bank_{category}.index")

    print("===== LOAD MODEL DEBUG =====")
    print("Base dir    :", base_dir)
    print("Category dir:", category_dir)
    print("Model path  :", model_path, "| exists:", os.path.exists(model_path))
    print("Index path  :", index_path, "| exists:", os.path.exists(index_path))
    if os.path.exists(category_dir):
        print("Files in category dir:", os.listdir(category_dir))
    print("============================")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Không tìm thấy index: {index_path}")

    model = ViTCoreExtractor().to(device)
    ckpt = torch.load(model_path, map_location=device)

    # Phòng trường hợp checkpoint được lưu dưới dạng dict.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model.load_state_dict(ckpt, strict=True)
    model.eval()

    index = faiss.read_index(index_path)
    print("Loaded FAISS index dim:", index.d)
    print("Loaded FAISS index ntotal:", index.ntotal)

    LOADED_MODELS_CACHE[category] = {
        "model": model,
        "index": index,
    }

    return LOADED_MODELS_CACHE[category]


# ==========================================
# 3. HÀM XỬ LÝ LÕI CHO GIAO DIỆN WEB
# ==========================================
def process_image(input_img, category):
    try:
        if input_img is None:
            return None, None, "Vui lòng tải lên một bức ảnh."

        cache_data = get_or_load_model(category, BASE_DIR)
        model = cache_data["model"]
        index = cache_data["index"]

        # Đọc threshold riêng theo category mỗi lần chạy.
        best_threshold, threshold_source = load_threshold(category, BASE_DIR)

        # Tiền xử lý ảnh gốc
        original_img = Image.fromarray(input_img).convert("RGB")

        display_img = original_img.resize((256, 256), Image.Resampling.BILINEAR)
        w, h = display_img.size
        left, top = (w - 224) / 2, (h - 224) / 2
        right, bottom = (w + 224) / 2, (h + 224) / 2
        display_img = display_img.crop((left, top, right, bottom))

        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(original_img).unsqueeze(0).to(device)

        # Chạy AI Inference
        with torch.no_grad():
            features = model.forward_features(input_tensor)
            img_scores, patch_scores = calculate_anomaly_scores_full(features, index)

        img_score = float(img_scores[0])

        # Hậu xử lý Heatmap
        score_map = cv2.resize(patch_scores[0], (224, 224), interpolation=cv2.INTER_LINEAR)
        score_map = gaussian_filter(score_map, sigma=2)

        score_map_norm = (
            (score_map - score_map.min()) /
            (score_map.max() - score_map.min() + 1e-8) * 255
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(np.array(display_img), 0.5, heatmap, 0.5, 0)

        # Đưa ra phán quyết
        is_defect = img_score > best_threshold
        status = "🔴 PHÁT HIỆN LỖI KHUYẾT TẬT" if is_defect else "🟢 SẢN PHẨM BÌNH THƯỜNG"

        result_text = (
            f"{status}\n"
            f"Danh mục: {category}\n"
            f"Điểm phân tích: {img_score:.2f}\n"
            f"Ngưỡng của danh mục: {best_threshold:.2f}\n"
            f"Nguồn ngưỡng: {threshold_source}\n"
            f"Quy tắc: score > threshold → LỖI"
        )

        print("===== INFERENCE RESULT =====")
        print(result_text)
        print("============================")

        return blended, heatmap, result_text

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return None, None, f"LỖI KHI CHẠY MÔ HÌNH:\n{str(e)}"


# ==========================================
# 4. THIẾT KẾ GIAO DIỆN GRADIO
# ==========================================
MVTEC_CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "toothbrush", "zipper",
]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    gr.Markdown(
        """
        # Hệ thống Phân tích Lỗi Bề mặt Công nghiệp (RealNet-SIA)
        *Tự động quét khuyết tật bằng Swin Transformer & FAISS Memory Bank.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=MVTEC_CATEGORIES,
                value="toothbrush",
                label="1. Chọn Danh mục Sản phẩm",
            )
            image_input = gr.Image(label="2. Kéo thả hoặc chọn ảnh kiểm tra", type="numpy")
            run_btn = gr.Button("KIỂM TRA LỖI", variant="primary")

        with gr.Column(scale=2):
            score_output = gr.Textbox(label="Kết luận từ AI", lines=6)
            with gr.Row():
                blended_output = gr.Image(label="Ảnh Định vị Lỗi")
                heatmap_output = gr.Image(label="Bản đồ Nhiệt")

    run_btn.click(
        fn=process_image,
        inputs=[image_input, category_dropdown],
        outputs=[blended_output, heatmap_output, score_output],
    )

if __name__ == "__main__":
    print("Running on device:", device)
    print("Using BASE_DIR:", BASE_DIR)
    app.launch(share=False, debug=True, show_error=True)
