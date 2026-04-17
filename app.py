import os
import json
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
# 1. KIẾN TRÚC MẠNG VÀ HÀM SUY LUẬN
# ==========================================
class ViTCoreExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Khởi tạo khung mạng Swin giống hệt lúc train
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            module.extracted_feature = output
        self.backbone.layers[2].blocks[2].register_forward_hook(hook_fn)
        self.backbone.layers[2].blocks[3].register_forward_hook(hook_fn)

    def forward_features(self, x):
        _ = self.backbone.forward_features(x)
        feat3 = self.backbone.layers[2].blocks[2].extracted_feature
        feat4 = self.backbone.layers[2].blocks[3].extracted_feature
        block3 = feat3.permute(0, 3, 1, 2)
        block4 = feat4.permute(0, 3, 1, 2)
        block4_pooled = self.avg_pool(block4)
        return torch.cat([block3, block4_pooled], dim=1)

def calculate_anomaly_scores_full(test_features, memory_bank_index, k=9):
    B, C, H, W_dim = test_features.shape 
    test_patches = test_features.view(B, C, H * W_dim).permute(0, 2, 1).reshape(-1, C).cpu().numpy()
    
    distances, _ = memory_bank_index.search(test_patches, k)
    distances = np.maximum(distances, 0) # Chặn số âm
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
# 2. HÀM XỬ LÝ LÕI CHO GIAO DIỆN WEB
# ==========================================
def process_image(input_img, category):
    if input_img is None:
        return None, None, "⚠️ Vui lòng tải lên một bức ảnh."

    # 🟢 SỬA LỖI ĐƯỜNG DẪN: Lấy thư mục gốc chứa file app.py này
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ghép nối đường dẫn tuyệt đối một cách an toàn
    model_path = os.path.join(base_dir, category, f"vit_core_swin_{category}.pth")
    index_path = os.path.join(base_dir, category, f"memory_bank_{category}.index")
    metrics_path = os.path.join(base_dir, category, f"metrics_{category}.json")

    # Kiểm tra khắt khe xem file có tồn tại không
    if not os.path.exists(model_path) or not os.path.exists(index_path):
        return None, None, f"❌ LỖI: Không tìm thấy Model hoặc Index tại thư mục: {os.path.join(base_dir, category)}"

    # 🟢 ĐỌC NGƯỠNG (THRESHOLD)
    best_threshold = 200.0 # Giá trị mặc định
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            best_threshold = metrics_data.get('Best_Threshold', 200.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model (Strict=True để báo lỗi ngay nếu trọng số bị sai)
    model = ViTCoreExtractor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    # Load FAISS Index
    index = faiss.read_index(index_path)

    # 🟢 SỬA LỖI ĐỒNG BỘ TIỀN XỬ LÝ ẢNH
    original_img = Image.fromarray(input_img).convert('RGB')
    
    # 1. Transform chuẩn đưa vào Model (Y hệt lúc Train trên Kaggle)
    transform = T.Compose([
        T.Resize((256, 256)), 
        T.CenterCrop((224, 224)),      
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # 2. Xử lý ảnh hiển thị (Phải crop y hệt Tensor để Heatmap không bị lệch)
    display_img = original_img.resize((256, 256), Image.Resampling.BILINEAR)
    # Cắt chuẩn xác 224x224 từ tâm (giống T.CenterCrop)
    w, h = display_img.size
    left, top = (w - 224) / 2, (h - 224) / 2
    right, bottom = (w + 224) / 2, (h + 224) / 2
    display_img = display_img.crop((left, top, right, bottom))

    # Suy luận
    with torch.no_grad():
        features = model.forward_features(input_tensor)
        img_scores, patch_scores = calculate_anomaly_scores_full(features, index)
    
    img_score = float(img_scores[0])

    # 🟢 SỬA LỖI HEATMAP NHÒE
    score_map = cv2.resize(patch_scores[0], (224, 224), interpolation=cv2.INTER_LINEAR)
    score_map = gaussian_filter(score_map, sigma=2) # Đã giảm sigma xuống 2
    
    # Chuẩn hóa màu Heatmap
    score_map_norm = ((score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_map_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Hòa trộn Heatmap lên ảnh gốc đã crop
    blended = cv2.addWeighted(np.array(display_img), 0.5, heatmap, 0.5, 0)

    # Phán quyết
    is_defect = img_score > best_threshold
    status = "🔴 PHÁT HIỆN LỖI KHUYẾT TẬT" if is_defect else "🟢 SẢN PHẨM BÌNH THƯỜNG"
        
    result_text = (
        f"{status}\n"
        f"• Điểm phân tích: {img_score:.2f}\n"
        f"• Ngưỡng an toàn: {best_threshold:.2f}"
    )

    return blended, heatmap, result_text

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN GRADIO
# ==========================================
# Hãy thêm các danh mục bạn đã tải về máy vào danh sách này
MVTEC_CATEGORIES = [
    'toothbrush', 'zipper', 'bottle', 'capsule', 'hazelnut' 
]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    gr.Markdown(
        """
        # 🚀 Hệ thống Phân tích Lỗi Bề mặt Công nghiệp (RealNet-SIA)
        *Tự động quét khuyết tật bằng Swin Transformer & FAISS Memory Bank.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(choices=MVTEC_CATEGORIES, value="toothbrush", label="1. Chọn Danh mục Sản phẩm")
            image_input = gr.Image(label="2. Kéo thả hoặc chọn ảnh kiểm tra")
            run_btn = gr.Button("🔍 KIỂM TRA LỖI", variant="primary")
            
        with gr.Column(scale=2):
            score_output = gr.Textbox(label="Kết luận từ AI", lines=3)
            with gr.Row():
                blended_output = gr.Image(label="Ảnh Định vị Lỗi (Khớp 100%)")
                heatmap_output = gr.Image(label="Bản đồ Nhiệt (Sigma=2)")

    run_btn.click(
        fn=process_image, 
        inputs=[image_input, category_dropdown], 
        outputs=[blended_output, heatmap_output, score_output]
    )

if __name__ == "__main__":
    # Bật debug=True để nếu có lỗi, nó sẽ in thẳng ra Terminal cho bạn dễ xem
    app.launch(share=False, debug=True)