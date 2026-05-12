# MVTec AD Anomaly Detection Demo – ViT-Core / Swin Transformer + FAISS

## 1. Giới thiệu

Dự án này xây dựng hệ thống phát hiện lỗi bề mặt công nghiệp trên bộ dữ liệu **MVTec AD** theo hướng tham khảo mô hình **ViT-Core: Lightweight Anomaly Detection Model using Transformer-based Feature Extractor**. Quy trình được triển khai theo hai giai đoạn chính:

1. **Huấn luyện và đánh giá trên Kaggle**: xử lý dữ liệu MVTec AD, fine-tune mô hình theo từng danh mục sản phẩm, xây dựng FAISS Memory Bank và lưu các artifact cần thiết.
2. **Triển khai local trên máy cá nhân**: tải model/index/metrics từ Hugging Face về máy, chạy giao diện Gradio để kiểm thử ảnh đầu vào, hiển thị heatmap và kết luận sản phẩm **TỐT** hoặc **LỖI** theo ngưỡng riêng của từng category.

Dự án tập trung vào bài toán **industrial anomaly detection**, trong đó mô hình học đặc trưng của ảnh bình thường và phát hiện các vùng có biểu hiện khác biệt so với phân bố bình thường.

---

## 2. Tài liệu tham khảo và dữ liệu

### 2.1. Bài báo tham khảo

- **ViT-Core: Lightweight Anomaly Detection Model using Transformer-based Feature Extractor**
- IEEE Access, 2025
- DOI: https://doi.org/10.1109/ACCESS.2025.3618462
- Trang tham khảo: https://www.researchgate.net/publication/396267323_ViT-Core_Lightweight_Anomaly_Detection_Model_using_Transformer-based_Feature_Extractor

Ý tưởng chính được tham khảo từ bài báo là sử dụng **Transformer-based feature extractor**, cụ thể là hướng dùng **Swin Transformer** để trích xuất đặc trưng thay cho backbone CNN truyền thống, từ đó xây dựng biểu diễn đặc trưng hiệu quả hơn cho bài toán anomaly detection.

### 2.2. Bộ dữ liệu

- MVTec AD official dataset: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Kaggle mirror sử dụng trong quá trình thực nghiệm: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
- Hugging Face dataset mirror: https://huggingface.co/datasets/Voxel51/mvtec-ad

MVTec AD gồm nhiều nhóm sản phẩm và bề mặt công nghiệp, mỗi category có cấu trúc dữ liệu gồm:

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

Trong dự án này, tập `train/good` được dùng để xây dựng biểu diễn bình thường, còn tập `test/good` và `test/<defect_type>` được dùng để kiểm tra khả năng phân loại ảnh tốt/lỗi và trực quan hóa vùng bất thường.

---

## 3. Các category đã triển khai

Phiên bản local hiện tập trung vào các category sau:

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

Mỗi category cần có bộ ba artifact riêng:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

Ví dụ với `toothbrush`:

```text
mvtec_anomaly_detection/
└── toothbrush/
    ├── vit_core_swin_toothbrush.pth
    ├── memory_bank_toothbrush.index
    └── metrics_toothbrush.json
```

---

## 4. Những gì đã thực hiện trên Kaggle

### 4.1. Chuẩn bị dữ liệu

Dữ liệu MVTec AD được đưa vào môi trường Kaggle theo đường dẫn dạng:

```text
/kaggle/input/datasets/ipythonx/mvtec-ad
```

Notebook huấn luyện đọc dữ liệu theo từng category, ví dụ:

```text
/kaggle/input/datasets/ipythonx/mvtec-ad/toothbrush/train/good
/kaggle/input/datasets/ipythonx/mvtec-ad/toothbrush/test/good
/kaggle/input/datasets/ipythonx/mvtec-ad/toothbrush/test/defective
```

### 4.2. Fine-tune mô hình bằng Cut-Paste

Mô hình được fine-tune theo hướng tự giám sát bằng kỹ thuật **Cut-Paste augmentation**. Ảnh lỗi giả được tạo từ ảnh bình thường bằng cách cắt một vùng ảnh rồi dán sang vị trí khác. Cách làm này giúp mô hình học được sự khác biệt giữa ảnh bình thường và ảnh bất thường mà không cần dùng nhãn lỗi thật trong quá trình huấn luyện.

Quy trình cho từng category:

```text
Ảnh train/good
→ tạo ảnh lỗi giả bằng Cut-Paste
→ fine-tune backbone Swin Transformer
→ trích xuất đặc trưng
→ xây dựng FAISS Memory Bank
→ đánh giá trên test/good và test/defect
```

### 4.3. Feature extractor

Mô hình sử dụng backbone:

```python
swin_base_patch4_window7_224
```

Trong phiên bản local đang sử dụng, extractor lấy đặc trưng từ block:

```python
self.backbone.layers[2].blocks[3]
```

Đặc trưng đầu ra được đưa về dạng:

```text
B, C, H, W
```

sau đó tách thành các patch feature để so sánh với FAISS Memory Bank.

### 4.4. Xây dựng FAISS Memory Bank

Sau khi fine-tune, mô hình trích xuất đặc trưng từ ảnh `train/good`. Các đặc trưng này được đưa vào FAISS index để tạo Memory Bank cho từng category.

Khi inference, ảnh mới được trích xuất feature, sau đó so sánh khoảng cách với các vector bình thường trong FAISS Memory Bank. Điểm bất thường càng cao thì ảnh càng có khả năng là ảnh lỗi.

### 4.5. Đánh giá mô hình

Trên Kaggle, mô hình được đánh giá bằng các chỉ số:

```text
Image AUROC: đánh giá phân loại ảnh tốt/lỗi
Pixel AUROC: đánh giá khả năng khoanh vùng lỗi
FPS: tốc độ suy luận
```

Kết quả huấn luyện cho thấy các category như `carpet`, `leather`, `bottle`, `toothbrush` đạt Image AUROC cao, chứng tỏ mô hình có khả năng phân biệt ảnh tốt và ảnh lỗi tương đối tốt trên bộ MVTec AD.

### 4.6. Lưu artifact sau huấn luyện

Với mỗi category, sau khi huấn luyện và đánh giá, hệ thống lưu ra 3 file:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

Trong đó:

- `.pth`: trọng số mô hình sau khi fine-tune.
- `.index`: FAISS Memory Bank chứa feature của ảnh bình thường.
- `.json`: lưu thông tin đánh giá và ngưỡng phân loại của category.

Các file này được upload lên Hugging Face repo:

```text
Manh2005/base-version
```

---

## 5. Đưa mô hình từ Kaggle về local

Sau khi có artifact trên Hugging Face, dự án được đưa về chạy trên máy cá nhân theo quy trình:

```text
Hugging Face repo
→ download.py tải model/index/metrics về local
→ app.py load model theo category
→ Gradio nhận ảnh đầu vào
→ mô hình tính anomaly score
→ so sánh với threshold riêng
→ hiển thị kết quả và heatmap
```

### 5.1. Tải artifact về local

File `download.py` dùng `snapshot_download` để tải toàn bộ repo về thư mục local:

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

Sau khi tải xong, cần kiểm tra cấu trúc thư mục local:

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

Trong mỗi category cần có đủ:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

---

## 6. Chạy demo local bằng Gradio

### 6.1. Cài đặt thư viện

Khuyến nghị tạo môi trường ảo trước khi chạy:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Cài các thư viện cần thiết:

```bash
pip install torch torchvision timm faiss-cpu opencv-python pillow scipy gradio huggingface_hub numpy
```

Nếu máy có GPU NVIDIA, nên cài PyTorch theo đúng phiên bản CUDA từ trang chính thức của PyTorch.

### 6.2. Chạy app

Sau khi đã tải đủ model/index/metrics, chạy:

```bash
python app.py
```

hoặc nếu dùng bản đã sửa:

```bash
python app_fixed.py
```

Gradio sẽ mở giao diện local tại địa chỉ dạng:

```text
http://127.0.0.1:7860
```

Trên giao diện, người dùng chọn category, tải ảnh cần kiểm tra, sau đó nhấn **KIỂM TRA LỖI**.

---

## 7. Logic inference trên local

Quy trình inference trong `app.py` gồm các bước:

```text
Ảnh đầu vào
→ Resize và CenterCrop về 224x224
→ Normalize theo ImageNet mean/std
→ Swin Transformer trích xuất feature
→ So sánh feature với FAISS Memory Bank
→ Tính anomaly score
→ Resize score map thành heatmap
→ So sánh image score với threshold riêng của category
→ Trả kết luận TỐT hoặc LỖI
```

Quy tắc phân loại:

```python
if image_score > best_threshold:
    status = "LỖI"
else:
    status = "TỐT"
```

Tức là:

```text
score <= threshold  → sản phẩm bình thường
score > threshold   → phát hiện lỗi
```

---

## 8. Ngưỡng riêng theo từng category

Ban đầu app từng dùng ngưỡng mặc định:

```python
best_threshold = 200.0
```

Điều này gây ra vấn đề: điểm phân tích thực tế của ảnh chỉ khoảng vài chục, nên nếu threshold là 200 thì gần như mọi ảnh đều bị báo là **SẢN PHẨM BÌNH THƯỜNG**.

Để khắc phục, dự án chuyển sang cơ chế **ngưỡng riêng theo category**. Mỗi category có một `Best_Threshold` riêng trong file:

```text
metrics_<category>.json
```

Ví dụ:

```json
{
    "Image_AUROC": 0.9972,
    "Pixel_AUROC": 0.9884,
    "FPS": 1.78,
    "Best_Threshold": 30.0
}
```

### 8.1. Cập nhật ngưỡng local

File `update_thresholds_local.py` được dùng để ghi ngưỡng riêng vào từng file metrics:

```python
CATEGORY_THRESHOLDS = {
    "carpet": 35.0,
    "grid": 30.0,
    "leather": 30.0,
    "tile": 35.0,
    "wood": 35.0,
    "bottle": 25.0,
    "cable": 35.0,
    "capsule": 35.0,
    "hazelnut": 30.0,
    "toothbrush": 30.0,
    "zipper": 35.0,
}
```

Chạy:

```bash
python update_thresholds_local.py
```

Sau đó chạy lại app:

```bash
python app_fixed.py
```

Lưu ý: cần restart app sau khi thay đổi threshold để tránh dùng lại cache cũ.

### 8.2. Cách chọn threshold

Ngưỡng nên được chọn dựa trên phân bố score thực tế của từng category:

```text
Nếu ảnh lỗi vẫn bị báo TỐT  → giảm threshold
Nếu ảnh tốt bị báo LỖI      → tăng threshold
```

Ví dụ:

```text
toothbrush good score: 12–25
toothbrush defect score: 40–70
→ threshold hợp lý khoảng 30
```

Không nên dùng một threshold chung cho mọi category vì mỗi loại sản phẩm có phân bố score khác nhau.

---

## 9. Các lỗi đã xử lý khi chuyển từ Kaggle sang local

### 9.1. Sai đường dẫn model

Ban đầu app tìm model trong thư mục chứa `app.py`, trong khi artifact thật nằm trong:

```text
E:\dataScience\Year_3_Documents\Project_NCKH\NCKH\mvtec_anomaly_detection
```

Đã sửa `base_dir` để trỏ đến đúng thư mục chứa trực tiếp các category.

### 9.2. Không có file model/index/metrics

Bộ dữ liệu MVTec AD không chứa sẵn các file:

```text
vit_core_swin_<category>.pth
memory_bank_<category>.index
metrics_<category>.json
```

Các file này phải được tạo sau khi train trên Kaggle hoặc tải từ Hugging Face repo đã upload artifact.

### 9.3. Sai chiều feature với FAISS index

Khi app dùng extractor 2 block nhưng FAISS index được tạo từ extractor 1 block, FAISS báo lỗi:

```text
AssertionError: assert d == self.d
```

Đã sửa `ViTCoreExtractor` trong local app để dùng đúng extractor 1 block, khớp với FAISS index đã tạo từ bản base.

### 9.4. Threshold quá cao

Ngưỡng mặc định 200 làm mọi ảnh đều được báo là bình thường. Đã chuyển sang `Best_Threshold` riêng theo từng category trong file metrics.

---

## 10. Cấu trúc project đề xuất

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

## 11. Kết quả đầu ra của demo

Khi người dùng tải ảnh lên giao diện, hệ thống trả về:

1. Ảnh overlay giữa ảnh đầu vào và heatmap.
2. Heatmap vùng nghi ngờ lỗi.
3. Kết luận từ AI:

```text
🔴 PHÁT HIỆN LỖI KHUYẾT TẬT
Danh mục: toothbrush
Điểm phân tích: 45.82
Ngưỡng của danh mục: 30.00
```

hoặc:

```text
🟢 SẢN PHẨM BÌNH THƯỜNG
Danh mục: toothbrush
Điểm phân tích: 18.24
Ngưỡng của danh mục: 30.00
```

---

## 12. Hạn chế hiện tại

- Ngưỡng threshold hiện cần hiệu chỉnh thủ công theo từng category.
- Phiên bản local mới thử nghiệm trên một số category thuộc MVTec AD, chưa mở rộng toàn bộ 15 category.
- Kết quả phụ thuộc chặt chẽ vào việc model `.pth`, FAISS `.index` và extractor trong `app.py` phải khớp nhau.
- Heatmap được normalize theo từng ảnh, do đó vùng đỏ trên ảnh tốt không nhất thiết là lỗi; kết luận chính thức dựa vào `image_score` so với `Best_Threshold`.
- Mô hình hiện mới là demo nghiên cứu, chưa phải hệ thống kiểm định công nghiệp hoàn chỉnh.

---

## 13. Hướng phát triển tiếp theo

- Tự động tính threshold theo phân bố score của `train/good` hoặc tập validation riêng.
- Lưu threshold tối ưu theo Youden’s J statistic hoặc F1-score trên tập test có nhãn.
- Bổ sung đủ 15 category của MVTec AD.
- Cho phép upload nhiều ảnh cùng lúc.
- Xuất báo cáo kết quả kiểm tra theo file CSV.
- Đóng gói thành ứng dụng desktop hoặc Docker container.
- Tối ưu tốc độ inference cho CPU và GPU local.

---

## 14. Tóm tắt quy trình thực hiện

```text
1. Nghiên cứu bài báo ViT-Core và bài toán anomaly detection.
2. Sử dụng bộ dữ liệu MVTec AD trên Kaggle.
3. Fine-tune mô hình Swin Transformer bằng Cut-Paste augmentation.
4. Trích xuất đặc trưng ảnh bình thường từ train/good.
5. Xây dựng FAISS Memory Bank cho từng category.
6. Đánh giá bằng Image AUROC, Pixel AUROC và FPS.
7. Lưu model, index, metrics thành artifact.
8. Upload artifact lên Hugging Face.
9. Tải artifact về local bằng download.py.
10. Sửa app.py để khớp đường dẫn local và extractor.
11. Sửa lỗi dimension mismatch giữa feature và FAISS index.
12. Thêm cơ chế threshold riêng theo category.
13. Chạy demo local bằng Gradio để kiểm tra ảnh tốt/lỗi.
```

---

## 15. Ghi chú

Dự án này là bản thực nghiệm phục vụ nghiên cứu và demo kỹ thuật. Để sử dụng trong môi trường sản xuất thực tế, cần bổ sung quy trình kiểm định dữ liệu, hiệu chỉnh threshold ổn định hơn, đánh giá trên dữ liệu sản xuất thật và tối ưu độ trễ suy luận.
