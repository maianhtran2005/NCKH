import os
import json

# Thư mục chứa trực tiếp các category: carpet, grid, ..., toothbrush, zipper
BASE_DIR = r"E:\dataScience\Year_3_Documents\Project_NCKH\NCKH\mvtec_anomaly_detection"

# Sửa các số này theo score thực tế của từng category.
# Quy tắc trong app.py: score > Best_Threshold thì báo LỖI.
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

for category, threshold in CATEGORY_THRESHOLDS.items():
    category_dir = os.path.join(BASE_DIR, category)
    metrics_path = os.path.join(category_dir, f"metrics_{category}.json")

    if not os.path.exists(category_dir):
        print(f"Không thấy thư mục category: {category_dir}")
        continue

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Không đọc được {metrics_path}: {e}. Tạo file mới.")
            metrics = {}
    else:
        metrics = {}

    metrics["Best_Threshold"] = float(threshold)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Đã cập nhật {metrics_path} -> Best_Threshold = {threshold}")

print("Hoàn tất cập nhật threshold theo từng category.")
