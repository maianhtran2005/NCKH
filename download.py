from huggingface_hub import snapshot_download

repo_id = "Manh2005/base-version"
print(f" Đang tải toàn bộ dữ liệu từ: {repo_id}...")


TARGET_DIR = r"E:\\dataScience\\Year_3_Documents\\Project_NCKH\\NCKH\\mvtec_anomaly_detection"

snapshot_download(
    repo_id=repo_id, 
    local_dir=TARGET_DIR, 
    repo_type="model",
    local_dir_use_symlinks=False 
)
print(f"✅ Đã tải xong! Các file đã được lưu trực tiếp vào: {TARGET_DIR}")