from huggingface_hub import snapshot_download

repo_id = "calmm-m/upgrade-version"
print(f"🚀 Đang tải toàn bộ dữ liệu từ: {repo_id}...")

snapshot_download(
    repo_id=repo_id, 
    local_dir=".", # Tải thẳng vào thư mục hiện tại
    repo_type="model",
    local_dir_use_symlinks=False 
)
print("✅ Đã tải xong! Các thư mục danh mục (toothbrush, zipper...) đã có sẵn trên máy.")