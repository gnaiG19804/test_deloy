import os
import requests
from tqdm import tqdm

MODEL_DIR = os.path.join("app", "llm", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phobertsum_lightweight.pt")
URL = "https://github.com/Group9-prj1/Project-1-Group-9/releases/download/v1.0-model/phobertsum_lightweight.pt"

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f" Đang tải model từ GitHub: {URL}")

        with requests.get(URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 8192
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            progress_bar.close()
        print("Model tải hoàn tất!")
    else:
        print(" Model đã tồn tại, bỏ qua tải lại.")

if __name__ == "__main__":
    ensure_model()
