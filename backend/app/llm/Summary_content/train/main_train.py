from Summary_content.model_summary_contents.summarization_utils import print_memory_usage
from transformers.models import AutoTokenizer
from sentence_transformers import SentenceTransformer
from Summary_content.model_summary_contents.create_dataset_cache import VietnameseSummarizationDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from Summary_content.model_summary_contents.light_phobert_model import LightweightPhoBERTSUM

from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
embed_model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base").to(device)
file_data_train = "/home/manh/Data/data_colab-20250921T163531Z-1-001/data_colab/vietnamese_news.csv"

cache_dir = "/Summary_content/train/folder_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache = os.path.join(cache_dir, "train.pkl")
val_cache = os.path.join(cache_dir, "val.pkl")
test_cache = os.path.join(cache_dir, "test.pkl")

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
chunk_size = 100


def main():
    train_dfs, val_dfs, test_dfs = [], [], []
    print("Doc du lieu theo chunk")
    for chunk_ in pd.read_csv(file_data_train, usecols=['content', 'summary'], chunksize=chunk_size):
        chunk = chunk_.dropna(subset=['content', 'summary']).reset_index(drop=True)
        train_chunk, temp_chunk = train_test_split(chunk, test_size=0.2, random_state=seed)
        val_chunk, test_chunk = train_test_split(temp_chunk, test_size=0.5, random_state=seed)
        train_dfs.append(train_chunk)
        val_dfs.append(val_chunk)
        test_dfs.append(test_chunk)
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    if not (os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache)):
        print("Tạo cache mới từ dữ liệu gốc...")
        embed_model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base").to(device)
        
        print("Tạo train cache...")
        train_dataset = VietnameseSummarizationDataset(train_df['content'].values, train_df['summary'].values,
                                                       tokenizer, embed_model, top_n=2, max_content_length=96, cache_file=train_cache)
        print("train_dataset : ", train_dataset)
        
        print("Tạo validation cache...")
        val_dataset = VietnameseSummarizationDataset(
            val_df['content'].values, val_df['summary'].values,
            tokenizer, embed_model, top_n=2,
            max_content_length=96, cache_file=val_cache
        )
        
        print("Tạo test cache...")
        test_dataset = VietnameseSummarizationDataset(
            test_df['content'].values, test_df['summary'].values,
            tokenizer, embed_model, top_n=2,
            max_content_length=96, cache_file=test_cache
        )
        
        del embed_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    #  chia nho cach file da co
    print("Chia nhỏ cache đã tiền xử lý...")
    train_chunk_files, val_chunk_files, test_chunk_files = split_existing_cache()
    print_memory_usage("After splitting cache")
    
    #      Su dung StreamingDataset
    print("Tạo streaming datasets...")
    train_dataset = StreamingDataset(train_chunk_files, shuffle=True)
    val_dataset = StreamingDataset(val_chunk_files, shuffle=False)
    print_memory_usage("After creating datasets")
    
    # === THAY ĐỔI: Sử dụng mô hình nhẹ hơn ===
    print("Khởi tạo mô hình nhẹ...")
    model = LightweightPhoBERTSUM()
    model.to(device)
    print(f"Tổng số tham số: {sum(p.numel() for p in model.parameters()):,}")
    print_memory_usage("After loading model")
    
    # Tạo thư mục checkpoint
    checkpoint_dir = "weight/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Kiểm tra xem có checkpoint cũ không
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            print(f"Tìm thấy checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Đã tải checkpoint từ epoch {checkpoint['epoch']}")
    
    epochs = 10
    print("Bắt đầu huấn luyện với memory optimization...")
    model = train_model_memory_efficient(
        model, train_dataset, val_dataset, device,
        epochs=epochs, batch_size=4, accum_steps=4,
        checkpoint_dir=checkpoint_dir,
        use_focal_loss=True
    )
    print_memory_usage("After training")
    
    # Lưu mô hình
    model_save_path = "weight/phobertsum_lightweight.pt"
    print(f"Lưu mô hình vào {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    
    # Đánh giá hiệu quả trên một phần nhỏ của tập test
    print("Đánh giá mô hình trên tập test (lấy mẫu)...")
    results = evaluate_memory_efficient(model, tokenizer, device, test_chunk_files, max_sent=3)
    
    # Demo
    print("\nVí dụ tóm tắt:")
    try:
        # Lấy một vài mẫu từ tập test
        with open(test_chunk_files[0], 'rb') as f:
            test_samples = pickle.load(f)
        
        for i in range(min(3, len(test_samples))):
            sample = test_samples[i]
            input_ids = sample['input_ids']
            sentences = [tokenizer.decode(sent, skip_special_tokens=True) for sent in input_ids]
            content = " ".join(sentences)
            
            print(f"\nMẫu {i + 1}:")
            # print("Văn bản:", content[:300] + "..." if len(content) > 300 else content)
            print("Văn bản:", content)
            summary = extractive_summarize_from_inputids(content, model, tokenizer, device)
            print("Tóm tắt sinh:", summary)
            print("-" * 50)
    except Exception as e:
        print(f"Lỗi khi demo: {e}")
    
    print("✅ Hoàn thành huấn luyện và đánh giá mô hình!")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_usage("After completion")

if __name__ == '__main__':
    main()
