import os
from collections import Counter
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import copy
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from Summary_content.train.streaming_dataset import StreamingDataset
import gc
import pickle
from main_model import split_cache_file
import numpy as np


MAX_ERRORS_TO_PRINT = 5  # Chỉ in tối đa 5 lỗi cùng loại
error_counters = {}

def collate_fn(batch):
    """Xử lý batch với số câu khác nhau."""
    # Kiểm tra batch rỗng
    if not batch:
        return None
    
    try:
        max_sent_count = max([b['input_ids'].shape[0] for b in batch])
        for i, b in enumerate(batch):
            sent_count = b['input_ids'].shape[0]
            if sent_count < max_sent_count:
                padding_size = max_sent_count - sent_count
                seq_len = b['input_ids'].shape[1]
                
                # Sửa: sử dụng kiểu số đúng
                padding = torch.zeros((padding_size, seq_len), dtype=torch.long)
                batch[i]['input_ids'] = torch.cat([b['input_ids'], padding])
                
                attn_padding = torch.zeros((padding_size, seq_len), dtype=torch.long)
                batch[i]['attention_mask'] = torch.cat([b['attention_mask'], attn_padding])
                
                label_padding = torch.zeros(padding_size, dtype=torch.float)
                batch[i]['labels'] = torch.cat([b['labels'], label_padding])
        
        return {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'labels': torch.stack([b['labels'] for b in batch])
        }
    except Exception as e:
        error_type = str(type(e).__name__)
        if error_type not in error_counters:
            error_counters[error_type] = 0
        error_counters[error_type] += 1
        
        if error_counters[error_type] <= MAX_ERRORS_TO_PRINT:
            print(f"Lỗi trong collate_fn ({error_type}): {e}")
        
        # Trả về None để bỏ qua batch này
        return None

def focal_loss(outputs, targets, gamma=2.0, alpha=0.25):
    """Focal loss cho phân loại nhị phân với dữ liệu mất cân bằng."""
    BCE_loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # ngăn NaN khi xác suất bằng 0
    loss = alpha * (1 - pt) ** gamma * BCE_loss
    return loss.mean()

# === PHẦN 5: HÀM HUẤN LUYỆN GIẢI PHÓNG BỘ NHỚ THƯỜNG XUYÊN ===
def train_model_memory_efficient(model, train_dataset, val_dataset, device, epochs,
                                 batch_size=4, accum_steps=4, lr=1e-5, patience=3,
                                 min_delta=0.001, checkpoint_dir=None, use_focal_loss=True):
    """Huấn luyện mô hình với cách tối ưu bộ nhớ."""
    # Reset bộ đếm lỗi
    error_counters.clear()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = focal_loss if use_focal_loss else nn.BCEWithLogitsLoss()
    
    # Tạo DataLoader
    is_streaming = isinstance(train_dataset, StreamingDataset)
    if is_streaming:
        print("Sử dụng StreamingDataset")
        # Với StreamingDataset, không cần shuffle trong DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            collate_fn=collate_fn, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            collate_fn=collate_fn, num_workers=0, pin_memory=False
        )
        # Ước tính tổng số bước
        total_steps = (train_dataset.total_size // batch_size) * epochs // accum_steps
    else:
        print("Sử dụng ChunkDataset")
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            collate_fn=collate_fn, num_workers=0, pin_memory=False
        )
        total_steps = len(train_loader) * epochs // accum_steps
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    scaler = amp.GradScaler()  # Sửa: dùng amp.GradScaler() thay vì torch.cuda.amp.GradScaler()
    
    model.to(device)
    train_losses, val_losses = [], []
    
    # Tạo thư mục checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    early_stopped = False
    
    for epoch in range(epochs):
        print(f"Train Epoch {epoch + 1}/{epochs}:")
        model.train()
        running_loss = 0
        optimizer.zero_grad()
        
        # Tạo DataLoader mới cho mỗi epoch nếu sử dụng StreamingDataset
        if is_streaming:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                collate_fn=collate_fn, num_workers=0, pin_memory=False
            )
        
        # Tạo progress bar
        if is_streaming:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", total=train_dataset.total_size // batch_size)
        else:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        batch_count = 0
        for i, batch in enumerate(pbar):
            # Kiểm tra batch rỗng (khi collate_fn trả về None do lỗi)
            if batch is None:
                continue
            
            try:
                # Đưa dữ liệu lên device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Sử dụng amp.autocast() thay vì torch.autocast()
                with amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    if use_focal_loss:
                        loss = focal_loss(outputs, labels) / accum_steps
                    else:
                        loss = criterion(outputs, labels) / accum_steps
                
                scaler.scale(loss).backward()
                running_loss += loss.item() * accum_steps
                pbar.set_postfix({'loss': loss.item() * accum_steps})
                
                if (i + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                batch_count += 1
                
                # Giải phóng bộ nhớ định kỳ
                if (i + 1) % 10 == 0:
                    del input_ids, attention_mask, labels, outputs, loss
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            except Exception as e:
                error_type = str(type(e).__name__)
                if error_type not in error_counters:
                    error_counters[error_type] = 0
                error_counters[error_type] += 1
                
                if error_counters[error_type] <= MAX_ERRORS_TO_PRINT:
                    print(f"Lỗi khi xử lý batch ({error_type}): {e}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # In tóm tắt số lỗi gặp phải trong epoch
        if error_counters:
            print(f"Tổng số lỗi trong epoch {epoch + 1}:")
            for err_type, count in error_counters.items():
                print(f"  - {err_type}: {count} lỗi")
            # Reset bộ đếm cho epoch mới
            error_counters.clear()
        
        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)
        print(f"Train loss: {avg_loss:.4f}")
        
        # Đánh giá trên validation set
        model.eval()
        val_loss = 0
        val_count = 0
        val_limit = 200  # Giới hạn số batch để đánh giá
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
                # Kiểm tra batch rỗng
                if batch is None:
                    continue
                
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Sử dụng amp.autocast()
                    with amp.autocast():
                        outputs = model(input_ids, attention_mask)
                        if use_focal_loss:
                            loss = focal_loss(outputs, labels)
                        else:
                            loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_count += 1
                    
                    # Giải phóng bộ nhớ sau mỗi batch
                    del input_ids, attention_mask, labels, outputs, loss
                    gc.collect()
                    
                    if i >= val_limit:
                        break
                except Exception as e:
                    error_type = str(type(e).__name__)
                    if error_type not in error_counters:
                        error_counters[error_type] = 0
                    error_counters[error_type] += 1
                    
                    if error_counters[error_type] <= MAX_ERRORS_TO_PRINT:
                        print(f"Lỗi khi validation ({error_type}): {e}")
                    continue
        
        avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
        val_losses.append(avg_val_loss)
        print(f"Val loss: {avg_val_loss:.4f}")
        
        # Lưu checkpoint sau mỗi epoch
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint đã lưu vào {checkpoint_path}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            # Lưu mô hình tốt nhất
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"✓ Val loss improved! Best val loss: {best_val_loss:.4f}")
        else:
            counter += 1
            print(f"✗ No improvement in val loss. Counter: {counter}/{patience}")
            
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                early_stopped = True
                # Khôi phục mô hình tốt nhất
                model.load_state_dict(best_model_state)
                break
        
        # Giải phóng bộ nhớ sau mỗi epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Vẽ đồ thị quá trình học
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('weight/loss_curve.png')
        plt.close()
    except Exception as e:
        print(f"Lỗi khi vẽ đồ thị: {e}")
    
    return model


def compute_rouge(gold, pred):
    """Tính chỉ số ROUGE-1 và ROUGE-2 dựa trên tần suất xuất hiện."""
    gold_tokens = gold.split()
    pred_tokens = pred.split()
    gold_unigrams = Counter(gold_tokens)
    pred_unigrams = Counter(pred_tokens)
    common_unigrams = sum((gold_unigrams & pred_unigrams).values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        rouge1_recall = rouge1_precision = rouge1_f1 = 0.0
    else:
        rouge1_recall = common_unigrams / len(gold_tokens)
        rouge1_precision = common_unigrams / len(pred_tokens)
        rouge1_f1 = 2 * rouge1_precision * rouge1_recall / (rouge1_precision + rouge1_recall) if (rouge1_precision + rouge1_recall) > 0 else 0.0
    
    gold_bigrams = Counter(zip(gold_tokens, gold_tokens[1:]))
    pred_bigrams = Counter(zip(pred_tokens, pred_tokens[1:]))
    common_bigrams = sum((gold_bigrams & pred_bigrams).values())
    if len(gold_bigrams) == 0 or len(pred_bigrams) == 0:
        rouge2_recall = rouge2_precision = rouge2_f1 = 0.0
    else:
        rouge2_recall = common_bigrams / sum(gold_bigrams.values())
        rouge2_precision = common_bigrams / sum(pred_bigrams.values())
        rouge2_f1 = 2 * rouge2_precision * rouge2_recall / (rouge2_precision + rouge2_recall) if (rouge2_precision + rouge2_recall) > 0 else 0.0
    return {'rouge1': rouge1_f1, 'rouge2': rouge2_f1}

def extractive_summarize_from_inputids(input_ids, model, tokenizer, device, max_sent=3):
    """Sinh tóm tắt trích xuất từ input_ids (không cần content gốc)."""
    model.eval()
    with torch.no_grad():
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.tensor(input_ids)
        if len(input_ids.shape) == 3 and input_ids.shape[0] == 1:
            input_ids = input_ids.squeeze(0)
        num_sent, seq_len = input_ids.shape
        attention_mask = (input_ids != 1).long()  # PhoBERT: 1 là <pad>
        encoding = {
            'input_ids': input_ids.unsqueeze(0).to(device),
            'attention_mask': attention_mask.unsqueeze(0).to(device)
        }
        with amp.autocast():
            outputs = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.sigmoid(outputs[0]).cpu().numpy()
        top_idx = np.argsort(probs)[-max_sent:]
        top_idx = sorted(top_idx)
        sentences = [tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(num_sent)]
        selected = [sentences[i] for i in top_idx if sentences[i].strip()]
        return " ".join(selected)

def evaluate_memory_efficient(model, tokenizer, device, test_chunk_files, max_sent=3, max_samples=None):
    """
    Đánh giá mô hình trên test set chỉ chứa input_ids, attention_mask, labels.
    Tính gold summary bằng cách nối các câu có label=1 trong một sample.
    """
    rouge1_scores, rouge2_scores = [], []
    sample_count = 0
    for chunk_file in test_chunk_files:
        try:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            for sample in tqdm(chunk_data, desc=f"Evaluating {chunk_file}"):
                input_ids = sample['input_ids']
                labels = sample['labels']
                # Gold summary: các câu có label=1
                sentences = [tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(len(input_ids))]
                gold = " ".join([sentences[i] for i in range(len(labels)) if (labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]) > 0.5 and sentences[i].strip()])
                pred = extractive_summarize_from_inputids(input_ids, model, tokenizer, device, max_sent=max_sent)
                rouge_scores = compute_rouge(gold, pred)
                rouge1_scores.append(rouge_scores['rouge1'])
                rouge2_scores.append(rouge_scores['rouge2'])
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break
            gc.collect()
            if max_samples is not None and sample_count >= max_samples:
                break
        except Exception as e:
            print(f"Lỗi khi đánh giá file {chunk_file}: {e}")
            continue
    print(f"Đánh giá trên {sample_count} mẫu:")
    print(f"Avg ROUGE-1 F1: {np.mean(rouge1_scores):.4f} ± {np.std(rouge1_scores):.4f}")
    print(f"Avg ROUGE-2 F1: {np.mean(rouge2_scores):.4f} ± {np.std(rouge2_scores):.4f}")
    return {'rouge1': np.mean(rouge1_scores), 'rouge2': np.mean(rouge2_scores)}

def split_existing_cache():
    """Chia nhỏ cache đã có thành các file nhỏ."""
    cache_dir = "/Summary_content/train/folder_cache"
    train_cache = os.path.join(cache_dir, "train.pkl")
    val_cache = os.path.join(cache_dir, "val.pkl")
    test_cache = os.path.join(cache_dir, "test.pkl")
    
    # Tạo thư mục cho chunks
    train_chunks_dir = os.path.join(cache_dir, "train_chunks")
    val_chunks_dir = os.path.join(cache_dir, "val_chunks")
    test_chunks_dir = os.path.join(cache_dir, "test_chunks")
    
    # Chia nhỏ các file cache có sẵn
    train_chunk_files = []
    val_chunk_files = []
    test_chunk_files = []
    
    # Kiểm tra và xử lý từng cache
    if os.path.exists(train_cache):
        if not os.path.exists(train_chunks_dir) or len(os.listdir(train_chunks_dir)) == 0:
            print("Chia nhỏ train cache...")
            train_chunk_files = split_cache_file(train_cache, train_chunks_dir, chunk_size=500)
        else:
            print("Sử dụng train chunks có sẵn...")
            train_chunk_files = [os.path.join(train_chunks_dir, f) for f in sorted(os.listdir(train_chunks_dir)) if f.endswith('.pkl')]
    
    if os.path.exists(val_cache):
        if not os.path.exists(val_chunks_dir) or len(os.listdir(val_chunks_dir)) == 0:
            print("Chia nhỏ validation cache...")
            val_chunk_files = split_cache_file(val_cache, val_chunks_dir, chunk_size=200)
        else:
            print("Sử dụng validation chunks có sẵn...")
            val_chunk_files = [os.path.join(val_chunks_dir, f) for f in sorted(os.listdir(val_chunks_dir)) if f.endswith('.pkl')]
    
    if os.path.exists(test_cache):
        if not os.path.exists(test_chunks_dir) or len(os.listdir(test_chunks_dir)) == 0:
            print("Chia nhỏ test cache...")
            test_chunk_files = split_cache_file(test_cache, test_chunks_dir, chunk_size=200)
        else:
            print("Sử dụng test chunks có sẵn...")
            test_chunk_files = [os.path.join(test_chunks_dir, f) for f in sorted(os.listdir(test_chunks_dir)) if f.endswith('.pkl')]
    
    return train_chunk_files, val_chunk_files, test_chunk_files
