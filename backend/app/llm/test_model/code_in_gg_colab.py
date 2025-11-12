import torch
import torch.nn as nn
import numpy as np
import re
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from underthesea import sent_tokenize
from google.colab import drive
import nest_asyncio


nest_asyncio.apply()
from fastapi import FastAPI
from pyngrok import ngrok
import threading
import uvicorn
import asyncio


print("Đã cài đặt xong thư viện!")

# BƯỚC 2: Mount Google Drive và tìm model
print("\nBƯỚC 2: Kết nối Google Drive...")
try:
    drive.mount('/content/drive')
    print("Đã kết nối Google Drive!")
    
    # Tìm file model
    print("\nĐang tìm file model (.pt) trong Drive...")
    drive_path = '/content/drive/MyDrive/Train_LLM_Project/weight'
    model_files = []
    
    for root, dirs, files in os.walk(drive_path):
        for file in files:
            if file.endswith('.pt'):
                full_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(full_path) / (1024 * 1024)
                    model_files.append((full_path, file_size))
                except:
                    continue
    
    if model_files:
        print(f"Tìm thấy {len(model_files)} file model:")
        print("-" * 50)
        for i, (path, size) in enumerate(model_files, 1):
            print(f"{i}. {path}")
            print(f"{size:.1f} MB")
        print("-" * 50)
        default_model = model_files[0][0]
        print(f"Sử dụng model mặc định: {default_model}")
    else:
        default_model = None
        print("Không tìm thấy file .pt nào trong Drive")
        print("Hãy đảm bảo đã upload model lên Google Drive")

except Exception as e:
    default_model = None
    print(f"Lỗi kết nối Drive: {e}")

# BƯỚC 3: Định nghĩa model và các hàm cần thiết
print(f"\nBƯỚC 3: Chuẩn bị mô hình...")

# Model Definition
class LightweightPhoBERTSUM(nn.Module):
    def __init__(self, phobert_model_name="vinai/phobert-base", hidden_size=768, dropout=0.1, max_sent=96):
        super().__init__()
        config = AutoConfig.from_pretrained(phobert_model_name)
        config.num_hidden_layers = 6
        self.encoder = AutoModel.from_pretrained(phobert_model_name, config=config)
        small_hidden_size = 384
        self.dim_reduction = nn.Linear(hidden_size, small_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=small_hidden_size,
            nhead=6,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=1024
        )
        self.inter_sentence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.layer_norm = nn.LayerNorm(small_hidden_size)
        self.classifier = nn.Linear(small_hidden_size, 1)
        self.register_buffer("pos_encoder", self._create_positional_encoding(max_sent, small_hidden_size))
    
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, input_ids, attention_mask):
        batch_size, num_sent, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]
        cls_vec = cls_vec.view(batch_size, num_sent, -1)
        cls_vec = self.dim_reduction(cls_vec)
        sent_attention = attention_mask.view(batch_size, num_sent, -1)
        sent_mask = (sent_attention.sum(dim=2) == 0)
        if num_sent <= self.pos_encoder.size(1):
            cls_vec = cls_vec + self.pos_encoder[:, :num_sent, :]
        trans_out = self.inter_sentence_transformer(
            cls_vec,
            src_key_padding_mask=sent_mask
        )
        trans_out = trans_out + cls_vec
        trans_out = self.layer_norm(trans_out)
        logits = self.classifier(trans_out)
        return logits.squeeze(-1)

def optimal_sentence_split(content, tokenizer, max_length=96):
    sentences = sent_tokenize(content)
    new_sentences = []
    for sent in sentences:
        tokens = tokenizer.encode(sent, add_special_tokens=False)
        if len(tokens) <= max_length:
            new_sentences.append(sent)
        else:
            sub_sents = re.split(r'[;,]', sent)
            for sub in sub_sents:
                sub = sub.strip()
                if not sub:
                    continue
                tokens_sub = tokenizer.encode(sub, add_special_tokens=False)
                if len(tokens_sub) <= max_length:
                    new_sentences.append(sub)
                else:
                    for i in range(0, len(tokens_sub), max_length // 2):
                        chunk = tokens_sub[i:i + max_length]
                        chunk_text = tokenizer.decode(chunk)
                        new_sentences.append(chunk_text.strip())
    result = [s for s in new_sentences if s.strip()]
    return result if result else [content]

def extractive_summarize(text, model, tokenizer, device, max_sent=3, max_length=96):
    model.eval()
    with torch.no_grad():
        sentences = optimal_sentence_split(text, tokenizer, max_length=max_length)
        if not sentences:
            sentences = [text]
        if len(sentences) <= max_sent:
            return " ".join(sentences)
        
        max_sentences = 30
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        encoding = tokenizer(
            sentences, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].unsqueeze(0).to(device)
        attention_mask = encoding['attention_mask'].unsqueeze(0).to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs[0]).cpu().numpy()
        top_idx = np.argsort(probs)[-max_sent:]
        top_idx = sorted(top_idx)
        selected = [sentences[i] for i in top_idx if sentences[i].strip()]
        return " ".join(selected)

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
        model = LightweightPhoBERTSUM()
        
        # Sửa: Xử lý tương thích với các phiên bản PyTorch
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except TypeError:
            # Nếu weights_only không khả dụng trong phiên bản PyTorch
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Lỗi khi load model: {str(e)}")
        return None, None, None

if default_model:
    print(f"Đang tải model từ {default_model}...")
    model, tokenizer, device = load_model(default_model)
    if model:
        print(f"Model đã được tải thành công!")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Đang chạy trên CPU")
    else:
        model, tokenizer, device = None, None, None
        print("Không thể tải model!")
else:
    model, tokenizer, device = None, None, None
    print("Không có model mặc định, bạn sẽ cần nhập đường dẫn model.")

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    text = data.get("text", "")
    
    if not model:
        return {"error": "Model is not loaded"}
    
    summary = extractive_summarize(text, model, tokenizer, device, max_sent=3)
    return {"summary": summary}


public_url = ngrok.connect(8000).public_url
print("Public API URL:", public_url)


def run_app():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

thread = threading.Thread(target=run_app)
thread.daemon = True
thread.start()
