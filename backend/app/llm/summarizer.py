import os, time, re, torch, numpy as np, torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from huggingface_hub import snapshot_download     
from underthesea import sent_tokenize
from scripts.fetch_model import ensure_model

MODEL_PATH = ensure_model()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".hf_cache")  
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  

MODEL_PATH   = os.path.join(BASE_DIR, "models", "phobertsum_lightweight.pt")
PHOBERT_NAME = "vinai/phobert-base"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

class LightweightPhoBERTSUM(nn.Module):
    def __init__(self, phobert_model_name=PHOBERT_NAME, hidden_size=768, dropout=0.1, max_sent=96):
        super().__init__()
        config = AutoConfig.from_pretrained(phobert_model_name)
        config.num_hidden_layers = 6
        self.encoder = AutoModel.from_pretrained(phobert_model_name, config=config)
        small_hidden_size = 384
        self.dim_reduction = nn.Linear(hidden_size, small_hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=small_hidden_size, nhead=6, dropout=dropout, batch_first=True, dim_feedforward=1024
        )
        self.inter_sentence_transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.layer_norm = nn.LayerNorm(small_hidden_size)
        self.classifier = nn.Linear(small_hidden_size, 1)
        self.register_buffer("pos_encoder", self._pos(max_sent, small_hidden_size))

    def _pos(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids, attention_mask):
        b, n, L = input_ids.size()
        input_ids = input_ids.view(-1, L)
        attention_mask = attention_mask.view(-1, L)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        cls_vec = out.view(b, n, -1)
        cls_vec = self.dim_reduction(cls_vec)
        sent_mask = (attention_mask.view(b, n, -1).sum(dim=2) == 0)
        if n <= self.pos_encoder.size(1):
            cls_vec = cls_vec + self.pos_encoder[:, :n, :]
        trans_out = self.inter_sentence_transformer(cls_vec, src_key_padding_mask=sent_mask)
        trans_out = self.layer_norm(trans_out + cls_vec)
        logits = self.classifier(trans_out)
        return logits.squeeze(-1)

_model, _tokenizer = None, None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    local_repo = snapshot_download(
        repo_id="vinai/phobert-base",
        cache_dir=CACHE_DIR,
        local_files_only=False, 
        allow_patterns=[
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "vocab.txt",
            "bpe.codes",
            "tokenization_*",
            "special_tokens_map.json",
        ],
    )

    _tokenizer = AutoTokenizer.from_pretrained(local_repo, use_fast=True, local_files_only=True)
    _ = AutoConfig.from_pretrained(local_repo, local_files_only=True)  # nếu bạn cần config

    model = LightweightPhoBERTSUM(phobert_model_name=local_repo)
    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)

    _model = model
    return _model, _tokenizer


def _split_sent(content, tokenizer, max_length=96):
    sents = sent_tokenize(content) or [content]
    parts = []
    for s in sents:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if len(toks) <= max_length:
            parts.append(s); continue
        subs = [x.strip() for x in re.split(r'[;,]', s) if x.strip()] or [s]
        for sub in subs:
            tsub = tokenizer.encode(sub, add_special_tokens=False)
            if len(tsub) <= max_length:
                parts.append(sub)
            else:
                step = max(1, max_length // 2)
                for i in range(0, len(tsub), step):
                    chunk = tsub[i:i+max_length]
                    parts.append(tokenizer.decode(chunk).strip())
    return [p for p in parts if p] or [content]

def summarize(text: str, max_sent=3, max_length=96):
    model, tok = load_model()
    t0 = time.time()
    sents = _split_sent(text, tok, max_length)
    if len(sents) <= max_sent:
        return " ".join(sents), (time.time() - t0) * 1000
    sents = sents[:30]
    enc = tok(sents, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    ids, mask = enc['input_ids'].unsqueeze(0).to(DEVICE), enc['attention_mask'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(ids, mask)[0]
        probs = torch.sigmoid(logits).cpu().numpy()
    top = sorted(np.argsort(probs)[-max_sent:])
    out = " ".join([sents[i] for i in top if sents[i].strip()])
    return out, (time.time() - t0) * 1000
