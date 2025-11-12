import gc

import torch
from torch.utils.data import Dataset
import os
import pickle
from tqdm import tqdm
from Summary_content.model_summary_contents.summarization_utils import optimal_sentence_split, generate_soft_label


class VietnameseSummarizationDataset(Dataset):
    def __init__(self, contents, summaries, tokenizer, embed_model, top_n=2, max_content_length=96, cache_file=None):
        self.tokenizer = tokenizer
        self.top_n = top_n
        self.max_content_length = max_content_length
        
        if cache_file and os.path.exists(cache_file):
            print("Tai du lieu tu cache : {}".format(cache_file))
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = []
            
            for content, summary in tqdm(zip(contents, summaries), total=len(contents), desc="Preprocessing"):
                content = str(content)
                summary = str(summary)
                sentences = optimal_sentence_split(content, tokenizer, max_content_length)
                if not sentences:
                    sentences = [content]
                labels, _ = generate_soft_label(sentences, [summary], embed_model, top_n=top_n)
                encoding = tokenizer(sentences, max_length=max_content_length, truncation=True, padding='max_length', return_tensors='pt')
                self.data.append({
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'labels': torch.tensor(labels, dtype=torch.float)
                })
                del encoding, labels, sentences
                gc.collect()
            if cache_file:
                print("Luu tru du lieu vao cache : {}".format(cache_file))
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as file:
                    pickle.dump(self.data, file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem(self, idx):
        return self.data[idx]
