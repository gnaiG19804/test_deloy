import gc
import math
import pickle
import random

import torch.utils.data
from torch.utils.data import Dataset, DataLoader, IterableDataset


# Dataset de stream du lieu tu nhieu file theo trinh tu, tranh tran RAM
class StreamingDataset(IterableDataset):
    def __init__(self, chunk_files, shuffle=True):
        self.chunk_files = chunk_files
        self.shuffle = shuffle
        
        self.total_size = 0
        for file in chunk_files:
            try:
                with open(file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    self.total_size += len(chunk_data)
                    del chunk_data
            except Exception as e:
                print(f"Lỗi khi đọc file {file}: {e}")
        print(f"StreamingDataset: {len(chunk_files)} chunk files, ước tính {self.total_size} mẫu")
    
    def __iter__(self):
        # Trộn thứ tự các file nếu cần
        worker_info = torch.utils.data.get_worker_info()
        chunk_files = self.chunk_files
        if self.shuffle:
            random.shuffle(chunk_files)
        
        if worker_info is not None:
            per_worker = int(math.ceil(len(chunk_files) / worker_info.num_workers))
            worker_id = worker_info.id
            chunk_files = chunk_files[worker_id * per_worker: (worker_id + 1) * per_worker]
        for file in chunk_files:
            try:
                with open(file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    # Trộn dữ liệu trong chunk nếu cần
                    if self.shuffle:
                        indices = list(range(len(chunk_data)))
                        random.shuffle(indices)
                        samples = [chunk_data[i] for i in indices]
                    else:
                        samples = chunk_data
                    for sample in samples:
                        yield sample
                    
                    del chunk_data, samples
                    gc.collect()
            except Exception as e:
                print(f"Lỗi khi đọc file {file}: {e}")
                continue
