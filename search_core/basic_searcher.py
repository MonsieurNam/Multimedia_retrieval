import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
class BasicSearcher:
    def __init__(self, index_path, metadata_path, video_path_map, clip_features_path, device="cuda"): 
        self.device = device
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_parquet(metadata_path)
        if 'index' not in self.metadata.columns:
            self.metadata['index'] = self.metadata.index
        self.metadata['video_path'] = self.metadata['video_id'].map(video_path_map)
        self.model = SentenceTransformer('clip-ViT-B-32-multilingual-v1', device=self.device)
        self.clip_features_numpy = np.load(clip_features_path)

    def get_all_clip_features(self) -> np.ndarray: # <-- PHƯƠNG THỨC MỚI
        """Trả về ma trận NumPy của tất cả các vector CLIP."""
        return self.clip_features_numpy
    
    def search(self, query_text: str, top_k: int = 10):
        query_vector = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vector.astype('float32'))
        distances, indices = self.index.search(query_vector.astype('float32'), k=top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                info = self.metadata.iloc[idx].to_dict()
                info['clip_score'] = 1 - dist
                results.append(info)
        return results
