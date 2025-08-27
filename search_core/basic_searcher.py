import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

class BasicSearcher:
    def __init__(self, index_path, metadata_path, video_path_map, device="cuda"):
        self.device = device
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_parquet(metadata_path)
        self.metadata['video_path'] = self.metadata['video_id'].map(video_path_map)
        self.model = SentenceTransformer('clip-ViT-B-32-multilingual-v1', device=self.device)
    
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
