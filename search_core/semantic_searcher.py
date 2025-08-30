# ==============================================================================
# SEMANTIC SEARCHER - PHIÊN BẢN V5 (TỐI ƯU HÓA & NÂNG CẤP TỪ BẢN GỐC)
# ==============================================================================
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Any

from search_core.basic_searcher import BasicSearcher

class SemanticSearcher:
    def __init__(self, 
                 basic_searcher: BasicSearcher, 
                 device: str = "cuda"):
        # --- KHÔNG THAY ĐỔI ---
        print("--- 🧠 Khởi tạo SemanticSearcher (Reranking Engine) ---")
        self.device = device
        self.basic_searcher = basic_searcher
        
        print("   -> Đang tải mô hình Bi-Encoder tiếng Việt...")
        self.semantic_model = SentenceTransformer(
            'bkai-foundation-models/vietnamese-bi-encoder', 
            device=self.device
        )
        print("--- ✅ Tải model Bi-Encoder thành công! ---")

    def search(self, 
            query_text: str, 
            top_k_final: int, 
            top_k_retrieval: int, 
            precomputed_analysis: Dict[str, Any] = None
            ) -> List[Dict[str, Any]]:
        
        # --- LOG THÔNG BÁO GỐC (GIỮ NGUYÊN) ---
        print(f"\n--- Bắt đầu Semantic Search cho context: '{query_text}' ---")
        
        # --- BƯỚC 1 (GIỮ NGUYÊN) ---
        candidates = self.basic_searcher.search(query_text, top_k=top_k_retrieval)
        if not candidates:
            print("-> Không tìm thấy ứng viên nào.")
            return []

        enhanced_query = precomputed_analysis or {}
        
        rerank_keywords_en = enhanced_query.get('objects_en', [])
        rerank_context_vi = enhanced_query.get('search_context', query_text).lower().strip()
        
        # --- BƯỚC 3: TÁI XẾP HẠNG (NÂNG CẤP) ---

        # --- 3a. Mã hóa Context và Bối cảnh Keyframe (theo batch) ---
        context_vector = self.semantic_model.encode(rerank_context_vi, convert_to_tensor=True, device=self.device)
        
        # *** NÂNG CẤP 1: TẠO BỐI CẢNH "SẠCH" TỪ CÁC CỘT CHUYÊN BIỆT ***
        candidate_contexts = [
            (cand.get('transcript_text', '') + " " + cand.get('object_text', '')).strip() 
            for cand in candidates
        ]
        
        candidate_vectors = self.semantic_model.encode(
            candidate_contexts, # <-- Sử dụng context đã được làm sạch
            convert_to_tensor=True, 
            device=self.device, 
            batch_size=128, 
            show_progress_bar=False
        )
        semantic_scores_tensor = util.pytorch_cos_sim(context_vector, candidate_vectors)[0]

        # --- 3b. Mã hóa Object (theo batch) (GIỮ NGUYÊN TOÀN BỘ LOGIC GỐC) ---
        all_detected_objects = []
        cand_object_indices = []
        for cand in candidates:
            detected_objects_en_raw = cand.get('objects_detected', [])
            detected_objects_en = list(detected_objects_en_raw) if isinstance(detected_objects_en_raw, np.ndarray) else list(detected_objects_en_raw)
            start_index = len(all_detected_objects)
            all_detected_objects.extend(detected_objects_en)
            end_index = len(all_detected_objects)
            cand_object_indices.append((start_index, end_index))

        all_object_vectors = None
        if all_detected_objects:
            all_object_vectors = self.semantic_model.encode(
                all_detected_objects, 
                convert_to_tensor=True, 
                device=self.device, 
                batch_size=256, 
                show_progress_bar=False
            )

        query_object_vectors = None
        if rerank_keywords_en:
            query_object_vectors = self.semantic_model.encode(
                rerank_keywords_en, convert_to_tensor=True, device=self.device)

        # --- 3c. Tính toán điểm cuối cùng (GIỮ NGUYÊN TOÀN BỘ LOGIC GỐC) ---
        reranked_results = []
        for i, cand in enumerate(candidates):
            # Tính Object Score
            object_score = 0.0
            start, end = cand_object_indices[i]
            if query_object_vectors is not None and start < end and all_object_vectors is not None:
                detected_object_vectors = all_object_vectors[start:end]
                cosine_scores = util.pytorch_cos_sim(query_object_vectors, detected_object_vectors)
                if cosine_scores.numel() > 0:
                    max_scores_per_query_obj = torch.max(cosine_scores, dim=1).values
                    object_score = torch.mean(max_scores_per_query_obj).item()

            # Tính Semantic Score
            semantic_score = (semantic_scores_tensor[i].item() + 1) / 2
            
            original_clip_score = cand['clip_score'] # Nằm trong khoảng [-1, 1]
            # Chuẩn hóa điểm CLIP về [0, 1] để cộng hưởng tốt hơn
            normalized_clip_score = (original_clip_score + 1) / 2
            
            # Tính Final Score
            w_clip = precomputed_analysis.get('w_clip', 0.4)
            w_obj = precomputed_analysis.get('w_obj', 0.3)
            w_semantic = precomputed_analysis.get('w_semantic', 0.3)
            
            final_score = (w_clip * normalized_clip_score + 
                           w_obj * object_score + 
                           w_semantic * semantic_score)
            
            cand['final_score'] = final_score
            cand['scores'] = {'clip': normalized_clip_score, 'object': object_score, 'semantic': semantic_score}
            reranked_results.append(cand)

        # --- SẮP XẾP VÀ TRẢ VỀ (GIỮ NGUYÊN) ---
        reranked_results = sorted(reranked_results, key=lambda x: x['final_score'], reverse=True)
        
        # --- LOG THÔNG BÁO GỐC (GIỮ NGUYÊN) ---
        print(f"--- ✅ Tái xếp hạng cho context '{query_text}' hoàn tất! ---")
        
        return reranked_results[:top_k_final]