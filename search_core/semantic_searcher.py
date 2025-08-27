import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Any

# Import BasicSearcher để sử dụng làm nền tảng
from search_core.basic_searcher import BasicSearcher

# Import thư viện Gemini và decorator retrier
import google.generativeai as genai

class SemanticSearcher:
    # --- THAY ĐỔI __init__ ---
    def __init__(self, 
                 basic_searcher: BasicSearcher, 
                 device: str = "cuda"):
        """
        Khởi tạo SemanticSearcher.
        Nó không còn quản lý model AI nữa.
        """
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
            top_k_final: int = 12, 
            top_k_retrieval: int = 100, 
            precomputed_analysis: Optional[Dict] = None
            ) -> List[Dict[str, Any]]:
        """
        Thực hiện pipeline tìm kiếm ngữ nghĩa hoàn chỉnh cho một context.
        *** PHIÊN BẢN TỐI ƯU HÓA: Tắt progress bar và mã hóa object theo batch ***

        Args:
            query_text (str): Bối cảnh tìm kiếm (search_context).
            top_k_final (int): Số kết quả cuối cùng trả về.
            top_k_retrieval (int): Số ứng viên ban đầu được lấy từ FAISS.
            precomputed_analysis (Optional[Dict]): Kết quả phân tích Gemini đã được tính toán trước.

        Returns:
            List[Dict[str, Any]]: Danh sách các keyframe kết quả đã được xếp hạng.
        """
        print(f"\n--- Bắt đầu Semantic Search cho context: '{query_text}' ---")
        
        # --- Bước 1: Truy xuất ứng viên bằng CLIP ---
        candidates = self.basic_searcher.search(query_text, top_k=top_k_retrieval)
        if not candidates:
            print("-> Không tìm thấy ứng viên nào.")
            return []

        # --- Bước 2: Lấy thông tin Phân tích & Tăng cường ---
        if precomputed_analysis:
            enhanced_query = precomputed_analysis
        else:
            enhanced_query = self.enhance_query_with_gemini(query_text)
        
        rerank_keywords_en = enhanced_query.get('objects_en', [])
        rerank_context_vi = enhanced_query.get('search_context', query_text).lower().strip()
        
        # --- Bước 3: Tái xếp hạng (Reranking) ---

        # --- 3a. Mã hóa Context và Transcript (theo batch) ---
        context_vector = self.semantic_model.encode(rerank_context_vi, convert_to_tensor=True, device=self.device)
        candidate_texts = [cand.get('searchable_text', '') for cand in candidates]
        candidate_vectors = self.semantic_model.encode(
            candidate_texts, 
            convert_to_tensor=True, 
            device=self.device, 
            batch_size=128, 
            show_progress_bar=False # TẮT THANH TIẾN TRÌNH
        )
        semantic_scores_tensor = util.pytorch_cos_sim(context_vector, candidate_vectors)[0]

        # --- 3b. Mã hóa Object (tối ưu hóa theo batch) ---
        # Gom tất cả các object từ tất cả các ứng viên lại
        all_detected_objects = []
        cand_object_indices = [] # Lưu lại index (start, end) để map kết quả về
        for cand in candidates:
            detected_objects_en_raw = cand.get('objects_detected', [])
            # Chuyển đổi an toàn sang list
            detected_objects_en = list(detected_objects_en_raw) if isinstance(detected_objects_en_raw, np.ndarray) else list(detected_objects_en_raw)
            
            start_index = len(all_detected_objects)
            all_detected_objects.extend(detected_objects_en)
            end_index = len(all_detected_objects)
            cand_object_indices.append((start_index, end_index))

        # Mã hóa tất cả các object trong một lần duy nhất
        all_object_vectors = None
        if all_detected_objects:
            all_object_vectors = self.semantic_model.encode(
                all_detected_objects, 
                convert_to_tensor=True, 
                device=self.device, 
                batch_size=256, 
                show_progress_bar=False # TẮT THANH TIẾN TRÌNH
            )

        # Mã hóa object của truy vấn một lần
        query_object_vectors = None
        if rerank_keywords_en:
            query_object_vectors = self.semantic_model.encode(
                rerank_keywords_en, convert_to_tensor=True, device=self.device)

        # --- 3c. Tính toán điểm cuối cùng cho mỗi ứng viên ---
        reranked_results = []
        for i, cand in enumerate(candidates):
            # Tính Object Score
            object_score = 0.0
            start, end = cand_object_indices[i]
            
            # Lấy các vector đã được tính toán trước, không cần encode lại
            if query_object_vectors is not None and start < end and all_object_vectors is not None:
                detected_object_vectors = all_object_vectors[start:end]
                cosine_scores = util.pytorch_cos_sim(query_object_vectors, detected_object_vectors)
                if cosine_scores.numel() > 0:
                    max_scores_per_query_obj = torch.max(cosine_scores, dim=1).values
                    object_score = torch.mean(max_scores_per_query_obj).item()

            # Tính Semantic Score
            semantic_score = (semantic_scores_tensor[i].item() + 1) / 2
            
            # Tính Final Score
            w_clip, w_obj, w_semantic = 0.4, 0.3, 0.3
            normalized_clip_score = cand['clip_score']
            
            final_score = (w_clip * normalized_clip_score + 
                        w_obj * object_score + 
                        w_semantic * semantic_score)
            
            cand['final_score'] = final_score
            cand['scores'] = {'clip': normalized_clip_score, 'object': object_score, 'semantic': semantic_score}
            reranked_results.append(cand)

        # Sắp xếp lại dựa trên điểm cuối cùng và trả về
        reranked_results = sorted(reranked_results, key=lambda x: x['final_score'], reverse=True)
        print(f"--- ✅ Tái xếp hạng cho context '{query_text}' hoàn tất! ---")
        
        return reranked_results[:top_k_final]