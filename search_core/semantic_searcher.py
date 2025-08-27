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
from .basic_searcher import BasicSearcher

# Import thư viện Gemini
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class SemanticSearcher:
    """
    Class thực hiện tìm kiếm ngữ nghĩa nâng cao.

    Bao gồm các bước:
    1.  Truy xuất ứng viên ban đầu bằng CLIP đa ngôn ngữ (thông qua BasicSearcher).
    2.  Tăng cường truy vấn bằng Gemini để trích xuất thực thể và bối cảnh.
    3.  Tái xếp hạng (rerank) các ứng viên dựa trên một công thức điểm kết hợp:
        - Điểm tương đồng CLIP (hình ảnh).
        - Điểm tương đồng ngữ nghĩa của đối tượng (object).
        - Điểm tương đồng ngữ nghĩa của bối cảnh (context/transcript).
    """

    def __init__(self, 
                 basic_searcher: BasicSearcher, 
                 gemini_model: Optional[genai.GenerativeModel] = None, 
                 device: str = "cuda"):
        """
        Khởi tạo SemanticSearcher.

        Args:
            basic_searcher (BasicSearcher): Một instance của BasicSearcher đã được khởi tạo.
            gemini_model (Optional[genai.GenerativeModel]): Instance của Gemini model để chia sẻ.
            device (str): Thiết bị để chạy model ('cuda' hoặc 'cpu').
        """
        print("--- 🧠 Khởi tạo SemanticSearcher (Phiên bản Nâng cao) ---")
        self.device = device
        self.basic_searcher = basic_searcher
        self.gemini_model = gemini_model

        print("   -> Đang tải mô hình Bi-Encoder tiếng Việt ('bkai-foundation-models/vietnamese-bi-encoder')...")
        self.semantic_model = SentenceTransformer(
            'bkai-foundation-models/vietnamese-bi-encoder', 
            device=self.device
        )
        print("--- ✅ Tải model Bi-Encoder thành công! ---")
        
    def enhance_query_with_gemini(self, query: str) -> Dict[str, Any]:
        """
        Sử dụng Gemini để phân tích, tăng cường và dịch truy vấn của người dùng.

        Args:
            query (str): Câu truy vấn gốc bằng tiếng Việt.

        Returns:
            Dict[str, Any]: Một dictionary chứa 'objects_vi', 'objects_en', và 'context_vi'.
        """
        fallback_result = {'objects_vi': query.split(), 'objects_en': query.split(), 'context_vi': query}
        
        if not self.gemini_model:
            print("--- ⚠️ Gemini model chưa được khởi tạo. Sử dụng fallback cho enhance_query. ---")
            return fallback_result

        prompt = f"""
        You are a helpful assistant for a Vietnamese video search system. Your task is to analyze a Vietnamese user query and extract key information.
        Return ONLY a single, valid JSON object with three keys: "objects_vi", "objects_en", and "context_vi".
        - "objects_vi": A list of important nouns and entities from the query in Vietnamese.
        - "objects_en": The direct English translation for EACH item in "objects_vi". The two lists must have the same length.
        - "context_vi": A simple sentence in Vietnamese that describes the main action or context of the query.

        **Example 1:**
        Query: "cô gái mặc váy vàng đi dạo trong công viên gần bờ hồ"
        JSON: {{"objects_vi": ["cô gái", "váy vàng", "công viên", "bờ hồ"], "objects_en": ["girl", "yellow dress", "park", "lakeshore"], "context_vi": "một cô gái đang đi dạo trong công viên"}}

        **Example 2:**
        Query: "xe cứu hỏa phun nước chữa cháy tòa nhà"
        JSON: {{"objects_vi": ["xe cứu hỏa", "nước", "tòa nhà"], "objects_en": ["fire truck", "water", "building"], "context_vi": "một chiếc xe cứu hỏa đang chữa cháy"}}

        **Query to process:** "{query}"
        **JSON:**
        """
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        try:
            response = self.gemini_model.generate_content(prompt, safety_settings=safety_settings)
            match = re.search(r"\{.*\}", response.text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in Gemini response.")
            
            result = json.loads(match.group(0))
            
            if all(k in result for k in ['objects_vi', 'objects_en', 'context_vi']) and \
               isinstance(result['objects_vi'], list) and \
               isinstance(result['objects_en'], list) and \
               len(result['objects_vi']) == len(result['objects_en']):
                return result

            print("--- ⚠️ JSON từ Gemini không hợp lệ. Sử dụng fallback. ---")
            return fallback_result
        except Exception as e:
            print(f"--- ⚠️ Lỗi khi gọi API Gemini để tăng cường truy vấn: {e}. Sử dụng fallback. ---")
            return fallback_result

    def search(self, 
               query_text: str, 
               top_k_final: int = 12, 
               top_k_retrieval: int = 100, 
               precomputed_analysis: Optional[Dict] = None
              ) -> List[Dict[str, Any]]:
        """
        Thực hiện pipeline tìm kiếm ngữ nghĩa hoàn chỉnh.

        Args:
            query_text (str): Câu truy vấn của người dùng.
            top_k_final (int): Số kết quả cuối cùng trả về.
            top_k_retrieval (int): Số ứng viên ban đầu được lấy từ FAISS.
            precomputed_analysis (Optional[Dict]): Kết quả phân tích Gemini đã được tính toán trước.

        Returns:
            List[Dict[str, Any]]: Danh sách các keyframe kết quả đã được xếp hạng.
        """
        print(f"\n--- Bắt đầu Semantic Search cho: '{query_text}' ---")
        
        # --- Bước 1: Truy xuất ứng viên bằng CLIP ---
        print(f"1. Truy xuất Top {top_k_retrieval} ứng viên bằng CLIP...")
        candidates = self.basic_searcher.search(query_text, top_k=top_k_retrieval)
        if not candidates:
            print("-> Không tìm thấy ứng viên nào.")
            return []
        print(f"-> Tìm thấy {len(candidates)} ứng viên.")

        # --- Bước 2: Phân tích & Tăng cường truy vấn ---
        if precomputed_analysis:
            print("\n2. Sử dụng kết quả phân tích truy vấn đã có...")
            enhanced_query = precomputed_analysis
        else:
            print("\n2. Tăng cường và Dịch truy vấn bằng Gemini...")
            enhanced_query = self.enhance_query_with_gemini(query_text)
        
        rerank_keywords_en = enhanced_query.get('objects_en', [])
        rerank_context_vi = enhanced_query.get('context_vi', '').lower().strip()
        
        print(f" -> English Keywords for Reranking: {rerank_keywords_en}")
        print(f" -> Vietnamese Context for Reranking: '{rerank_context_vi}'")

        # --- Bước 3: Tái xếp hạng (Reranking) ---
        print("\n3. Bắt đầu tái xếp hạng các ứng viên...")
        
        # Mã hóa ngữ cảnh truy vấn một lần
        context_vector = self.semantic_model.encode(rerank_context_vi, convert_to_tensor=True, device=self.device)
        
        # Mã hóa các văn bản của ứng viên theo batch để tăng tốc
        candidate_texts = [cand.get('searchable_text', '') for cand in candidates]
        candidate_vectors = self.semantic_model.encode(
            candidate_texts, 
            convert_to_tensor=True, 
            device=self.device, 
            batch_size=128, 
            show_progress_bar=True
        )
        # Tính điểm tương đồng ngữ nghĩa cho tất cả ứng viên cùng lúc
        semantic_scores_tensor = util.pytorch_cos_sim(context_vector, candidate_vectors)[0]
        
        # Mã hóa các từ khóa object trong truy vấn một lần
        query_object_vectors = None
        if rerank_keywords_en:
            query_object_vectors = self.semantic_model.encode(
                rerank_keywords_en, 
                convert_to_tensor=True, 
                device=self.device
            )

        reranked_results = []
        for i, cand in enumerate(tqdm(candidates, desc="Calculating Final Scores")):
            # --- Tính Object Score ---
            object_score = 0.0
            
            # Lấy danh sách object, đảm bảo nó là list
            detected_objects_en_raw = cand.get('objects_detected', [])
            
            # Chuyển đổi an toàn từ NumPy array (nếu có) sang list
            if isinstance(detected_objects_en_raw, np.ndarray):
                detected_objects_en = detected_objects_en_raw.tolist()
            else:
                # Nếu nó đã là list hoặc kiểu dữ liệu khác, chuyển nó thành list
                detected_objects_en = list(detected_objects_en_raw)

            # --- DÒNG CẦN SỬA ĐÂY ---
            # Sửa từ: `if query_object_vectors is not None and detected_objects_en and len(detected_objects_en) > 0:`
            # Thành: `if query_object_vectors is not None and len(detected_objects_en) > 0:`
            if query_object_vectors is not None and len(detected_objects_en) > 0:
                detected_object_vectors = self.semantic_model.encode(
                    detected_objects_en, 
                    convert_to_tensor=True, 
                    device=self.device
                )
                
                # So sánh mỗi object truy vấn với tất cả object được phát hiện
                cosine_scores = util.pytorch_cos_sim(query_object_vectors, detected_object_vectors)
                
                # Kiểm tra để đảm bảo cosine_scores không rỗng trước khi thực hiện max()
                if cosine_scores.numel() > 0:
                    # Với mỗi object truy vấn, tìm điểm tương đồng cao nhất
                    max_scores_per_query_obj = torch.max(cosine_scores, dim=1).values
                    # Điểm object là trung bình của các điểm cao nhất này
                    object_score = torch.mean(max_scores_per_query_obj).item()

            # --- Tính Semantic Score ---
            # Chuẩn hóa điểm cosine similarity (từ [-1, 1]) về thang [0, 1]
            semantic_score = (semantic_scores_tensor[i].item() + 1) / 2
            
            # --- Tính Final Score ---
            w_clip, w_obj, w_semantic = 0.4, 0.3, 0.3
            # Giả sử clip_score từ BasicSearcher đã được chuẩn hóa trong thang [0,1]
            normalized_clip_score = cand['clip_score']
            
            final_score = (w_clip * normalized_clip_score + 
                           w_obj * object_score + 
                           w_semantic * semantic_score)
            
            cand['final_score'] = final_score
            cand['scores'] = {'clip': normalized_clip_score, 'object': object_score, 'semantic': semantic_score}
            reranked_results.append(cand)

        # Sắp xếp lại dựa trên điểm cuối cùng và trả về
        reranked_results = sorted(reranked_results, key=lambda x: x['final_score'], reverse=True)
        print("--- ✅ Tái xếp hạng hoàn tất! ---")
        
        return reranked_results[:top_k_final]