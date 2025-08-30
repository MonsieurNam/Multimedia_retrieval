from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import util
import faiss 

class MMRResultBuilder:
    """
    Xây dựng lại danh sách kết quả cuối cùng bằng thuật toán Maximal Marginal Relevance (MMR)
    để tăng cường sự đa dạng.
    """
    def __init__(self, clip_features: np.ndarray, device: str = "cuda"):
        """
        Khởi tạo MMRResultBuilder.

        Args:
            clip_features (np.ndarray): Ma trận NumPy chứa tất cả các vector CLIP đã được nạp sẵn.
            device (str): Thiết bị để chạy tính toán (cuda hoặc cpu).
        """
        print("--- 🎨 Khởi tạo MMR Result Builder (Diversity Engine) ---")
        self.device = device
        try:
            print(f"   -> Đang chuyển ma trận vector CLIP sang tensor trên {self.device}...")
            
            # --- BƯỚC SỬA LỖI ---
            # 1. Đảm bảo ma trận là C-contiguous và có kiểu float32
            features_copy = np.ascontiguousarray(clip_features.astype('float32'))
            
            # 2. Chuẩn hóa L2 trên NumPy float32
            faiss.normalize_L2(features_copy)
            
            # 3. Chuyển sang tensor
            self.clip_features_tensor = torch.from_numpy(features_copy).to(self.device)

            print(f"--- ✅ Chuyển đổi thành công {self.clip_features_tensor.shape[0]} vector CLIP. ---")
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi xử lý vector CLIP: {e}. MMR sẽ bị vô hiệu hóa. ---")
            # In ra traceback để debug dễ hơn
            import traceback
            traceback.print_exc()
            self.clip_features_tensor = None
            
    def _calculate_similarity(self, cand_A: Dict, cand_B: Dict, w_visual: float = 0.8, w_time: float = 0.2) -> float:
        """
        Tính toán độ tương đồng kết hợp giữa hai ứng viên.
        """
        if self.clip_features_tensor is None:
            return 0.0

        # --- 1. Visual Similarity ---
        idx_A = cand_A['original_index'] # Cần thêm 'original_index' vào metadata
        idx_B = cand_B['original_index']
        vec_A = self.clip_features_tensor[idx_A]
        vec_B = self.clip_features_tensor[idx_B]
        visual_sim = util.pytorch_cos_sim(vec_A, vec_B).item()
        
        # --- 2. Temporal Similarity ---
        temporal_sim = 0.0
        if cand_A['video_id'] == cand_B['video_id']:
            time_diff = abs(cand_A['timestamp'] - cand_B['timestamp'])
            # Dùng hàm decay, ví dụ: điểm giảm một nửa sau mỗi 10 giây
            temporal_sim = np.exp(-0.0693 * time_diff) # -ln(0.5)/10 ≈ 0.0693

        # --- 3. Kết hợp ---
        combined_sim = (w_visual * visual_sim) + (w_time * temporal_sim)
        return combined_sim

    def build_diverse_list(self, 
                           candidates: List[Dict], 
                           target_size: int = 100, 
                           lambda_val: float = 0.7
                          ) -> List[Dict]:
        """
        Xây dựng danh sách kết quả đa dạng bằng thuật toán MMR.
        """
        if not candidates or self.clip_features_tensor is None:
            return candidates[:target_size]

        print(f"--- Bắt đầu xây dựng danh sách đa dạng bằng MMR (λ={lambda_val}) ---")
        
        # Chuyển đổi candidates thành một dictionary để truy cập nhanh
        candidates_pool = {i: cand for i, cand in enumerate(candidates)}
        # Thêm 'original_index' vào mỗi candidate để truy xuất vector CLIP
        for i, cand in enumerate(candidates):
             # Giả định metadata đã có cột 'index' là vị trí của nó trong file parquet gốc
            candidates_pool[i]['original_index'] = cand.get('index')

        final_results_indices = []
        
        # --- Bước khởi tạo: Chọn ứng viên tốt nhất đầu tiên ---
        if not candidates_pool: return []
        
        best_initial_idx = max(candidates_pool, key=lambda idx: candidates_pool[idx]['final_score'])
        final_results_indices.append(best_initial_idx)
        
        # --- Vòng lặp MMR ---
        while len(final_results_indices) < min(target_size, len(candidates)):
            
            best_mmr_score = -np.inf
            best_candidate_idx = -1
            
            # Các ứng viên còn lại để xét
            remaining_indices = set(candidates_pool.keys()) - set(final_results_indices)
            if not remaining_indices: break

            for cand_idx in remaining_indices:
                candidate = candidates_pool[cand_idx]
                
                relevance_score = candidate['final_score']
                
                # Tính max similarity với các kết quả đã chọn
                max_similarity = 0.0
                if final_results_indices:
                    similarities = [
                        self._calculate_similarity(candidate, candidates_pool[selected_idx])
                        for selected_idx in final_results_indices
                    ]
                    max_similarity = max(similarities)
                
                # Tính điểm MMR
                mmr_score = (lambda_val * relevance_score) - ((1 - lambda_val) * max_similarity)
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate_idx = cand_idx
            
            if best_candidate_idx != -1:
                final_results_indices.append(best_candidate_idx)
            else:
                # Không tìm thấy ứng viên nào phù hợp nữa
                break
                
        # Trả về danh sách các dictionary ứng viên theo đúng thứ tự MMR đã tìm được
        final_diverse_list = [candidates_pool[idx] for idx in final_results_indices]
        print(f"--- ✅ Xây dựng danh sách MMR hoàn tất với {len(final_diverse_list)} kết quả. ---")
        
        return final_diverse_list