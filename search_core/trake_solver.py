
from typing import List, Dict, Any
from search_core.openai_handler import OpenAIHandler 

# Import SemanticSearcher để type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .semantic_searcher import SemanticSearcher

class TRAKESolver:
    """
    Class xử lý Nhiệm vụ TRAKE.
    Giờ đây nó nhận một AI Handler để thực hiện việc phân rã truy vấn.
    """
    # --- THAY ĐỔI __init__ ---
    def __init__(self, ai_handler: OpenAIHandler):
        """
        Khởi tạo TRAKESolver.

        Args:
            ai_handler (OpenAIHandler): Một instance của OpenAIHandler đã được khởi tạo.
        """
        self.ai_handler = ai_handler

    # --- THAY ĐỔI decompose_query ---
    def decompose_query(self, query: str) -> List[str]:
        """
        Sử dụng AI Handler được cung cấp để tách truy vấn TRAKE thành các bước con.
        """
        print(f"--- 🤖 Phân rã truy vấn TRAKE bằng AI Handler... ---")
        # Ủy quyền hoàn toàn việc gọi API cho handler
        return self.ai_handler.decompose_trake_query(query)

    def find_sequences(self, sub_queries: List[str], searcher: 'SemanticSearcher', top_k_per_step: int = 15, max_sequences: int = 50) -> List[Dict[str, Any]]:
        """
        Tìm các chuỗi keyframe hợp lệ dựa trên danh sách các truy vấn con.

        Args:
            sub_queries (List[str]): Danh sách các truy vấn cho từng bước.
            searcher (SemanticSearcher): Instance của SemanticSearcher để thực hiện tìm kiếm.
            top_k_per_step (int): Số lượng ứng viên hàng đầu cần lấy cho mỗi bước.
            max_sequences (int): Số lượng chuỗi kết quả tối đa trả về.

        Returns:
            List[Dict[str, Any]]: Một danh sách các chuỗi hợp lệ, được sắp xếp theo điểm.
                                  Mỗi chuỗi là một dict chứa 'video_id', 'sequence' (list of keyframes), và 'score'.
        """
        if not sub_queries:
            return []

        print(f"--- Bắt đầu tìm kiếm ứng viên cho {len(sub_queries)} bước TRAKE ---")
        
        # --- Bước 1: Tìm kiếm song song để lấy ứng viên cho mỗi bước ---
        step_candidates = []
        for i, sub_query in enumerate(sub_queries):
            print(f"   -> Tìm kiếm cho Bước {i+1}: '{sub_query}'")
            # Lấy nhiều ứng viên hơn một chút để tăng cơ hội tìm thấy chuỗi
            results = searcher.search(sub_query, top_k_final=top_k_per_step, top_k_retrieval=200)
            step_candidates.append(results)

        # --- Bước 2: Nhóm các ứng viên theo video_id để giảm không gian tìm kiếm ---
        print("\n--- Đang nhóm các ứng viên theo video ---")
        candidates_by_video: Dict[str, List[List[Dict]]] = {}
        for i, candidates in enumerate(step_candidates):
            for cand in candidates:
                video_id = cand['video_id']
                if video_id not in candidates_by_video:
                    # Tạo cấu trúc [[], [], [], ...] cho mỗi video
                    candidates_by_video[video_id] = [[] for _ in sub_queries]
                candidates_by_video[video_id][i].append(cand)
        
        # --- Bước 3: Duyệt qua từng video để tìm các chuỗi hợp lệ ---
        print("\n--- Bắt đầu lắp ráp và xác thực các chuỗi ---")
        valid_sequences = []
        for video_id, steps in candidates_by_video.items():
            # Điều kiện tiên quyết: video phải có ít nhất một ứng viên cho MỖI bước
            if not all(steps):
                continue
            
            # Sử dụng `itertools.product` để tạo ra tất cả các tổ hợp chuỗi khả thi trong video này
            # Ví dụ: steps = [[a1, a2], [b1], [c1, c2]]
            # product -> (a1, b1, c1), (a1, b1, c2), (a2, b1, c1), (a2, b1, c2)
            for sequence_tuple in product(*steps):
                # Kiểm tra điều kiện thời gian tăng dần, một ràng buộc cứng
                is_valid_order = all(
                    sequence_tuple[i]['timestamp'] < sequence_tuple[i+1]['timestamp'] 
                    for i in range(len(sequence_tuple) - 1)
                )
                
                if is_valid_order:
                    # Tính điểm cho chuỗi, có thể là trung bình cộng hoặc các phương pháp phức tạp hơn
                    # Ở đây dùng trung bình cộng là một khởi đầu tốt
                    avg_score = sum(item['final_score'] for item in sequence_tuple) / len(sequence_tuple)
                    
                    valid_sequences.append({
                        "video_id": video_id,
                        "sequence": list(sequence_tuple), # Chuyển tuple thành list
                        "final_score": avg_score  # Thống nhất tên key là 'final_score'
                    })

        # --- Bước 4: Sắp xếp tất cả các chuỗi hợp lệ từ tất cả các video ---
        print(f"--- Tìm thấy tổng cộng {len(valid_sequences)} chuỗi hợp lệ. Đang sắp xếp... ---")
        sorted_sequences = sorted(valid_sequences, key=lambda x: x['final_score'], reverse=True)
        
        return sorted_sequences[:max_sequences]
