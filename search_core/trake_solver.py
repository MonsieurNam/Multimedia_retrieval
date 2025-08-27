
from typing import List, Dict, Any
from itertools import product

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from search_core.openai_handler import OpenAIHandler
    from search_core.semantic_searcher import SemanticSearcher

class TRAKESolver:
    """
    Class xử lý Nhiệm vụ 3: TRAKE (TRacking Action KEyframes).

    Nó bao gồm hai chức năng chính:
    1.  Phân rã một truy vấn hành động phức tạp thành các bước con (sử dụng AI handler).
    2.  Tìm kiếm các chuỗi keyframe hợp lệ bằng cách:
        a. Phân tích và tìm kiếm ứng viên cho từng bước con.
        b. Lắp ráp các ứng viên thành các chuỗi hợp lệ (cùng video, đúng thứ tự).
    """

    def __init__(self, ai_handler: 'OpenAIHandler'):
        """
        Khởi tạo TRAKESolver.

        Args:
            ai_handler (OpenAIHandler): Một instance của AI Handler (ví dụ: OpenAIHandler)
                                        để thực hiện việc phân rã và phân tích truy vấn.
        """
        self.ai_handler = ai_handler

    def decompose_query(self, query: str) -> List[str]:
        """
        Sử dụng AI Handler được cung cấp để tách truy vấn TRAKE thành các bước hành động con.
        """
        print(f"--- 🤖 Phân rã truy vấn TRAKE bằng AI Handler... ---")
        # Ủy quyền hoàn toàn việc gọi API cho handler
        return self.ai_handler.decompose_trake_query(query)

    def find_sequences(self, 
                       sub_queries: List[str], 
                       searcher: 'SemanticSearcher', 
                       top_k_per_step: int , 
                       max_sequences: int
                      ) -> List[Dict[str, Any]]:
        """
        Tìm các chuỗi keyframe hợp lệ dựa trên danh sách các truy vấn con.

        Đây là hàm cốt lõi của TRAKE, thực hiện tìm kiếm đa bước và lắp ráp kết quả.

        Args:
            sub_queries (List[str]): Danh sách các truy vấn cho từng bước.
            searcher (SemanticSearcher): Instance của SemanticSearcher để thực hiện reranking.
            top_k_per_step (int): Số lượng ứng viên hàng đầu cần lấy cho mỗi bước.
            max_sequences (int): Số lượng chuỗi kết quả tối đa trả về.

        Returns:
            List[Dict[str, Any]]: Một danh sách các chuỗi hợp lệ, được sắp xếp theo điểm.
        """
        if not sub_queries:
            return []

        print(f"--- Bắt đầu tìm kiếm ứng viên cho {len(sub_queries)} bước TRAKE ---")
        
        # --- Bước 1: Tìm kiếm ứng viên cho mỗi bước một cách độc lập ---
        step_candidates = []
        for i, sub_query in enumerate(sub_queries):
            print(f"   -> Bước {i+1}: Đang phân tích và tìm kiếm cho '{sub_query}'")
            
            # 1a. Tự gọi AI Handler để phân tích truy vấn con
            # Điều này đảm bảo mỗi bước được tìm kiếm với context và object chính xác nhất.
            sub_query_analysis = self.ai_handler.enhance_query(sub_query)
            search_context = sub_query_analysis.get('search_context', sub_query)

            # 1b. Truyền kết quả phân tích vào searcher để reranking
            # SemanticSearcher giờ đây không cần gọi API nữa, chỉ làm nhiệm vụ rerank.
            results = searcher.search(
                query_text=search_context,
                precomputed_analysis=sub_query_analysis,
                top_k_final=top_k_per_step,
                top_k_retrieval=200  # Lấy nhiều ứng viên thô để tăng cơ hội
            )
            step_candidates.append(results)

        # --- Bước 2: Nhóm các ứng viên theo video_id để giảm không gian tìm kiếm ---
        print("\n--- Đang nhóm các ứng viên theo video ---")
        candidates_by_video: Dict[str, List[List[Dict]]] = {}
        for i, candidates in enumerate(step_candidates):
            for cand in candidates:
                video_id = cand['video_id']
                if video_id not in candidates_by_video:
                    candidates_by_video[video_id] = [[] for _ in sub_queries]
                candidates_by_video[video_id][i].append(cand)
        
        # --- Bước 3: Duyệt qua từng video để tìm và xác thực các chuỗi ---
        print("\n--- Bắt đầu lắp ráp và xác thực các chuỗi ---")
        valid_sequences = []
        for video_id, steps in candidates_by_video.items():
            # Điều kiện tiên quyết: video phải có ít nhất một ứng viên cho MỖI bước
            if not all(steps):
                continue
            
            # Sử dụng `itertools.product` để tạo ra tất cả các tổ hợp chuỗi khả thi
            for sequence_tuple in product(*steps):
                # Ràng buộc cứng: thứ tự thời gian phải tăng dần
                is_valid_order = all(
                    sequence_tuple[i]['timestamp'] < sequence_tuple[i+1]['timestamp'] 
                    for i in range(len(sequence_tuple) - 1)
                )
                
                if is_valid_order:
                    # Tính điểm cho chuỗi bằng trung bình cộng điểm của các frame
                    avg_score = sum(item['final_score'] for item in sequence_tuple) / len(sequence_tuple)
                    
                    valid_sequences.append({
                        "video_id": video_id,
                        "sequence": list(sequence_tuple),
                        "final_score": avg_score
                    })

        # --- Bước 4: Sắp xếp tất cả các chuỗi hợp lệ và trả về kết quả cuối cùng ---
        print(f"--- Tìm thấy tổng cộng {len(valid_sequences)} chuỗi hợp lệ. Đang sắp xếp... ---")
        sorted_sequences = sorted(valid_sequences, key=lambda x: x['final_score'], reverse=True)
        
        return sorted_sequences[:max_sequences]