
import json
import re
from itertools import product
from typing import List, Dict, Any, Optional
import google.generativeai as genai

# Import SemanticSearcher để type hinting, nhưng không thực sự import để tránh circular dependency
# Nó sẽ được truyền vào từ MasterSearcher
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .semantic_searcher import SemanticSearcher

class TRAKESolver:
    """
    Class xử lý Nhiệm vụ 3: TRAKE (TRacking Action KEyframes).

    Nó bao gồm hai chức năng chính:
    1.  Phân rã một truy vấn hành động phức tạp thành các bước con mô tả ngắn gọn.
    2.  Tìm kiếm các chuỗi keyframe hợp lệ từ các ứng viên của mỗi bước,
        đảm bảo chúng cùng video và theo đúng thứ tự thời gian.
    """

    def __init__(self, gemini_model: genai.GenerativeModel):
        """
        Khởi tạo TRAKESolver.

        Args:
            gemini_model (genai.GenerativeModel): Một instance của Gemini model đã được khởi tạo
                                                  để sử dụng cho việc phân rã truy vấn.
        """
        self.gemini_model = gemini_model

    def decompose_query(self, query: str) -> List[str]:
        """
        Sử dụng Gemini để tách một truy vấn TRAKE thành các bước hành động con.

        Ví dụ: "Tìm 4 khoảnh khắc VĐV nhảy: (1) giậm nhảy, (2) bay qua xà, ..."
        -> ["vận động viên giậm nhảy", "vận động viên bay qua xà", ...]

        Args:
            query (str): Câu truy vấn TRAKE gốc của người dùng.

        Returns:
            List[str]: Một danh sách các chuỗi mô tả từng bước hành động.
        """
        # Prompt này hướng dẫn Gemini giữ lại ngữ cảnh (VD: "vận động viên")
        # cho mỗi bước, giúp việc tìm kiếm sau này chính xác hơn.
        prompt = f"""
        Analyze the following Vietnamese video search query that describes a sequence of actions. Your task is to decompose it into a short, descriptive search query for EACH key action step. Each step should be self-contained and understandable on its own.
        
        Return ONLY a valid JSON array of strings, where each string is a search query for one step.

        **Example 1:**
        Query: "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
        JSON: ["vận động viên giậm nhảy", "vận động viên bay qua xà", "vận động viên tiếp đất", "vận động viên đứng dậy"]

        **Example 2:**
        Query: "Một chiếc ô tô bắt đầu di chuyển, tăng tốc, và sau đó dừng lại"
        JSON: ["ô tô bắt đầu di chuyển", "ô tô tăng tốc trên đường", "ô tô dừng lại"]

        **Query to process:** "{query}"
        **JSON:**
        """
        try:
            # Sử dụng cùng safety settings như các module khác
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
            response = self.gemini_model.generate_content(prompt, safety_settings=safety_settings)
            
            # Trích xuất khối JSON một cách an toàn
            match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if match:
                decomposed_list = json.loads(match.group(0))
                if isinstance(decomposed_list, list) and all(isinstance(i, str) for i in decomposed_list):
                    print(f"--- ✅ Phân rã truy vấn TRAKE thành công: {decomposed_list} ---")
                    return decomposed_list
            
            # Fallback nếu không parse được JSON
            print(f"--- ⚠️ Không thể phân rã truy vấn TRAKE, sử dụng truy vấn gốc. ---")
            return [query]

        except Exception as e:
            print(f"--- ⚠️ Lỗi khi phân rã truy vấn TRAKE: {e}. Sử dụng truy vấn gốc. ---")
            return [query]

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

# --- Block để kiểm thử nhanh ---
if __name__ == '__main__':
    print("--- Chạy kiểm thử cho Module TRAKE Solver ---")

    # --- Mock các đối tượng cần thiết ---
    class MockSemanticSearcher:
        def search(self, query, top_k_final, top_k_retrieval):
            print(f"    (Mock Search) Đang tìm kiếm cho '{query}'")
            # Tạo dữ liệu giả dựa trên query
            if "giậm nhảy" in query:
                return [
                    {'video_id': 'V01', 'keyframe_id': '010', 'timestamp': 10.0, 'final_score': 0.9},
                    {'video_id': 'V02', 'keyframe_id': '015', 'timestamp': 15.0, 'final_score': 0.8},
                ]
            if "bay qua xà" in query:
                return [
                    {'video_id': 'V01', 'keyframe_id': '050', 'timestamp': 50.0, 'final_score': 0.85}, # Hợp lệ
                    {'video_id': 'V01', 'keyframe_id': '005', 'timestamp': 5.0, 'final_score': 0.7},  # Sai thứ tự
                ]
            if "tiếp đất" in query:
                return [
                    {'video_id': 'V01', 'keyframe_id': '100', 'timestamp': 100.0, 'final_score': 0.92},
                    {'video_id': 'V03', 'keyframe_id': '110', 'timestamp': 110.0, 'final_score': 0.88},
                ]
            return []

    class MockGeminiModel:
        def generate_content(self, prompt, safety_settings=None):
            class MockResponse:
                def __init__(self, text):
                    self.text = text
            # Giả lập Gemini trả về JSON
            return MockResponse(text='["vận động viên giậm nhảy", "vận động viên bay qua xà", "vận động viên tiếp đất"]')

    # --- Bắt đầu kiểm thử ---
    mock_gemini = MockGeminiModel()
    mock_searcher = MockSemanticSearcher()
    
    trake_solver = TRAKESolver(gemini_model=mock_gemini)
    
    test_query = "Các bước nhảy cao: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất"
    
    print("\n--- 1. Kiểm thử Decompose Query ---")
    sub_queries = trake_solver.decompose_query(test_query)
    print(f" -> Kết quả: {sub_queries}")
    assert len(sub_queries) == 3
    assert sub_queries[0] == "vận động viên giậm nhảy"

    print("\n--- 2. Kiểm thử Find Sequences ---")
    sequences = trake_solver.find_sequences(sub_queries, mock_searcher)
    
    print("\n--- Các chuỗi hợp lệ tìm thấy: ---")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i+1}: Video {seq['video_id']}, Score: {seq['final_score']:.3f}")
        timestamps = [frame['timestamp'] for frame in seq['sequence']]
        print(f"    Timestamps: {timestamps}")

    assert len(sequences) == 1, "Phải tìm thấy đúng 1 chuỗi hợp lệ"
    assert sequences[0]['video_id'] == 'V01'
    assert sequences[0]['sequence'][0]['timestamp'] == 10.0
    assert sequences[0]['sequence'][1]['timestamp'] == 50.0
    assert sequences[0]['sequence'][2]['timestamp'] == 100.0
    
    print("\n✅ Kiểm thử TRAKE Solver thành công!")
