from typing import Dict, Any, Optional
import google.generativeai as genai

# Import các module cốt lõi của hệ thống
from search_core.task_analyzer import TaskType, analyze_query_gemini, analyze_query_heuristic
from search_core.semantic_searcher import SemanticSearcher
from search_core.vqa_handler import VQAHandler
from search_core.trake_solver import TRAKESolver
class MasterSearcher:
    """
    Lớp điều phối chính của hệ thống tìm kiếm, hoạt động như một Facade.

    Đây là entry point duy nhất cho tất cả các truy vấn. Nó có trách nhiệm:
    1.  Khởi tạo và quản lý các handler chuyên biệt (Semantic, VQA, TRAKE).
    2.  Phân tích loại truy vấn của người dùng.
    3.  Điều phối truy vấn đến handler phù hợp để xử lý.
    4.  Thực hiện các logic nghiệp vụ phức tạp như kết hợp kết quả, cập nhật điểm số.
    5.  Trả về một kết quả có cấu trúc, sẵn sàng cho giao diện người dùng.
    """

    def __init__(self, 
                 semantic_searcher: SemanticSearcher, 
                 gemini_api_key: Optional[str] = None):
        """
        Khởi tạo MasterSearcher và tất cả các thành phần con của nó.

        Args:
            semantic_searcher (SemanticSearcher): Một instance của SemanticSearcher đã được khởi tạo.
            gemini_api_key (Optional[str]): API key cho Google Gemini. Nếu được cung cấp,
                                            các tính năng AI nâng cao sẽ được kích hoạt.
        """
        print("--- 🧠 Khởi tạo Master Searcher ---")
        
        self.semantic_searcher = semantic_searcher
        self.gemini_model: Optional[genai.GenerativeModel] = None
        self.vqa_handler: Optional[VQAHandler] = None
        self.trake_solver: Optional[TRAKESolver] = None
        self.ai_enabled = False

        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                # Sử dụng 'gemini-1.5-flash' cho sự cân bằng giữa tốc độ và khả năng
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Chia sẻ instance Gemini model cho tất cả các handler con
                # Điều này giúp tiết kiệm tài nguyên và thời gian khởi tạo
                self.semantic_searcher.gemini_model = self.gemini_model
                self.vqa_handler = VQAHandler(model=self.gemini_model)
                self.trake_solver = TRAKESolver(gemini_model=self.gemini_model)
                
                print("--- ✅ Gemini và các AI Handler đã được khởi tạo thành công! ---")
                self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo Gemini: {e}. AI Handler sẽ bị vô hiệu hóa. ---")
        else:
            print("--- ⚠️ Không có API Key. AI Handler (Q&A, TRAKE) sẽ bị vô hiệu hóa. ---")
            
        print("--- ✅ Master Searcher đã sẵn sàng! ---")

    def search(self, query: str, top_k: int = 100) -> Dict[str, Any]:
        """
        Hàm tìm kiếm chính, điều phối toàn bộ pipeline.

        Args:
            query (str): Câu truy vấn của người dùng.
            top_k (int): Số lượng kết quả cuối cùng mong muốn.

        Returns:
            Dict[str, Any]: Một dictionary chứa:
                - 'task_type' (TaskType): Loại nhiệm vụ đã được phân loại.
                - 'results' (list): Danh sách các kết quả đã được xử lý.
                - 'query_analysis' (dict): Thông tin phân tích từ Gemini (nếu có).
        """
        query_analysis = {}
        if self.ai_enabled:
            print("--- 🧠 Bắt đầu phân tích và tăng cường truy vấn bằng Gemini... ---")
            query_analysis = self.semantic_searcher.enhance_query_with_gemini(query)
            task_type = analyze_query_gemini(query, self.gemini_model)
        else:
            print("--- Chạy ở chế độ KIS cơ bản do AI bị vô hiệu hóa ---")
            task_type = analyze_query_heuristic(query)
        
        print(f"--- Đã phân loại truy vấn là: {task_type.value} ---")

        final_results = []
        
        # --- Điều phối dựa trên loại nhiệm vụ ---
        if task_type == TaskType.TRAKE:
            if not self.trake_solver:
                print("--- ⚠️ Không thể xử lý TRAKE. Đang chạy tìm kiếm KIS thay thế. ---")
                final_results = self.semantic_searcher.search(
                    query, top_k_final=top_k, precomputed_analysis=query_analysis)
            else:
                # 1. Phân rã truy vấn thành các bước con
                sub_queries = self.trake_solver.decompose_query(query)
                # 2. Tìm các chuỗi hợp lệ
                final_results = self.trake_solver.find_sequences(
                    sub_queries, self.semantic_searcher, max_sequences=top_k)
        else: # Mặc định tất cả các truy vấn khác đều là KIS
            final_results = self.semantic_searcher.search(
                query, 
                top_k_final=top_k,
                precomputed_analysis=query_analysis
            )

        return {
            "task_type": task_type,
            "results": final_results[:top_k], # Đảm bảo số lượng kết quả cuối cùng đúng bằng top_k
            "query_analysis": query_analysis
        }
        
    def perform_vqa(self, keyframe_info: Dict, question: str) -> Dict:
        """
        Thực hiện VQA trên một keyframe cụ thể theo yêu cầu.
        Đây là cổng API mới cho giao diện.
        """
        if not self.vqa_handler:
            return {"answer": "Lỗi: Tính năng VQA chưa được kích hoạt.", "confidence": 0.0}
        
        if not keyframe_info or not 'keyframe_path' in keyframe_info:
            return {"answer": "Lỗi: Thiếu thông tin keyframe.", "confidence": 0.0}

        print(f"--- 🗣️ Thực hiện VQA theo yêu cầu cho keyframe {keyframe_info.get('keyframe_id')} ---")
        return self.vqa_handler.get_answer(keyframe_info['keyframe_path'], question)