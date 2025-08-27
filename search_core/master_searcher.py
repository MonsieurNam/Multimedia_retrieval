from typing import Dict, Any, Optional
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions # Import để bắt lỗi cụ thể

# Import các module cốt lõi của hệ thống
from .task_analyzer import TaskType, analyze_query_gemini, analyze_query_heuristic
from .semantic_searcher import SemanticSearcher
from .vqa_handler import VQAHandler
from .trake_solver import TRAKESolver
from ..utils import gemini_api_retrier # Import retrier

class MasterSearcher:
    """
    Lớp điều phối chính của hệ thống tìm kiếm.
    """

    def __init__(self, 
                 semantic_searcher: SemanticSearcher, 
                 gemini_api_key: Optional[str] = None):
        """
        Khởi tạo MasterSearcher và tất cả các thành phần con của nó.
        """
        print("--- 🧠 Khởi tạo Master Searcher ---")
        
        self.semantic_searcher = semantic_searcher
        self.gemini_model: Optional[genai.GenerativeModel] = None
        self.vqa_handler: Optional[VQAHandler] = None
        self.trake_solver: Optional[TRAKESolver] = None
        self.ai_enabled = False # Mặc định là False

        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
                # --- THÊM BƯỚC KIỂM TRA API HEALTH ---
                if self._check_api_health():
                    # Chỉ khởi tạo các handler con nếu API hoạt động
                    self.semantic_searcher.gemini_model = self.gemini_model
                    self.vqa_handler = VQAHandler(model=self.gemini_model)
                    self.trake_solver = TRAKESolver(gemini_model=self.gemini_model)
                    self.ai_enabled = True
                    print("--- ✅ Gemini và các AI Handler đã được khởi tạo và xác thực thành công! ---")
                else:
                    # Nếu health check thất bại, vô hiệu hóa các tính năng AI
                    print("--- ❌ Kiểm tra API thất bại. Các tính năng AI sẽ bị vô hiệu hóa. ---")
                    self.ai_enabled = False # Đảm bảo vẫn là False
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo Gemini: {e}. AI Handler sẽ bị vô hiệu hóa. ---")
                self.ai_enabled = False
        else:
            print("--- ⚠️ Không có API Key. AI Handler (Q&A, TRAKE) sẽ bị vô hiệu hóa. ---")
            self.ai_enabled = False
            
        print(f"--- ✅ Master Searcher đã sẵn sàng! (AI Enabled: {self.ai_enabled}) ---")

    # --- HÀM MỚI ---
    @gemini_api_retrier(max_retries=2, initial_delay=1) # Thử lại 2 lần nếu có lỗi mạng tạm thời
    def _check_api_health(self) -> bool:
        """
        Thực hiện một lệnh gọi API đơn giản để kiểm tra xem API key có hợp lệ và hoạt động không.
        
        Sử dụng count_tokens, một API call nhẹ và rẻ.

        Returns:
            bool: True nếu API hoạt động, False nếu không.
        """
        print("--- 🩺 Đang thực hiện kiểm tra trạng thái API Gemini... ---")
        try:
            # count_tokens là một lệnh gọi API nhẹ nhàng nhất
            self.gemini_model.count_tokens("kiểm tra")
            print("--- ✅ Trạng thái API: OK ---")
            return True
        except google_exceptions.PermissionDenied as e:
            # Lỗi này đặc trưng cho API key sai hoặc không có quyền truy cập model
            print(f"--- ❌ Lỗi API: Permission Denied. API Key có thể không hợp lệ. Lỗi: {e} ---")
            return False
        except Exception as e:
            # Bắt các lỗi khác (mạng, etc.)
            print(f"--- ❌ Lỗi API: Không thể kết nối đến Gemini. Lỗi: {e} ---")
            return False


    def search(self, query: str, top_k: int = 100) -> Dict[str, Any]:
        """
        Hàm tìm kiếm chính, điều phối pipeline theo quy chế thi MỚI.
        *** PHIÊN BẢN ĐÃ CẬP NHẬT LOGIC TÍNH ĐIỂM VQA ***
        """
        query_analysis = {}
        
        if self.ai_enabled:
            print("--- 🧠 Bắt đầu phân tích và tăng cường truy vấn bằng Gemini... ---")
            query_analysis = self.semantic_searcher.enhance_query_with_gemini(query)
            task_type = analyze_query_gemini(query, self.gemini_model)
        else:
            print("--- Chạy ở chế độ KIS cơ bản do AI bị vô hiệu hóa ---")
            query_analysis = {'search_context': query, 'objects_en': query.split()}
            task_type = analyze_query_heuristic(query)
        
        print(f"--- Đã phân loại truy vấn là: {task_type.value} ---")

        final_results = []
        
        if task_type == TaskType.TRAKE:
            if self.trake_solver:
                sub_queries = self.trake_solver.decompose_query(query)
                final_results = self.trake_solver.find_sequences(sub_queries, self.semantic_searcher, max_sequences=top_k)
            else:
                print("--- ⚠️ Không thể xử lý TRAKE. Fallback về tìm kiếm KIS. ---")
                task_type = TaskType.KIS
        
        if task_type == TaskType.KIS or task_type == TaskType.QNA:
            search_context = query_analysis.get('search_context', query)
            
            candidates = self.semantic_searcher.search(
                query_text=search_context, 
                top_k_final=top_k if task_type == TaskType.KIS else 20,
                top_k_retrieval=200,
                precomputed_analysis=query_analysis
            )
            
            if task_type == TaskType.KIS:
                final_results = candidates
            else: # task_type == TaskType.QNA
                if not self.vqa_handler:
                    print("--- ⚠️ Không thể xử lý QNA. Trả về kết quả tìm kiếm bối cảnh. ---")
                    final_results = candidates
                else:
                    specific_question = query_analysis.get('specific_question', query)
                    vqa_enhanced_candidates = []
                    for cand in candidates:
                        vqa_result = self.vqa_handler.get_answer(cand['keyframe_path'], specific_question)
                        
                        new_cand = cand.copy()
                        new_cand['answer'] = vqa_result['answer']
                        
                        # --- LOGIC TÍNH ĐIỂM MỚI ---
                        search_score = new_cand['final_score']
                        vqa_confidence = vqa_result['confidence']
                        final_vqa_score = search_score * vqa_confidence
                        
                        new_cand['final_score'] = final_vqa_score
                        new_cand['scores']['search_score'] = search_score
                        new_cand['scores']['vqa_confidence'] = vqa_confidence
                        
                        vqa_enhanced_candidates.append(new_cand)
                    
                    final_results = sorted(vqa_enhanced_candidates, key=lambda x: x['final_score'], reverse=True)

        return {
            "task_type": task_type,
            "results": final_results[:top_k],
            "query_analysis": query_analysis
        }