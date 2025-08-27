# /kaggle/working/search_core/master_searcher.py

from typing import Dict, Any, Optional

# Import các thành phần cần thiết
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.trake_solver import TRAKESolver
from search_core.openai_handler import OpenAIHandler # <-- IMPORT HANDLER MỚI
from search_core.task_analyzer import TaskType      # <-- Vẫn dùng Enum để code rõ ràng

class MasterSearcher:
    """
    Lớp điều phối chính của hệ thống tìm kiếm (OpenAI Edition).
    Nó quản lý OpenAIHandler và điều phối các tác vụ đến đúng nơi.
    """

    def __init__(self, 
                 basic_searcher: BasicSearcher, 
                 openai_api_key: Optional[str] = None):
        """
        Khởi tạo MasterSearcher phiên bản OpenAI.

        Args:
            basic_searcher (BasicSearcher): Một instance của BasicSearcher đã được khởi tạo.
            openai_api_key (Optional[str]): API key cho OpenAI.
        """
        print("--- 🧠 Khởi tạo Master Searcher (OpenAI Edition) ---")
        
        # SemanticSearcher không còn cần model AI nữa, nó chỉ làm nhiệm vụ reranking
        self.semantic_searcher = SemanticSearcher(basic_searcher=basic_searcher)
        
        self.openai_handler: Optional[OpenAIHandler] = None
        self.trake_solver: Optional[TRAKESolver] = None
        self.ai_enabled = False

        if openai_api_key:
            try:
                # Khởi tạo một handler duy nhất
                self.openai_handler = OpenAIHandler(api_key=openai_api_key)
                
                # TODO: Thêm health check cho OpenAI nếu cần, tương tự như đã làm với Gemini
                
                # Cung cấp handler cho các module con cần nó
                self.trake_solver = TRAKESolver(ai_handler=self.openai_handler)
                
                self.ai_enabled = True
                print("--- ✅ OpenAI Handler đã được khởi tạo thành công! ---")

            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo OpenAI Handler: {e}. AI sẽ bị vô hiệu hóa. ---")
                self.ai_enabled = False
        else:
            print("--- ⚠️ Không có OpenAI API Key. AI sẽ bị vô hiệu hóa. ---")
            self.ai_enabled = False
            
        print(f"--- ✅ Master Searcher đã sẵn sàng! (AI Enabled: {self.ai_enabled}) ---")

    def search(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hàm tìm kiếm chính, nhận một dictionary config để tùy chỉnh hành vi.
        *** PHIÊN BẢN HOÀN THIỆN: Sửa lỗi NameError và tái cấu trúc logic. ***
        """
        # --- Bước 1: Giải nén các tham số từ dictionary config ---
        # Cung cấp các giá trị mặc định an toàn nếu key không tồn tại
        top_k_final = config.get('top_k_final', 12)
        kis_retrieval = config.get('kis_retrieval', 100)
        vqa_candidates = config.get('vqa_candidates', 8)
        vqa_retrieval = config.get('vqa_retrieval', 200)
        trake_candidates_per_step = config.get('trake_candidates_per_step', 15)
        trake_max_sequences = config.get('trake_max_sequences', 50)
        
        # --- Bước 2: Phân tích truy vấn bằng AI ---
        query_analysis = {}
        task_type = TaskType.KIS # Mặc định

        if self.ai_enabled:
            print("--- 🤖 Bắt đầu phân tích truy vấn bằng OpenAI... ---")
            query_analysis = self.openai_handler.enhance_query(query)
            task_type_str = self.openai_handler.analyze_task_type(query)
            try:
                task_type = TaskType[task_type_str]
            except KeyError:
                print(f"--- ⚠️ Loại task không xác định '{task_type_str}'. Fallback về KIS. ---")
                task_type = TaskType.KIS
        
        print(f"--- Đã phân loại truy vấn là: {task_type.value} ---")

        final_results = []
        
        # --- Bước 3: Điều phối đến handler phù hợp ---

        if task_type == TaskType.TRAKE:
            if self.trake_solver:
                sub_queries = self.trake_solver.decompose_query(query)
                final_results = self.trake_solver.find_sequences(
                    sub_queries, 
                    self.semantic_searcher, 
                    top_k_per_step=trake_candidates_per_step,
                    max_sequences=trake_max_sequences
                )
            else:
                print("--- ⚠️ TRAKE handler chưa được kích hoạt. Fallback về tìm kiếm KIS. ---")
                task_type = TaskType.KIS # Chuyển task để khối logic tiếp theo xử lý
        
        # Khối này xử lý KIS, QNA, và cả fallback từ TRAKE
        if task_type == TaskType.QNA:
            search_context = query_analysis.get('search_context', query)
            candidates = self.semantic_searcher.search(
                query_text=search_context,
                precomputed_analysis=query_analysis,
                top_k_final=vqa_candidates,
                top_k_retrieval=vqa_retrieval
            )
            
            specific_question = query_analysis.get('specific_question', query)
            vqa_enhanced_candidates = []
            for cand in candidates:
                vqa_result = self.openai_handler.perform_vqa(cand['keyframe_path'], specific_question)
                
                new_cand = cand.copy()
                new_cand['answer'] = vqa_result['answer']
                
                search_score = new_cand['final_score']
                vqa_confidence = vqa_result['confidence']
                final_vqa_score = search_score * vqa_confidence
                
                new_cand['final_score'] = final_vqa_score
                new_cand['scores']['search_score'] = search_score
                new_cand['scores']['vqa_confidence'] = vqa_confidence
                vqa_enhanced_candidates.append(new_cand)
            
            final_results = sorted(vqa_enhanced_candidates, key=lambda x: x['final_score'], reverse=True)

        elif task_type == TaskType.KIS: # Chỉ chạy nếu task là KIS (hoặc fallback từ TRAKE)
            search_context = query_analysis.get('search_context', query)
            final_results = self.semantic_searcher.search(
                query_text=search_context,
                precomputed_analysis=query_analysis,
                top_k_final=top_k_final, 
                top_k_retrieval=kis_retrieval
            )

        # --- Bước 4: Trả về kết quả cuối cùng có cấu trúc ---
        return {
            "task_type": task_type,
            "results": final_results[:top_k_final], # Cắt theo top_k chung để hiển thị
            "query_analysis": query_analysis
        }