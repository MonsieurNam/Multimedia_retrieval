from typing import Dict, Any, Optional
import google.generativeai as genai # Vẫn cần cho type hinting nếu dùng Gemini
from google.api_core import exceptions as google_exceptions

# Import các module cốt lõi của hệ thống
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.trake_solver import TRAKESolver
from search_core.track_vqa_solver import TrackVQASolver
from search_core.openai_handler import OpenAIHandler
from search_core.task_analyzer import TaskType

class MasterSearcher:
    """
    Lớp điều phối chính của hệ thống tìm kiếm (OpenAI Edition).
    Nó quản lý OpenAIHandler và điều phối các tác vụ đến đúng solver/handler.
    Đây là entry point duy nhất cho toàn bộ backend.
    """

    def __init__(self, 
                 basic_searcher: BasicSearcher, 
                 openai_api_key: Optional[str] = None):
        """
        Khởi tạo MasterSearcher và tất cả các thành phần con của nó.

        Args:
            basic_searcher (BasicSearcher): Một instance của BasicSearcher đã được khởi tạo.
            openai_api_key (Optional[str]): API key cho OpenAI.
        """
        print("--- 🧠 Khởi tạo Master Searcher (OpenAI Edition) ---")
        
        # SemanticSearcher không quản lý model AI, chỉ làm nhiệm vụ reranking.
        self.semantic_searcher = SemanticSearcher(basic_searcher=basic_searcher)
        
        self.openai_handler: Optional[OpenAIHandler] = None
        self.trake_solver: Optional[TRAKESolver] = None
        self.track_vqa_solver: Optional[TrackVQASolver] = None
        self.ai_enabled = False

        if openai_api_key:
            try:
                # Khởi tạo một handler duy nhất để quản lý tất cả các lệnh gọi API
                self.openai_handler = OpenAIHandler(api_key=openai_api_key)
                
                # Thực hiện kiểm tra trạng thái API ngay khi khởi động
                if self.openai_handler.check_api_health():
                    # Nếu API hoạt động, khởi tạo các solver phụ thuộc vào nó
                    self.trake_solver = TRAKESolver(ai_handler=self.openai_handler)
                    self.track_vqa_solver = TrackVQASolver(ai_handler=self.openai_handler, semantic_searcher=self.semantic_searcher)
                    self.ai_enabled = True
                    print("--- ✅ OpenAI Handler và các AI Solver đã được khởi tạo và xác thực thành công! ---")
                else:
                    print("--- ❌ Kiểm tra API thất bại. Các tính năng AI sẽ bị vô hiệu hóa. ---")
                    self.ai_enabled = False
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
        """
        # --- Bước 1: Giải nén Config & Phân tích Truy vấn ---
        top_k_final = config.get('top_k_final', 12)
        
        query_analysis = {}
        task_type = TaskType.KIS

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
        
        # --- Bước 2: Khối Điều phối Logic ---
        search_context = query_analysis.get('search_context', query)

        if task_type == TaskType.TRACK_VQA:
            if self.track_vqa_solver:
                track_vqa_result = self.track_vqa_solver.solve(query_analysis)
                # Định dạng lại kết quả để UI có thể hiển thị
                final_results = [{
                    "is_aggregated_result": True,
                    "final_answer": track_vqa_result.get("final_answer", "Lỗi tổng hợp kết quả."),
                    "evidence_frames": track_vqa_result.get("evidence_frames", []),
                    "keyframe_path": track_vqa_result["evidence_frames"][0]['keyframe_path'] if track_vqa_result.get("evidence_frames") else "",
                    "video_id": track_vqa_result["evidence_frames"][0]['video_id'] if track_vqa_result.get("evidence_frames") else "N/A",
                    "timestamp": 0.0, "final_score": 1.0, "scores": {}
                }]
            else:
                print("--- ⚠️ TrackVQA handler chưa được kích hoạt. Fallback về KIS. ---")
                task_type = TaskType.KIS

        if task_type == TaskType.TRAKE:
            if self.trake_solver:
                sub_queries = self.trake_solver.decompose_query(query)
                final_results = self.trake_solver.find_sequences(
                    sub_queries, self.semantic_searcher, 
                    top_k_per_step=config.get('trake_candidates_per_step', 15),
                    max_sequences=config.get('trake_max_sequences', 50)
                )
            else:
                print("--- ⚠️ TRAKE handler chưa được kích hoạt. Fallback về KIS. ---")
                task_type = TaskType.KIS

        if task_type == TaskType.QNA:
            candidates = self.semantic_searcher.search(
                query_text=search_context,
                precomputed_analysis=query_analysis,
                top_k_final=config.get('vqa_candidates', 8),
                top_k_retrieval=config.get('vqa_retrieval', 200)
            )
            specific_question = query_analysis.get('specific_question', query)
            vqa_enhanced_candidates = []
            for cand in candidates:
                vqa_result = self.openai_handler.perform_vqa(cand['keyframe_path'], specific_question)
                new_cand = cand.copy()
                new_cand['answer'] = vqa_result['answer']
                search_score, vqa_confidence = new_cand['final_score'], vqa_result['confidence']
                new_cand['final_score'] = search_score * vqa_confidence
                new_cand['scores']['search_score'] = search_score
                new_cand['scores']['vqa_confidence'] = vqa_confidence
                vqa_enhanced_candidates.append(new_cand)
            final_results = sorted(vqa_enhanced_candidates, key=lambda x: x['final_score'], reverse=True)

        if task_type == TaskType.KIS: # Bắt các trường hợp KIS gốc và các fallback
            final_results = self.semantic_searcher.search(
                query_text=search_context,
                precomputed_analysis=query_analysis,
                top_k_final=top_k_final, 
                top_k_retrieval=config.get('kis_retrieval', 100)
            )

        # --- Bước 3: Trả về kết quả cuối cùng ---
        return {
            "task_type": task_type,
            "results": final_results,
            "query_analysis": query_analysis
        }