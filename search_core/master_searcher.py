from typing import Dict, Any, Optional
import google.generativeai as genai # Vẫn cần cho type hinting nếu dùng Gemini
from google.api_core import exceptions as google_exceptions

# Import các module cốt lõi của hệ thống
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.trake_solver import TRAKESolver
from search_core.track_vqa_solver import TrackVQASolver
from search_core.gemini_text_handler import GeminiTextHandler
from search_core.openai_handler import OpenAIHandler
from search_core.task_analyzer import TaskType

class MasterSearcher:
    """
    Lớp điều phối chính của hệ thống tìm kiếm (Hybrid AI Edition).
    Nó quản lý và điều phối các AI Handler khác nhau (Gemini cho text, OpenAI cho vision)
    để giải quyết các loại truy vấn phức tạp.
    """

    def __init__(self, 
                 basic_searcher: BasicSearcher, 
                 gemini_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """
        Khởi tạo MasterSearcher và hệ sinh thái AI lai.

        Args:
            basic_searcher (BasicSearcher): Instance của BasicSearcher đã được khởi tạo.
            gemini_api_key (Optional[str]): API key cho Google Gemini (dùng cho text).
            openai_api_key (Optional[str]): API key cho OpenAI (dùng cho vision/VQA).
        """
        print("--- 🧠 Khởi tạo Master Searcher (Hybrid AI Edition) ---")
        
        self.semantic_searcher = SemanticSearcher(basic_searcher=basic_searcher)
        
        self.gemini_handler: Optional[GeminiTextHandler] = None
        self.openai_handler: Optional[OpenAIHandler] = None
        self.trake_solver: Optional[TRAKESolver] = None
        self.track_vqa_solver: Optional[TrackVQASolver] = None
        self.ai_enabled = False

        # --- Khởi tạo và xác thực Gemini Handler cho các tác vụ TEXT ---
        if gemini_api_key:
            try:
                self.gemini_handler = GeminiTextHandler(api_key=gemini_api_key)
                self.ai_enabled = True # Bật cờ AI nếu ít nhất một handler hoạt động
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo Gemini Handler: {e}. Các tính năng text AI sẽ bị hạn chế. ---")

        # --- Khởi tạo và xác thực OpenAI Handler cho các tác vụ VISION ---
        if openai_api_key:
            try:
                self.openai_handler = OpenAIHandler(api_key=openai_api_key)
                if not self.openai_handler.check_api_health():
                    self.openai_handler = None # Vô hiệu hóa nếu health check thất bại
                else:
                    self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo OpenAI Handler: {e}. Các tính năng vision AI sẽ bị hạn chế. ---")
        
        # --- Khởi tạo các Solver phức tạp nếu các handler cần thiết đã sẵn sàng ---
        if self.gemini_handler:
            # TRAKE Solver chỉ cần text handler để phân rã truy vấn
            self.trake_solver = TRAKESolver(ai_handler=self.gemini_handler)
        
        if self.gemini_handler and self.openai_handler:
            # TrackVQASolver cần cả hai: text để phân tích, vision để hỏi đáp
            self.track_vqa_solver = TrackVQASolver(
                text_handler=self.gemini_handler, 
                vision_handler=self.openai_handler,
                semantic_searcher=self.semantic_searcher
            )

        print(f"--- ✅ Master Searcher đã sẵn sàng! (AI Enabled: {self.ai_enabled}) ---")

    def search(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hàm tìm kiếm chính, điều phối toàn bộ pipeline lai.
        """
        # --- Bước 1: Phân tích Truy vấn (Luôn dùng Gemini Text Handler) ---
        query_analysis = {}
        task_type = TaskType.KIS

        if self.ai_enabled and self.gemini_handler:
            print("--- ✨ Bắt đầu phân tích truy vấn bằng Gemini Text Handler... ---")
            query_analysis = self.gemini_handler.enhance_query(query)
            task_type_str = self.gemini_handler.analyze_task_type(query)
            try:
                task_type = TaskType[task_type_str]
            except KeyError:
                task_type = TaskType.KIS
        
        print(f"--- Đã phân loại truy vấn là: {task_type.value} ---")

        final_results = []
        top_k_final = config.get('top_k_final', 12)
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