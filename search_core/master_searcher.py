from typing import Dict, Any, Optional
import os
import json
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
                 openai_api_key: Optional[str] = None,
                 entities_path: str = None):
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
        self.known_entities: set = set()
        
        if entities_path and os.path.exists(entities_path):
            try:
                print(f"--- 📚 Đang tải Từ điển Đối tượng từ: {entities_path} ---")
                with open(entities_path, 'r') as f:
                    entities_list = [entity.lower() for entity in json.load(f)]
                    self.known_entities = set(entities_list)
                print(f"--- ✅ Tải thành công {len(self.known_entities)} thực thể đã biết. ---")
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi tải Từ điển Đối tượng: {e}. Tính năng Semantic Grounding sẽ bị vô hiệu hóa. ---")
                
        # --- Khởi tạo và xác thực Gemini Handler cho các tác vụ TEXT ---
        if gemini_api_key:
            try:
                self.gemini_handler = GeminiTextHandler(api_key=gemini_api_key)
                if self.known_entities and self.gemini_handler:
                    self.gemini_handler.load_known_entities(self.known_entities)
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
        Hàm tìm kiếm chính, nhận một dictionary config để tùy chỉnh hành vi.
        *** PHIÊN BẢN HOÀN THIỆN TÍCH HỢP ĐẦY ĐỦ CONFIG TỪ UI ***
        """
        # --- Bước 1: Giải nén toàn bộ Config từ UI với giá trị mặc định an toàn ---
        top_k_final = int(config.get('top_k_final', 12))
        
        # KIS config
        kis_retrieval = int(config.get('kis_retrieval', 100))
        
        # VQA config
        vqa_candidates_to_rank = int(config.get('vqa_candidates', 8))
        vqa_retrieval = int(config.get('vqa_retrieval', 200))

        # TRAKE config
        trake_candidates_per_step = int(config.get('trake_candidates_per_step', 15))
        trake_max_sequences = int(config.get('trake_max_sequences', 50))

        # Track-VQA config
        track_vqa_retrieval = int(config.get('track_vqa_retrieval', 200))
        track_vqa_candidates_to_analyze = int(config.get('track_vqa_candidates', 20))

        # --- Bước 2: Phân tích Truy vấn ---
        query_analysis = {}
        task_type = TaskType.KIS
        if self.ai_enabled and self.gemini_handler:
            print("--- ✨ Bắt đầu phân tích truy vấn bằng Gemini Text Handler... ---")
            query_analysis = self.gemini_handler.enhance_query(query)
            original_objects = query_analysis.get('objects_en', [])
            if original_objects: # Chỉ gọi API nếu có object để xử lý
                grounded_objects = self.gemini_handler.perform_semantic_grounding(original_objects)
                
                if original_objects != grounded_objects:
                     print(f"--- 🧠 Semantic Grounding: {original_objects} -> {grounded_objects} ---")
                
                query_analysis['objects_en'] = grounded_objects
                
            task_type_str = self.gemini_handler.analyze_task_type(query)
            try:
                task_type = TaskType[task_type_str]
            except KeyError:
                task_type = TaskType.KIS
        
        print(f"--- Đã phân loại truy vấn là: {task_type.value} ---")

        final_results = []
        search_context = query_analysis.get('search_context', query)

        # --- Bước 3: Khối Điều phối Logic (Cập nhật để truyền Config) ---

        if task_type == TaskType.TRACK_VQA:
            if self.track_vqa_solver:
                track_vqa_result = self.track_vqa_solver.solve(
                    query_analysis,
                    candidates_to_retrieve=track_vqa_retrieval,
                    candidates_to_analyze=track_vqa_candidates_to_analyze
                )
                
                evidence_frames = track_vqa_result.get("evidence_frames", [])
                for frame in evidence_frames:   
                    path = frame.get('keyframe_path')
                    print(f"DEBUG: Checking path '{path}'... Found: {path}") # <-- THÊM DÒNG NÀY
                # --- LOGIC "LÀM PHẲNG" DỮ LIỆU BẮT ĐẦU TỪ ĐÂY ---

                # 1. Tạo một danh sách các đường dẫn ảnh (chỉ string)
                evidence_paths = [
                    frame.get('keyframe_path') 
                    for frame in evidence_frames 
                    if frame.get('keyframe_path') and os.path.isfile(frame.get('keyframe_path'))
                ]
                
                # 2. Tạo một danh sách các chú thích (chỉ string)
                evidence_captions = [
                    f"{frame.get('video_id', 'N/A')} @{frame.get('timestamp', 0):.1f}s"
                    for frame in evidence_frames
                    if frame.get('keyframe_path') and os.path.isfile(frame.get('keyframe_path'))
                ]

                # 3. Tạo một "kết quả ảo" duy nhất chứa dữ liệu đã được làm phẳng
                final_results = [{
                    "is_aggregated_result": True,
                    "final_answer": track_vqa_result.get("final_answer", "Lỗi tổng hợp kết quả."),
                    
                    # Thay thế list of dicts phức tạp bằng các list of strings đơn giản
                    "evidence_paths": evidence_paths, 
                    "evidence_captions": evidence_captions,
                    
                    # Cung cấp keyframe đầu tiên để gallery có ảnh đại diện.
                    # Đảm bảo nó là None nếu không có bằng chứng nào hợp lệ.
                    "keyframe_path": evidence_paths[0] if evidence_paths else None,
                    
                    # Cung cấp các thông tin giả để các hàm khác không bị lỗi
                    "video_id": "Tổng hợp",
                    "timestamp": 0.0,
                    "final_score": 1.0, # Điểm cao nhất vì đây là kết quả cuối cùng
                    "scores": {}
                }]
            else:
                print("--- ⚠️ TrackVQA handler chưa được kích hoạt. Fallback về KIS. ---")
                task_type = TaskType.KIS

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
                task_type = TaskType.KIS # Fallback

        if task_type == TaskType.QNA:
            if self.openai_handler:
                candidates = self.semantic_searcher.search(
                    query_text=search_context,
                    precomputed_analysis=query_analysis,
                    top_k_final=vqa_candidates_to_rank,
                    top_k_retrieval=vqa_retrieval
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
                top_k_retrieval=kis_retrieval
            )
        return {
            "task_type": task_type,
            "results": final_results,
            "query_analysis": query_analysis
        }