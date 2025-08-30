from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
import os
import json

from tqdm import tqdm 
import google.generativeai as genai # Vẫn cần cho type hinting nếu dùng Gemini
from google.api_core import exceptions as google_exceptions

# Import các module cốt lõi của hệ thống
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.trake_solver import TRAKESolver
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
        w_clip = config.get('w_clip', 0.4)
        w_obj = config.get('w_obj', 0.3)
        w_semantic = config.get('w_semantic', 0.3)
        # --- Bước 2: Phân tích Truy vấn ---
        query_analysis = {}
        task_type = TaskType.KIS
        if self.ai_enabled and self.gemini_handler:
            print("--- ✨ Bắt đầu phân tích truy vấn bằng Gemini Text Handler... ---")
            query_analysis = self.gemini_handler.analyze_query_fully(query)
            
            original_objects = query_analysis.get('objects_en', [])
            if original_objects: # Chỉ gọi API nếu có object để xử lý
                grounded_objects = self.gemini_handler.perform_semantic_grounding(original_objects)
                
                if original_objects != grounded_objects:
                     print(f"--- 🧠 Semantic Grounding: {original_objects} -> {grounded_objects} ---")
                
                query_analysis['objects_en'] = grounded_objects
                
            task_type_str = query_analysis.get('task_type', 'KIS').upper()
            try:
                task_type = TaskType[task_type_str]
            except KeyError:
                task_type = TaskType.KIS
        
        print(f"--- Đã phân loại truy vấn là: {task_type.value} ---")

        final_results = []
        query_analysis['w_clip'] = w_clip
        query_analysis['w_obj'] = w_obj
        query_analysis['w_semantic'] = w_semantic
        search_context = query_analysis.get('search_context', query)

        # --- Bước 3: Khối Điều phối Logic (Cập nhật để truyền Config) ---
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

        elif task_type == TaskType.QNA:
            if self.openai_handler:
                # 1. Lấy ra các ứng viên bối cảnh (không đổi)
                candidates = self.semantic_searcher.search(
                    query_text=search_context,
                    precomputed_analysis=query_analysis,
                    # Lấy ra nhiều ứng viên hơn ở bước này, vì VQA sẽ lọc lại
                    top_k_final=vqa_retrieval,
                    top_k_retrieval=vqa_retrieval
                )

                if not candidates:
                    final_results = []
                else:
                    # Chọn ra số lượng ứng viên hàng đầu để phân tích VQA
                    candidates_for_vqa = candidates[:vqa_candidates_to_rank]
                    
                    specific_question = query_analysis.get('specific_question', query)
                    vqa_enhanced_candidates = []
                    
                    print(f"--- 💬 Bắt đầu Quét VQA song song trên {len(candidates_for_vqa)} ứng viên... ---")

                    # 2. Sử dụng ThreadPoolExecutor để xử lý song song
                    with ThreadPoolExecutor(max_workers=8) as executor: # Số worker có thể tinh chỉnh
                        
                        # Tạo một future cho mỗi candidate
                        future_to_candidate = {
                            executor.submit(
                                self.openai_handler.perform_vqa, 
                                image_path=cand['keyframe_path'], 
                                question=specific_question, 
                                context_text=cand.get('transcript_text', '')
                            ): cand 
                            for cand in candidates_for_vqa
                        }
                        
                        # Thu thập kết quả khi chúng hoàn thành
                        for future in tqdm(as_completed(future_to_candidate), total=len(candidates_for_vqa), desc="   -> VQA Progress"):
                            cand = future_to_candidate[future]
                            try:
                                vqa_result = future.result()
                                
                                new_cand = cand.copy()
                                new_cand['answer'] = vqa_result['answer']
                                
                                # Tính điểm kết hợp
                                search_score = new_cand.get('final_score', 0)
                                vqa_confidence = vqa_result.get('confidence', 0)
                                
                                # Công thức điểm mới có trọng số để cân bằng
                                # w_search = 0.6
                                # w_vqa_conf = 0.4
                                # new_cand['final_score'] = (w_search * search_score) + (w_vqa_conf * vqa_confidence)
                                new_cand['final_score'] = search_score * vqa_confidence # Giữ công thức cũ cho đơn giản

                                # Lưu lại điểm thành phần để hiển thị
                                new_cand['scores'] = new_cand.get('scores', {})
                                new_cand['scores']['search_score'] = search_score
                                new_cand['scores']['vqa_confidence'] = vqa_confidence
                                
                                vqa_enhanced_candidates.append(new_cand)
                                
                            except Exception as exc:
                                print(f"--- ❌ Lỗi khi xử lý VQA cho keyframe {cand.get('keyframe_id')}: {exc} ---")
                    
                    # 3. Sắp xếp lại danh sách cuối cùng dựa trên điểm số đã kết hợp
                    if vqa_enhanced_candidates:
                        final_results = sorted(vqa_enhanced_candidates, key=lambda x: x['final_score'], reverse=True)
                    else:
                        final_results = []
            else:
                print("--- ⚠️ OpenAI (VQA) handler chưa được kích hoạt. Fallback về KIS. ---")
                task_type = TaskType.KIS
                final_results = [] # Reset final_results để chạy khối KIS tiếp theo

        if not final_results or task_type == TaskType.KIS:
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