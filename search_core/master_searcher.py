# search_core/master_searcher.py

from typing import Dict, Any, Optional, List
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import các module cốt lõi của hệ thống
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.trake_solver import TRAKESolver
from search_core.gemini_text_handler import GeminiTextHandler
from search_core.openai_handler import OpenAIHandler
from search_core.task_analyzer import TaskType
from search_core.mmr_builder import MMRResultBuilder 


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
                 entities_path: str = None,
                clip_features_path: str = None): 
        """
        Khởi tạo MasterSearcher và hệ sinh thái AI lai.
        """
        print("--- 🧠 Khởi tạo Master Searcher (Hybrid AI Edition) ---")
        
        self.semantic_searcher = SemanticSearcher(basic_searcher=basic_searcher)
        self.mmr_builder: Optional[MMRResultBuilder] = None
        if clip_features_path and os.path.exists(clip_features_path):
            try:
                all_clip_features = basic_searcher.get_all_clip_features() 
                self.mmr_builder = MMRResultBuilder(clip_features=all_clip_features)
            except Exception as e:
                 print(f"--- ⚠️ Lỗi khi khởi tạo MMR Builder: {e}. MMR sẽ bị vô hiệu hóa. ---")
        else:
            print("--- ⚠️ Không tìm thấy file CLIP features, MMR sẽ không hoạt động. ---")

        print(f"--- ✅ Master Searcher đã sẵn sàng! (AI Enabled: {self.ai_enabled}) ---")
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
                print(f"--- ⚠️ Lỗi khi tải Từ điển Đối tượng: {e}. Semantic Grounding sẽ bị vô hiệu hóa. ---")
                
        # --- Khởi tạo và xác thực Gemini Handler cho các tác vụ TEXT ---
        if gemini_api_key:
            try:
                self.gemini_handler = GeminiTextHandler(api_key=gemini_api_key)
                if self.known_entities and self.gemini_handler:
                    self.gemini_handler.load_known_entities(self.known_entities)
                self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo Gemini Handler: {e}. Các tính năng text AI sẽ bị hạn chế. ---")

        # --- Khởi tạo và xác thực OpenAI Handler cho các tác vụ VISION ---
        if openai_api_key:
            try:
                self.openai_handler = OpenAIHandler(api_key=openai_api_key)
                if not self.openai_handler.check_api_health():
                    self.openai_handler = None
                else:
                    self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo OpenAI Handler: {e}. Các tính năng vision AI sẽ bị hạn chế. ---")
        
        # --- Khởi tạo các Solver phức tạp nếu các handler cần thiết đã sẵn sàng ---
        if self.gemini_handler:
            self.trake_solver = TRAKESolver(ai_handler=self.gemini_handler)

        print(f"--- ✅ Master Searcher đã sẵn sàng! (AI Enabled: {self.ai_enabled}) ---")

    def search(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hàm tìm kiếm chính, nhận một dictionary config để tùy chỉnh hành vi.
        """
        # --- Bước 1: Giải nén Config ---
        top_k_final = int(config.get('top_k_final', 100))
        kis_retrieval = int(config.get('kis_retrieval', 200))
        vqa_candidates_to_rank = int(config.get('vqa_candidates', 20))
        vqa_retrieval = int(config.get('vqa_retrieval', 200))
        trake_candidates_per_step = int(config.get('trake_candidates_per_step', 20))
        trake_max_sequences = int(config.get('trake_max_sequences', 50))
        w_clip = config.get('w_clip', 0.4)
        w_obj = config.get('w_obj', 0.3)
        w_semantic = config.get('w_semantic', 0.3)
        lambda_mmr = config.get('lambda_mmr', 0.7)

        # --- Bước 2: Phân tích Truy vấn ---
        query_analysis = {}
        task_type = TaskType.KIS
        if self.ai_enabled and self.gemini_handler:
            print("--- ✨ Bắt đầu phân tích truy vấn bằng Gemini Text Handler... ---")
            query_analysis = self.gemini_handler.analyze_query_fully(query)
            
            original_objects = query_analysis.get('objects_en', [])
            if original_objects:
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
        query_analysis.update({'w_clip': w_clip, 'w_obj': w_obj, 'w_semantic': w_semantic})
        search_context = query_analysis.get('search_context', query)

        # --- Bước 3: Khối Điều phối Logic ---

        if task_type == TaskType.TRAKE:
            if self.trake_solver:
                sub_queries = self.trake_solver.decompose_query(query)
                final_results = self.trake_solver.find_sequences(
                    sub_queries, 
                    self.semantic_searcher,
                    original_query_analysis=query_analysis,
                    top_k_per_step=trake_candidates_per_step,
                    max_sequences=trake_max_sequences
                )
            else:
                task_type = TaskType.KIS

        elif task_type == TaskType.QNA:
            if self.openai_handler:
                candidates = self.semantic_searcher.search(
                    query_text=search_context,
                    precomputed_analysis=query_analysis,
                    top_k_final=vqa_retrieval,
                    top_k_retrieval=vqa_retrieval
                )
                
                if not candidates:
                    final_results = []
                else:
                    candidates_for_vqa = candidates[:vqa_candidates_to_rank]
                    specific_question = query_analysis.get('specific_question', query)
                    vqa_enhanced_candidates = []
                    
                    print(f"--- 💬 Bắt đầu Quét VQA song song trên {len(candidates_for_vqa)} ứng viên... ---")
                    
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        future_to_candidate = {
                            executor.submit(
                                self.openai_handler.perform_vqa, 
                                image_path=cand['keyframe_path'], 
                                question=specific_question, 
                                context_text=cand.get('transcript_text', '')
                            ): cand 
                            for cand in candidates_for_vqa
                        }
                        
                        for future in tqdm(as_completed(future_to_candidate), total=len(candidates_for_vqa), desc="   -> VQA Progress"):
                            cand = future_to_candidate[future]
                            try:
                                vqa_result = future.result()
                                new_cand = cand.copy()
                                new_cand['answer'] = vqa_result['answer']
                                search_score = new_cand.get('final_score', 0)
                                vqa_confidence = vqa_result.get('confidence', 0)
                                new_cand['final_score'] = search_score * vqa_confidence
                                new_cand['scores']['vqa_confidence'] = vqa_confidence
                                vqa_enhanced_candidates.append(new_cand)
                            except Exception as exc:
                                print(f"--- ❌ Lỗi khi xử lý VQA cho keyframe {cand.get('keyframe_id')}: {exc} ---")
                    
                    if vqa_enhanced_candidates:
                        final_results = sorted(vqa_enhanced_candidates, key=lambda x: x['final_score'], reverse=True)
                    else:
                        final_results = []
            else:
                print("--- ⚠️ OpenAI (VQA) handler chưa được kích hoạt. Fallback về KIS. ---")
                task_type = TaskType.KIS

        if not final_results or task_type == TaskType.KIS:
            final_results = self.semantic_searcher.search(
                query_text=search_context,
                precomputed_analysis=query_analysis,
                top_k_final=kis_retrieval, 
                top_k_retrieval=kis_retrieval
            )
        # --- BƯỚC 4: ÁP DỤNG MMR ĐỂ TĂNG CƯỜNG ĐA DẠNG ---
        diverse_results = final_results
        if self.mmr_builder and final_results:
            # Lưu ý quan trọng: TRAKE trả về danh sách các chuỗi, không phải các frame đơn lẻ.
            # MMR chỉ nên áp dụng cho KIS và QNA.
            if task_type in [TaskType.KIS, TaskType.QNA]:
                diverse_results = self.mmr_builder.build_diverse_list(
                    candidates=final_results, 
                    target_size=len(final_results), # MMR sẽ sắp xếp lại toàn bộ list
                    lambda_val=lambda_mmr
                )

        # Tạm thời chỉ cắt bớt
        final_results_for_submission = diverse_results[:top_k_final]

        # *** LOG DEBUG ĐIỂM A ***
        print("\n" + "="*20 + " DEBUG LOG: MASTER SEARCHER OUTPUT " + "="*20)
        print(f"-> Task Type cuối cùng: {task_type.value}")
        print(f"-> Số lượng kết quả cuối cùng: {len(final_results)}")
        if final_results:
            print("-> Ví dụ kết quả đầu tiên:")
            first_result = final_results[0]
            if task_type == TaskType.TRAKE:
                print(f"  - video_id: {first_result.get('video_id')}")
                print(f"  - final_score: {first_result.get('final_score')}")
                print(f"  - Số bước trong chuỗi: {len(first_result.get('sequence', []))}")
            else: # KIS, QNA
                print(f"  - keyframe_id: {first_result.get('keyframe_id')}")
                print(f"  - final_score: {first_result.get('final_score')}")
                if 'answer' in first_result:
                    print(f"  - answer: {first_result.get('answer')}")
        else:
            print("-> Không có kết quả nào được tạo ra.")
        print("="*68 + "\n")
        
        return {
            "task_type": task_type,
            "results": final_results_for_submission,
            "query_analysis": query_analysis
        }