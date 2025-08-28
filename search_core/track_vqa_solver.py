from typing import List, Dict, Any

# Import các thành phần cần thiết để type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from search_core.openai_handler import OpenAIHandler
    from search_core.semantic_searcher import SemanticSearcher
    from search_core.gemini_text_handler import GeminiTextHandler

import time

class TrackVQASolver:
    """
    Class xử lý tác vụ "Theo dõi và Tổng hợp Thông tin" (Track-VQA).

    Đây là một pipeline phức tạp bao gồm các bước:
    1.  Truy xuất hàng loạt các khoảnh khắc ứng viên dựa trên bối cảnh.
    2.  (Tương lai) Lọc và gom cụm để tìm các khoảnh khắc độc nhất.
    3.  Thực hiện VQA trên từng khoảnh khắc để thu thập "bằng chứng".
    4.  Sử dụng AI để tổng hợp tất cả bằng chứng thành một câu trả lời cuối cùng.
    """

    def __init__(self, 
                 text_handler: 'GeminiTextHandler', 
                 vision_handler: 'OpenAIHandler', 
                 semantic_searcher: 'SemanticSearcher'):
        """
        Khởi tạo TrackVQASolver.

        Args:
            text_handler (GeminiTextHandler): Handler để thực hiện các tác vụ text (phân tích).
            vision_handler (OpenAIHandler): Handler để thực hiện các tác vụ vision (VQA lặp lại).
            semantic_searcher (SemanticSearcher): Searcher để truy xuất các khoảnh khắc.
        """
        # Đổi tên self.ai_handler thành các tên cụ thể hơn
        self.text_handler = text_handler
        self.vision_handler = vision_handler
        self.searcher = semantic_searcher


    def solve(self, query_analysis: Dict[str, Any], candidates_to_retrieve: int, candidates_to_analyze: int) -> Dict[str, Any]:
        """
        Thực thi toàn bộ pipeline Track-VQA.
        """
        print("--- 🔬 Bắt đầu pipeline Track-VQA ---")
        
        # --- 1. Truy xuất các khoảnh khắc ứng viên ---
        search_context = query_analysis.get('search_context')
        if not search_context:
            return {"final_answer": "Lỗi: Không xác định được bối cảnh tìm kiếm.", "evidence_frames": []}

        print(f"   -> 1/4: Đang truy xuất {candidates_to_retrieve} khoảnh khắc cho context: '{search_context}'")
        
        # Sử dụng các tham số từ config
        candidates = self.searcher.search(
            query_text=search_context, 
            precomputed_analysis=query_analysis,
            top_k_final=candidates_to_retrieve, # Lấy số lượng lớn ban đầu
            top_k_retrieval=candidates_to_retrieve # Có thể đặt bằng nhau hoặc tinh chỉnh thêm
        )
        if not candidates:
             return {"final_answer": "Không tìm thấy khoảnh khắc nào phù hợp.", "evidence_frames": []}

        # --- 2. Lọc và Gom cụm ---
        # Lấy số lượng ứng viên để phân tích từ config
        moments_to_analyze = candidates[:candidates_to_analyze]
        print(f"   -> 2/4: Đã chọn ra {len(moments_to_analyze)} khoảnh khắc hàng đầu để thực hiện VQA.")

        # --- 3. Thực hiện VQA lặp lại trên từng khoảnh khắc ---
        specific_question = query_analysis.get('specific_question')
        if not specific_question:
            return {"final_answer": "Lỗi: Không xác định được câu hỏi cụ thể để phân tích.", "evidence_frames": []}
            
        print(f"   -> 3/4: Đang thực hiện VQA lặp lại với câu hỏi: '{specific_question}'")
        successful_observations = [] 
        for moment in moments_to_analyze:
            # Chúng ta có thể thêm một khoảng chờ nhỏ ở đây để tránh rate limit
            time.sleep(0.5) 
            vqa_result = self.vision_handler.perform_vqa(moment['keyframe_path'], specific_question)
            print(f"     -> Phản hồi API: {vqa_result}")

            # Lọc câu trả lời dựa trên độ tự tin
            if vqa_result and vqa_result.get('confidence', 0) > 0.6:
                answer_text = vqa_result.get('answer', '')
                successful_observations.append({
                    "answer": vqa_result.get('answer', ''),
                    "frame_info": moment 
                })
                print(f"     -> ✅ Kết quả được chấp nhận (Conf > 0.6): '{answer_text}'")
            else:
                print(f"     -> ❌ Kết quả bị loại bỏ (Conf <= 0.6 hoặc lỗi).")

        if not successful_observations:
            return {"final_answer": "Không thể thu thập đủ thông tin từ các khoảnh khắc.", "evidence_frames": []}
        # Tách riêng danh sách câu trả lời để tổng hợp
        instance_answers = [obs['answer'] for obs in successful_observations]
        
        # --- 4. Tổng hợp các câu trả lời thành một kết quả duy nhất ---
        aggregation_instruction = query_analysis.get('aggregation_instruction')
        if not aggregation_instruction:
            return {"final_answer": "Lỗi: Không xác định được cách tổng hợp kết quả.", "evidence_frames": []}

        print(f"   -> 4/4: Đang tổng hợp {len(instance_answers)} câu trả lời...")
        final_answer = self.aggregate_answers(instance_answers, aggregation_instruction)
        
        evidence_frames = [obs['frame_info'] for obs in successful_observations]
        
        print(f"--- ✅ Pipeline Track-VQA hoàn tất! ---")
        
        return {
            "final_answer": final_answer,
            "evidence_frames": evidence_frames # Trả về các frame đã thực sự đóng góp vào câu trả lời
        }

    def aggregate_answers(self, answers: List[str], instruction: str) -> str:
        """
        Sử dụng AI để tổng hợp một danh sách các câu trả lời riêng lẻ.
        Đây là bước suy luận cuối cùng.

        Args:
            answers (List[str]): Danh sách các câu trả lời từ VQA trên từng frame.
            instruction (str): Hướng dẫn cách tổng hợp (ví dụ: "đếm và liệt kê màu").

        Returns:
            str: Câu trả lời tổng hợp cuối cùng.
        """
        # Chuyển danh sách thành một chuỗi dễ đọc cho AI
        formatted_answers = "\n".join([f"- {ans}" for ans in answers])
        
        prompt = f"""
        You are a data synthesis AI. Your task is to analyze a list of individual observations from different video frames and synthesize them into a single, final answer based on the user's ultimate goal. Be concise and directly answer the user's original intent.

        **Individual Observations:**
        {formatted_answers}

        **User's Final Goal:**
        "{instruction}"

        **Synthesized Final Answer (in Vietnamese):**
        """
        
        try:
            response = self.text_handler._gemini_text_call(prompt)
            return response.text
        except Exception as e:
            print(f"--- ⚠️ Lỗi khi tổng hợp câu trả lời: {e} ---")
            return "Không thể tổng hợp. Các quan sát riêng lẻ: " + ", ".join(answers)