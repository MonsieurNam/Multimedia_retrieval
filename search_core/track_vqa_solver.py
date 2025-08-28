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


    def solve(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi toàn bộ pipeline Track-VQA.

        Args:
            query_analysis (Dict[str, Any]): Dictionary kết quả phân tích từ AI Handler.

        Returns:
            Dict[str, Any]: Một dictionary chứa câu trả lời cuối cùng và các frame bằng chứng.
        """
        print("--- 🔬 Bắt đầu pipeline Track-VQA ---")
        
        # --- 1. Truy xuất các khoảnh khắc ứng viên ---
        search_context = query_analysis.get('search_context')
        if not search_context:
            return {"final_answer": "Lỗi: Không xác định được bối cảnh tìm kiếm.", "evidence_frames": []}

        print(f"   -> 1/4: Đang truy xuất các khoảnh khắc cho context: '{search_context}'")
        # Lấy một lượng lớn ứng viên để tăng độ bao phủ
        candidates = self.searcher.search(
            query_text=search_context, 
            precomputed_analysis=query_analysis,
            top_k_final=50, # Lấy 50 ứng viên tốt nhất để phân tích
            top_k_retrieval=300
        )
        if not candidates:
             return {"final_answer": "Không tìm thấy khoảnh khắc nào phù hợp.", "evidence_frames": []}

        # --- 2. Lọc và Gom cụm (TODO trong tương lai) ---
        # Hiện tại, chúng ta sẽ xử lý 10 ứng viên hàng đầu để cân bằng tốc độ và độ chính xác
        moments_to_analyze = candidates[:10]
        print(f"   -> 2/4: Đã chọn ra {len(moments_to_analyze)} khoảnh khắc hàng đầu để thực hiện VQA.")

        # --- 3. Thực hiện VQA lặp lại trên từng khoảnh khắc ---
        specific_question = query_analysis.get('specific_question')
        if not specific_question:
            return {"final_answer": "Lỗi: Không xác định được câu hỏi cụ thể để phân tích.", "evidence_frames": []}
            
        print(f"   -> 3/4: Đang thực hiện VQA lặp lại với câu hỏi: '{specific_question}'")
        instance_answers = []
        for moment in moments_to_analyze:
            # Chúng ta có thể thêm một khoảng chờ nhỏ ở đây để tránh rate limit
            time.sleep(0.5) 
            vqa_result = self.vision_handler.perform_vqa(moment['keyframe_path'], specific_question)
            
            # Chỉ thu thập các câu trả lời có độ tự tin cao để tránh "nhiễu"
            if vqa_result['confidence'] > 0.6:
                instance_answers.append(vqa_result['answer'])

        if not instance_answers:
            return {"final_answer": "Không thể thu thập đủ thông tin từ các khoảnh khắc tìm thấy.", "evidence_frames": candidates[:5]}

        # --- 4. Tổng hợp các câu trả lời thành một kết quả duy nhất ---
        aggregation_instruction = query_analysis.get('aggregation_instruction')
        if not aggregation_instruction:
            return {"final_answer": "Lỗi: Không xác định được cách tổng hợp kết quả.", "evidence_frames": []}

        print(f"   -> 4/4: Đang tổng hợp {len(instance_answers)} câu trả lời...")
        final_answer = self.aggregate_answers(instance_answers, aggregation_instruction)
        
        print(f"--- ✅ Pipeline Track-VQA hoàn tất! ---")
        
        return {
            "final_answer": final_answer,
            "evidence_frames": candidates[:5] # Luôn trả về 5 frame bằng chứng tốt nhất
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
        
        # Gọi API của ai_handler để nhận câu trả lời tổng hợp
        # Ở đây, chúng ta không cần JSON
        try:
            response = self.text_handler._gemini_text_call([{"role": "user", "content": prompt}], is_json=False)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            print(f"--- ⚠️ Lỗi khi tổng hợp câu trả lời: {e} ---")
            return "Không thể tổng hợp kết quả."