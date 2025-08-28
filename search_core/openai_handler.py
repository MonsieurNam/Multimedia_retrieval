# /kaggle/working/search_core/openai_handler.py

import openai
import json
import re
import base64
from typing import Dict, Any, List, Optional

from utils import api_retrier

class OpenAIHandler:
    """
    Một class "adapter" để đóng gói tất cả các lệnh gọi API đến OpenAI.
    Che giấu sự phức tạp của việc gọi API và cung cấp các phương thức
    rõ ràng cho các tác vụ cụ thể (phân tích, VQA, etc.).
    """
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Khởi tạo OpenAI Handler.

        Args:
            api_key (str): OpenAI API key.
            model (str): Tên model mặc định cho các tác vụ text.
                         GPT-4o-mini là một lựa chọn tốt về tốc độ và chi phí.
        """
        print(f"--- 🤖 Khởi tạo OpenAI Handler với model mặc định: {model} ---")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        # GPT-4o là model vision mạnh mẽ nhất hiện tại của OpenAI
        self.vision_model = "gpt-4o"

    @api_retrier(max_retries=3, initial_delay=2)
    def _openai_chat_completion(self, messages: List[Dict], is_json: bool = True, is_vision: bool = False) -> Optional[str]: # Thêm Optional[str]
        """
        Hàm con chung để thực hiện các lệnh gọi API chat completion.
        *** PHIÊN BẢN AN TOÀN HƠN ***
        """
        model_to_use = self.vision_model if is_vision else self.model
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        
        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            response_format=response_format,
            temperature=0.1,
            max_tokens=1024
        )
        
        # --- THÊM KIỂM TRA TẠI ĐÂY ---
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            # Trả về chuỗi rỗng nếu content là None, thay vì trả về chính None
            return content if content is not None else "" 
        
        # Nếu không có choices hoặc message, trả về chuỗi rỗng
        return ""

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Mã hóa một file ảnh thành chuỗi base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"--- ⚠️ Lỗi khi mã hóa ảnh {image_path}: {e} ---")
            return ""

    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Phân tích, tăng cường và dịch truy vấn của người dùng.
        """
        fallback_result = {'search_context': query, 'specific_question': "", 'objects_vi': [], 'objects_en': []}
        prompt = f"""
        Analyze a Vietnamese user query for a video search system. **Return ONLY a valid JSON object** with: "search_context", "specific_question", "objects_vi", and "objects_en".

        Rules:
        - "search_context": A Vietnamese phrase for finding the scene.
        - "specific_question": The specific question. For KIS queries, this is an empty string "".
        - "objects_vi": A list of Vietnamese nouns/entities.
        - "objects_en": The English translation for EACH item in "objects_vi". The two lists must have the same length.

        Example (VQA):
        Query: "Trong video quay cảnh bữa tiệc, người phụ nữ mặc váy đỏ đang cầm ly màu gì?"
        JSON: {{"search_context": "cảnh bữa tiệc có người phụ nữ mặc váy đỏ", "specific_question": "cô ấy đang cầm ly màu gì?", "objects_vi": ["bữa tiệc", "người phụ nữ", "váy đỏ"], "objects_en": ["party", "woman", "red dress"]}}

        Query: "{query}"
        JSON:
        """
        try:
            response_content = self._openai_chat_completion([{"role": "user", "content": prompt}], is_json=True)
            result = json.loads(response_content)
            # Validate...
            if all(k in result for k in ['search_context', 'specific_question', 'objects_vi', 'objects_en']):
                return result
            return fallback_result
        except Exception as e:
            print(f"Lỗi OpenAI enhance_query: {e}")
            return fallback_result

    def analyze_task_type(self, query: str) -> str:
        """
        Phân loại truy vấn thành 'KIS', 'QNA', 'TRAKE', hoặc 'TRACK_VQA' với độ chính xác cao.
        *** PHIÊN BẢN NÂNG CẤP VỚI TRACK_VQA ***
        """
        # Prompt mới, được thiết kế lại hoàn toàn với 4 loại nhiệm vụ
        prompt = f"""
        You are an expert query classifier for a sophisticated video search system. Your task is to analyze a Vietnamese user query and classify it into one of four precise categories: "KIS", "QNA", "TRAKE", or "TRACK_VQA".

        **Category Definitions:**

        1.  **QNA (Question Answering):** The query MUST be a direct question about a SINGLE, specific, implicitly defined scene. The user wants one answer about one moment.
            - Example: "người đàn ông mặc áo gì trong buổi họp báo?" (Asks about one specific man)
            - Example: "What color is the car?"

        2.  **TRAKE (Temporal Alignment):** The query explicitly asks for a an ordered SEQUENCE of SEVERAL DIFFERENT actions or events. It often contains numbers (1), (2) or keywords like "sau đó", "tiếp theo".
            - Example: "tìm cảnh người đàn ông (1) đứng lên, (2) đi ra cửa, và (3) mở cửa"
            - Example: "a car starts, accelerates, then stops"

        3.  **TRACK_VQA (Tracking & VQA):** The query requires two stages: first, FINDING ALL instances of an object/event, and then AGGREGATING information about them (e.g., counting them, listing their colors, summarizing their actions). It's a question about a COLLECTION of things.
            - Keywords: "đếm", "có bao nhiêu", "liệt kê", "tất cả", "những con/cái nào".
            - Example: "đếm xem có bao nhiêu chiếc xe màu đỏ trên đường cao tốc" (Find all red cars, then count)
            - Example: "liệt kê màu sắc của tất cả các con lân trong buổi biểu diễn" (Find all lions, then list their colors)

        4.  **KIS (Knowledge Intensive Search):** The default category. The query is a simple descriptive statement. It is NOT a direct question and NOT a sequence and NOT a request for aggregation.
            - Example: "cảnh múa lân"
            - Example: "người đàn ông phát biểu ở mỹ"

        **Your Task:**
        Analyze the following query and return ONLY the category as a single word: KIS, QNA, TRAKE, or TRACK_VQA.

        **Query:** "{query}"
        **Category:**
        """
        try:
            # Sử dụng hàm chat completion đã có
            response = self._openai_chat_completion([{"role": "user", "content": prompt}], is_json=False)
            task_type = response.strip().upper()
            
            # Kiểm tra xem kết quả trả về có hợp lệ không
            if task_type in ["KIS", "QNA", "TRAKE", "TRACK_VQA"]:
                print(f"--- ✅ Phân loại truy vấn (OpenAI): '{query}' -> {task_type} ---")
                return task_type
                
            print(f"--- ⚠️ Phân loại không hợp lệ từ OpenAI: '{task_type}'. Fallback về Heuristic. ---")
            return self._analyze_query_heuristic_fallback(query) # Vẫn giữ fallback

        except Exception as e:
            print(f"Lỗi OpenAI analyze_task_type: {e}. Fallback về Heuristic.")
            return self._analyze_query_heuristic_fallback(query)

    # Cập nhật hàm fallback heuristic để nó không bao giờ trả về TRACK_VQA
    def _analyze_query_heuristic_fallback(self, query: str) -> str:
        """
        Hàm heuristic dự phòng. Sẽ không phân loại TRACK_VQA, để an toàn.
        """
        query_lower = query.lower().strip()
        qna_keywords = ['màu gì', 'ai là', 'ai đang', 'ở đâu', 'khi nào', 'tại sao', 'cái gì', 'bao nhiêu']
        if '?' in query or any(query_lower.startswith(k) for k in qna_keywords):
            # Lưu ý: "bao nhiêu" có thể là TRACK_VQA, nhưng trong heuristic ta ưu tiên QNA cho an toàn
            return "QNA"
        trake_pattern = r'\(\d+\)|bước \d+|\d\.'
        if re.search(trake_pattern, query_lower) or "tìm các khoảnh khắc" in query_lower:
            return "TRAKE"
        return "KIS"

    def perform_vqa(self, image_path: str, question: str) -> Dict[str, any]:
        """
        Thực hiện VQA sử dụng GPT-4o.
        *** PHIÊN BẢN CÓ XỬ LÝ LỖI TỐT HƠN ***
        """
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            return {"answer": "Lỗi: Không thể xử lý ảnh", "confidence": 0.0}

        prompt = f"""
        You are a Visual Question Answering assistant. Based on the provided image, answer the user's question.
        
        **Your task is to return ONLY a valid JSON object** with two keys: "answer" and "confidence".
        - "answer": A short, direct answer in Vietnamese.
        - "confidence": Your confidence in the answer, from 0.0 (very unsure) to 1.0 (certain).

        **User's Question:** "{question}"
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]
        try:
            response_content = self._openai_chat_completion(messages, is_json=True, is_vision=True)
            
            if not response_content:
                print("--- ⚠️ OpenAI VQA không trả về nội dung. ---")
                return {"answer": "Không thể phân tích hình ảnh", "confidence": 0.1}

            result = json.loads(response_content)
            return {
                "answer": result.get("answer", "Không có câu trả lời"),
                "confidence": float(result.get("confidence", 0.5))
            }
        except (json.JSONDecodeError, TypeError) as e:
             # Bắt cả lỗi TypeError từ json.loads(None) và JSONDecodeError
            print(f"Lỗi OpenAI perform_vqa (JSON parsing): {e}. Response nhận được: '{response_content}'")
            return {"answer": "Lỗi định dạng phản hồi", "confidence": 0.0}
        except Exception as e:
            print(f"Lỗi không xác định trong OpenAI perform_vqa: {e}")
            return {"answer": "Lỗi xử lý VQA", "confidence": 0.0}

    def decompose_trake_query(self, query: str) -> List[str]:
        """Phân rã truy vấn TRAKE thành các bước con."""
        prompt = f"""
        Decompose the Vietnamese query describing a sequence of actions into a JSON array of short, self-contained phrases. Return ONLY the JSON array.

        Example:
        Query: "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
        JSON: ["vận động viên giậm nhảy", "vận động viên bay qua xà", "vận động viên tiếp đất", "vận động viên đứng dậy"]

        Query: "{query}"
        JSON:
        """
        try:
            response_content = self._openai_chat_completion([{"role": "user", "content": prompt}], is_json=True)
            result = json.loads(response_content)
            if isinstance(result, list):
                return result
            return [query] # Fallback
        except Exception as e:
            print(f"Lỗi OpenAI decompose_trake_query: {e}")
            return [query]