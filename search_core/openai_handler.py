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
        
    @api_retrier(max_retries=2, initial_delay=1)
    def check_api_health(self) -> bool:
        """
        Thực hiện một lệnh gọi API đơn giản để kiểm tra xem API key có hợp lệ và hoạt động không.
        
        Sử dụng việc tạo embedding cho một từ ngắn, đây là một API call nhẹ và rẻ.

        Returns:
            bool: True nếu API hoạt động, False nếu không.
        """
        print("--- 🩺 Đang thực hiện kiểm tra trạng thái API OpenAI... ---")
        try:
            # text-embedding-ada-002 hoặc text-embedding-3-small là lựa chọn tốt
            self.client.embeddings.create(
                input="kiểm tra",
                model="text-embedding-3-small"
            )
            print("--- ✅ Trạng thái API OpenAI: OK ---")
            return True
        except openai.AuthenticationError as e:
            # Lỗi này đặc trưng cho API key sai hoặc không hợp lệ
            print(f"--- ❌ Lỗi OpenAI API: Authentication Error. API Key có thể không hợp lệ. Lỗi: {e} ---")
            return False
        except Exception as e:
            # Bắt các lỗi khác (mạng, etc.)
            print(f"--- ❌ Lỗi OpenAI API: Không thể kết nối đến OpenAI. Lỗi: {e} ---")
            return False

    @api_retrier(max_retries=3, initial_delay=2)
    def _openai_vision_call(self, messages: List[Dict], is_json: bool = True, is_vision: bool = False) -> Optional[str]: # Thêm Optional[str]
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
        Phân loại truy vấn với Quy tắc Ưu tiên để xử lý các trường hợp lai.
        """
        prompt = f"""
        You are a highly precise query classifier. Your task is to classify a Vietnamese query into one of four categories: TRACK_VQA, TRAKE, QNA, or KIS. You MUST follow a strict priority order.

        **Priority Order for Classification (Check from top to bottom):**

        1.  **Check for TRACK_VQA first:** Does the query ask a question about a COLLECTION of items, requiring aggregation (counting, listing, summarizing)? Look for keywords like "đếm", "bao nhiêu", "liệt kê", "tất cả", "mỗi", or plural subjects. If it matches, classify as **TRACK_VQA** and stop.
            - Example: "trong buổi trình diễn múa lân, đếm xem có bao nhiêu con lân" -> This is a request to count a collection, so it is **TRACK_VQA**.

        2.  **Then, check for TRAKE:** If it's not TRACK_VQA, does the query ask for a SEQUENCE of DIFFERENT, ordered actions? Look for patterns like "(1)...(2)...", "bước 1... bước 2", "sau đó". If it matches, classify as **TRAKE** and stop.
            - Example: "người đàn ông đứng lên rồi bước đi"

        3.  **Then, check for QNA:** If it's not TRACK_VQA or TRAKE, is it a direct question about a SINGLE item? Look for a question mark "?" or interrogative words like "cái gì", "ai". If it matches, classify as **QNA** and stop.
            - Example: "người phụ nữ mặc áo màu gì?"

        4.  **Default to KIS:** If the query does not meet any of the criteria above, it is a simple description of a scene. Classify as **KIS**.
            - Example: "cảnh múa lân"

        **Your Task:**
        Follow the priority order strictly. Analyze the query below and return ONLY the final category as a single word.

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