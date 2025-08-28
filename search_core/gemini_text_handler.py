import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, List
import json
import re

# Tái sử dụng decorator retrier của chúng ta
from utils import api_retrier

class GeminiTextHandler:
    """
    Một class chuyên dụng để xử lý TẤT CẢ các tác vụ liên quan đến văn bản
    bằng API của Google Gemini (cụ thể là model Flash).
    
    Bao gồm: phân loại tác vụ, phân tích chi tiết truy vấn, và phân rã truy vấn TRAKE.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Khởi tạo Gemini Text Handler.

        Args:
            api_key (str): Google API Key.
            model_name (str): Tên model Gemini sẽ sử dụng.
        """
        print(f"--- ✨ Khởi tạo Gemini Text Handler với model: {model_name} ---")
        # Khởi tạo client riêng để quản lý kết nối
        # Lưu ý: genai.configure là lệnh global, chỉ cần gọi một lần
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.health_check() # Thực hiện health check ngay khi khởi tạo
            print("--- ✅ Gemini Text Handler đã được khởi tạo và xác thực thành công! ---")
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi khởi tạo Gemini Text Handler: {e} ---")
            # Ném lại lỗi để MasterSearcher có thể bắt và vô hiệu hóa các tính năng liên quan
            raise e

    @api_retrier(max_retries=3, initial_delay=1)
    def _gemini_text_call(self, prompt: str):
        """Hàm con được "trang trí", chỉ để thực hiện lệnh gọi API text của Gemini."""
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = self.model.generate_content(prompt, safety_settings=safety_settings)
        return response

    def health_check(self):
        """Thực hiện một lệnh gọi API đơn giản để kiểm tra key và kết nối."""
        print("--- 🩺 Đang thực hiện kiểm tra trạng thái API Gemini... ---")
        try:
            self.model.count_tokens("kiểm tra")
            print("--- ✅ Trạng thái API Gemini: OK ---")
            return True
        except Exception as e:
            print(f"--- ❌ Lỗi API Gemini: {e} ---")
            # Ném lỗi để __init__ có thể bắt được
            raise e

    def analyze_task_type(self, query: str) -> str:
        """Phân loại truy vấn bằng Gemini, sử dụng prompt có Quy tắc Ưu tiên."""
        prompt = f"""
        You are a highly precise query classifier. Classify a Vietnamese query into "KIS", "QNA", "TRAKE", or "TRACK_VQA". You MUST follow a strict priority order.

        **Priority Order for Classification (Check from top to bottom):**
        1. **Check for TRACK_VQA first:** ... (Dán prompt `analyze_task_type` hoàn chỉnh đã sửa ở bước trước vào đây)
        2. **Then, check for TRAKE:** ...
        3. **Then, check for QNA:** ...
        4. **Default to KIS:** ...

        **Query:** "{query}"
        **Category:**
        """
        try:
            response = self._gemini_text_call(prompt)
            task_type = response.text.strip().upper()
            if task_type in ["KIS", "QNA", "TRAKE", "TRACK_VQA"]:
                return task_type
            return "KIS"
        except Exception:
            return "KIS" # Fallback an toàn

    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Phân tích và trích xuất thông tin truy vấn bằng Gemini."""
        fallback_result = {
            'search_context': query, 'specific_question': "", 'aggregation_instruction': "",
            'objects_vi': [], 'objects_en': []
        }
        prompt = f"""
        Analyze a Vietnamese user query. Return ONLY a valid JSON object.
        **JSON Structure to return:**
        - "search_context": ...
        - "specific_question": ... (Dán prompt `enhance_query` hoàn chỉnh đã sửa ở bước trước vào đây)
        - "aggregation_instruction": ...
        - "objects_vi" / "objects_en": ...

        **Query:** "{query}"
        **JSON:**
        """
        try:
            response = self._gemini_text_call(prompt)
            # Trích xuất JSON từ markdown block (Gemini thường trả về như vậy)
            match = re.search(r"```json\s*(\{.*?\})\s*```", response.text, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*?\})", response.text, re.DOTALL) # Thử tìm JSON không có markdown
            
            if match:
                result = json.loads(match.group(1))
                # Validate ...
                return result
            return fallback_result
        except Exception as e:
            print(f"Lỗi Gemini enhance_query: {e}")
            return fallback_result
            
    def decompose_trake_query(self, query: str) -> List[str]:
        """Phân rã truy vấn TRAKE bằng Gemini."""
        prompt = f"""
        Decompose the Vietnamese query describing a sequence of actions into a JSON array of short, self-contained phrases. Return ONLY the JSON array.

        Example:
        Query: "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
        JSON: ["vận động viên giậm nhảy", "vận động viên bay qua xà", "vận động viên tiếp đất", "vận động viên đứng dậy"]

        Query: "{query}"
        JSON:
        """
        try:
            response = self._gemini_text_call(prompt)
            match = re.search(r"\[.*?\]", response.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return [query]
        except Exception:
            return [query]