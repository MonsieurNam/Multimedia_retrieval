# /kaggle/working/search_core/openai_handler.py

import openai
import json
import re
import base64
from typing import Dict, Any, List

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
    def _openai_chat_completion(self, messages: List[Dict], is_json: bool = True, is_vision: bool = False):
        """
        Hàm con chung để thực hiện các lệnh gọi API chat completion.

        Args:
            messages (List[Dict]): Danh sách các message theo định dạng OpenAI.
            is_json (bool): True nếu muốn model trả về JSON object.
            is_vision (bool): True nếu đây là một lệnh gọi có hình ảnh.
        """
        model_to_use = self.vision_model if is_vision else self.model
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        
        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            response_format=response_format,
            temperature=0.1, # Giảm nhiệt độ để kết quả nhất quán
            max_tokens=1024  # Giới hạn token để tránh chi phí không mong muốn
        )
        return response.choices[0].message.content

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
        Analyze a Vietnamese user query for a video search system. Return ONLY a valid JSON object with: "search_context", "specific_question", "objects_vi", and "objects_en".

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
        """Phân loại truy vấn thành 'KIS', 'QNA', hoặc 'TRAKE'."""
        prompt = f"""
        Classify the Vietnamese query into "KIS", "QNA", or "TRAKE". Return ONLY the type as a single word.
        - "KIS": Looking for a single scene.
        - "QNA": Asking a question about a scene.
        - "TRAKE": Looking for a sequence of actions.
        Query: "{query}"
        Type:
        """
        try:
            response = self._openai_chat_completion([{"role": "user", "content": prompt}], is_json=False)
            task_type = response.strip().upper()
            if task_type in ["KIS", "QNA", "TRAKE"]:
                return task_type
            return "KIS" # Fallback an toàn
        except Exception as e:
            print(f"Lỗi OpenAI analyze_task_type: {e}")
            return "KIS"

    def perform_vqa(self, image_path: str, question: str) -> Dict[str, any]:
        """
        Thực hiện VQA sử dụng GPT-4o.
        """
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            return {"answer": "Lỗi: Không thể xử lý ảnh", "confidence": 0.0}

        prompt = f"""
        You are a Visual Question Answering assistant. Based on the provided image, answer the user's question.
        Return ONLY a valid JSON object with two keys: "answer" and "confidence".
        - "answer": A short, direct answer in Vietnamese.
        - "confidence": Your confidence in the answer, from 0.0 (very unsure) to 1.0 (certain).

        Question: "{question}"
        JSON:
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
            result = json.loads(response_content)
            return {
                "answer": result.get("answer", "Không có câu trả lời"),
                "confidence": float(result.get("confidence", 0.5))
            }
        except Exception as e:
            print(f"Lỗi OpenAI perform_vqa: {e}")
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