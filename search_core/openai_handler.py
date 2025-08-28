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
        Phân tích truy vấn (KIS, QNA, TRAKE, hoặc TRACK_VQA) và trích xuất các thành phần
        có cấu trúc để hệ thống có thể hành động.
        *** PHIÊN BẢN NÂNG CẤP VỚI TRACK_VQA ***
        """
        # Fallback giờ đây linh hoạt hơn, phù hợp với các loại task khác nhau
        fallback_result = {
            'search_context': query,
            'specific_question': "" if "?" not in query else query,
            'aggregation_instruction': "",
            'objects_vi': query.split(),
            'objects_en': query.split()
        }
        
        if not self.model: # Sửa lại để kiểm tra self.model thay vì self.gemini_model
            print("--- ⚠️ OpenAI model chưa được khởi tạo. Sử dụng fallback cho enhance_query. ---")
            return fallback_result

        # Prompt mới, được thiết kế lại hoàn toàn với cấu trúc JSON mới và 4 loại nhiệm vụ
        prompt = f"""
        You are an expert query analyzer for a sophisticated Vietnamese video search system. Your task is to analyze a user query and break it down into a structured JSON object.

        **JSON Structure to return:**
        - "search_context": (String) The main scene, event, or object to search for. This should be a descriptive phrase.
        - "specific_question": (String) The specific question to ask about an instance. For KIS and TRAKE, this MUST be an empty string "".
        - "aggregation_instruction": (String) A command in English describing how to combine multiple answers. This is ONLY for TRACK_VQA tasks. For all other tasks, it MUST be an empty string "".
        - "objects_vi": (Array of Strings) A list of important Vietnamese nouns/entities from the "search_context".
        - "objects_en": (Array of Strings) The direct English translation for EACH item in "objects_vi". The two lists must have the same length.

        **Detailed Examples:**

        1.  **TRACK_VQA Query:** "trong buổi trình diễn múa lân, đếm xem có bao nhiêu con lân và có màu gì"
            **JSON:** {{
                "search_context": "buổi trình diễn múa lân có các con lân",
                "specific_question": "Con lân trong ảnh này có màu gì?",
                "aggregation_instruction": "Count the unique lions and list their colors from the answers.",
                "objects_vi": ["múa lân", "con lân"],
                "objects_en": ["lion dance", "lion"]
            }}

        2.  **QNA Query:** "Người phụ nữ mặc váy đỏ trong bữa tiệc đang cầm ly màu gì?"
            **JSON:** {{
                "search_context": "bữa tiệc có người phụ nữ mặc váy đỏ",
                "specific_question": "cô ấy đang cầm ly màu gì?",
                "aggregation_instruction": "",
                "objects_vi": ["bữa tiệc", "người phụ nữ", "váy đỏ"],
                "objects_en": ["party", "woman", "red dress"]
            }}

        3.  **TRAKE Query:** "tìm cảnh một người (1) chạy đến, (2) nhảy lên, (3) tiếp đất"
            **JSON:** {{
                "search_context": "một người đang thực hiện cú nhảy",
                "specific_question": "",
                "aggregation_instruction": "",
                "objects_vi": ["người", "cú nhảy"],
                "objects_en": ["person", "jump"]
            }}

        4.  **KIS Query:** "cảnh một chiếc xe buýt màu vàng trên đường phố"
            **JSON:** {{
                "search_context": "cảnh một chiếc xe buýt màu vàng trên đường phố",
                "specific_question": "",
                "aggregation_instruction": "",
                "objects_vi": ["xe buýt màu vàng", "đường phố"],
                "objects_en": ["yellow bus", "street"]
            }}

        **Your Task:**
        Analyze the following query and return ONLY the valid JSON object.

        **Query:** "{query}"
        **JSON:**
        """
        try:
            response_content = self._openai_chat_completion([{"role": "user", "content": prompt}], is_json=True)
            result = json.loads(response_content)
            
            # Validate các trường quan trọng
            if all(k in result for k in ['search_context', 'specific_question', 'aggregation_instruction', 'objects_vi', 'objects_en']):
                print(f"--- ✅ Phân tích truy vấn chi tiết thành công. Context: '{result['search_context']}' ---")
                return result
            
            print("--- ⚠️ JSON chi tiết từ OpenAI không hợp lệ. Sử dụng fallback. ---")
            return fallback_result
        except Exception as e:
            print(f"Lỗi OpenAI enhance_query: {e}")
            return fallback_result

    def analyze_task_type(self, query: str) -> str:
        """
        Phân loại truy vấn thành 'KIS', 'QNA', 'TRAKE', hoặc 'TRACK_VQA' với sự tập trung
        vào việc phân biệt giữa câu hỏi về một đối tượng (QNA) và một tập hợp (TRACK_VQA).
        """
        prompt = f"""
        You are an expert query classifier for a video search system. Your task is to analyze a Vietnamese query and classify it into one of four precise categories: "KIS", "QNA", "TRAKE", or "TRACK_VQA".

        **Core Principle: Singular vs. Plural/Collection**

        1.  **QNA (Question Answering - Singular):** The query asks a question about a SINGLE, implicitly defined subject. The user assumes we can find that one subject and wants a specific detail about IT.
            - **Keywords:** Often uses singular nouns (e.g., "người đàn ông", "chiếc xe").
            - **Test:** Can the question be answered by looking at only ONE frame? If yes, it's likely QNA.
            - **Example:** "người đàn ông đang cầm vật gì trên tay?" (Asks about ONE specific man)
            - **Example:** "What color is THE car?"

        2.  **TRACK_VQA (Tracking & VQA - Plural/Collection):** The query asks a question that requires finding a COLLECTION of objects/events first, and then AGGREGATING information about them.
            - **Keywords:** Often uses plural nouns ("những chiếc xe", "các con lân") or aggregation words ("đếm", "bao nhiêu", "liệt kê", "tất cả", "mỗi").
            - **Test:** Does the user need to see MULTIPLE moments to get the final answer? If yes, it's TRACK_VQA.
            - **Example:** "đếm xem có bao nhiêu chiếc xe màu đỏ" (Must find ALL red cars, then count)
            - **Example:** "liệt kê màu của tất cả các con lân" (Must find ALL lions, then list their colors)

        3.  **TRAKE (Temporal Alignment):** Asks for a SEQUENCE of DIFFERENT actions in a specific order.
            - **Example:** "tìm cảnh người (1) đứng lên và (2) rời đi"

        4.  **KIS (Knowledge Intensive Search):** A simple description of a scene. Not a question, not a sequence, not a request for aggregation.
            - **Example:** "cảnh múa lân"

        **Your Task:**
        Analyze the following query and return ONLY the category as a single word.

        **Query:** "{query}"
        **Category:**
        """
        try:
            response = self._openai_chat_completion([{"role": "user", "content": prompt}], is_json=False)
            task_type = response.strip().upper()
            
            if task_type in ["KIS", "QNA", "TRAKE", "TRACK_VQA"]:
                print(f"--- ✅ Phân loại truy vấn (OpenAI): '{query}' -> {task_type} ---")
                return task_type
                
            print(f"--- ⚠️ Phân loại không hợp lệ từ OpenAI: '{task_type}'. Fallback về Heuristic. ---")
            return self._analyze_query_heuristic_fallback(query)

        except Exception as e:
            print(f"Lỗi OpenAI analyze_task_type: {e}. Fallback về Heuristic.")
            return self._analyze_query_heuristic_fallback(query)

    def _analyze_query_heuristic_fallback(self, query: str) -> str:
        """
        Hàm heuristic dự phòng. Sẽ không phân loại TRACK_VQA, để an toàn.
        """
        query_lower = query.lower().strip()
        
        # Các từ khóa mạnh của TRACK_VQA
        track_vqa_keywords = ["đếm", "bao nhiêu", "liệt kê", "tất cả các", "những con", "những cái"]
        if any(k in query_lower for k in track_vqa_keywords):
            return "TRACK_VQA" # Heuristic có thể thử phân loại TRACK_VQA với các từ khóa mạnh

        # Heuristic cho QNA
        qna_keywords = ['màu gì', 'ai là', 'ai đang', 'ở đâu', 'khi nào', 'tại sao', 'cái gì']
        if '?' in query or any(query_lower.startswith(k) for k in qna_keywords):
            return "QNA"
        
        # Heuristic cho TRAKE
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