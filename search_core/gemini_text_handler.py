import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, List, Set
import json
import re

from utils import api_retrier

class GeminiTextHandler:
    """
    Một class chuyên dụng để xử lý TẤT CẢ các tác vụ liên quan đến văn bản
    bằng API của Google Gemini (cụ thể là model Flash).
    
    Bao gồm: phân loại tác vụ, phân tích chi tiết truy vấn, và phân rã truy vấn TRAKE.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Khởi tạo Gemini Text Handler.

        Args:
            api_key (str): Google API Key.
            model_name (str): Tên model Gemini sẽ sử dụng.
        """
        print(f"--- ✨ Khởi tạo Gemini Text Handler với model: {model_name} ---")
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.known_entities_prompt_segment = ""
            self.health_check() # Thực hiện health check ngay khi khởi tạo
            print("--- ✅ Gemini Text Handler đã được khởi tạo và xác thực thành công! ---")
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi khởi tạo Gemini Text Handler: {e} ---")
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
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = self.model.generate_content(prompt, safety_settings=safety_settings, generation_config=generation_config)
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
            raise e

    def analyze_task_type(self, query: str) -> str:
        """Phân loại truy vấn bằng Gemini, sử dụng prompt có Quy tắc Ưu tiên."""
        prompt = f"""
        You are a highly precise query classifier. Your task is to classify a Vietnamese query into one of four categories: TRACK_VQA, TRAKE, QNA, or KIS. You MUST follow a strict priority order.

        **Priority Order for Classification (Check from top to bottom):**

        1.  **Check for TRACK_VQA first:** Does the query ask a question about a COLLECTION of items, requiring aggregation (counting, listing, summarizing)? Look for keywords like "đếm", "bao nhiêu", "liệt kê", "tất cả", "mỗi", or plural subjects. If it matches, classify as **TRACK_VQA** and stop.
            - Example: "trong buổi trình diễn múa lân, đếm xem có bao nhiêu con lân" -> This is a request to count a collection, so it is **TRACK_VQA**.

        2.  **Then, check for TRAKE:** If it's not TRACK_VQA, does the query ask for a SEQUENCE of DIFFERENT, ordered actions? Look for patterns like "(1)...(2)...", "bước 1... bước 2", "sau đó". If it matches, classify as **TRAKE** and stop.
            - Example: "người đàn ông đứng lên rồi bước đi"

        3.  **Then, check for QNA:** If not TRACK_VQA or TRAKE, does the query ask a **direct question** that expects a factual answer about something in the scene? This is more than just describing a scene. Look for:
            - **Interrogative words:** "ai", "cái gì", "ở đâu", "khi nào", "tại sao", "như thế nào", "màu gì", "hãng nào", etc.
            - **Question structures:** "có phải là...", "đang làm gì", "là ai", "trông như thế nào".
            - A question mark "?".
            If it matches, classify as **QNA** and stop.
            - Example: "người phụ nữ mặc áo màu gì?" -> QNA
            - Example: "ai là người đàn ông đang phát biểu?" -> QNA
            - Example: "có bao nhiêu chiếc xe trên đường?" -> This asks for a count of a single scene, so it is **QNA**. (Distinguish this from TRACK_VQA which counts across multiple scenes/videos).

        4.  **Default to KIS:** If the query is a statement or a descriptive phrase looking for a moment, classify as **KIS**. It describes "what to find", not "what to answer".
            - Example: "cảnh người đàn ông đang phát biểu" -> KIS
            - Example: "tìm người phụ nữ mặc áo đỏ" -> KIS
            - Example: "một chiếc xe đang chạy" -> KIS

        **Your Task:**
        Follow the priority order strictly. Analyze the query below and return ONLY the final category as a single word.

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

    def analyze_query_fully(self, query: str) -> Dict[str, Any]:
        """
        Thực hiện phân tích toàn diện một truy vấn, bao gồm cả phân loại tác vụ,
        trong MỘT lần gọi API duy nhất, yêu cầu output dạng JSON.
        """
        fallback_result = {
            'task_type': 'KIS', 'search_context': query, 'specific_question': "",
            'aggregation_instruction': "", 'objects_vi': [], 'objects_en': []
        }
        
        prompt = f"""
        You are a master Vietnamese query analyzer for a video search system. Analyze the user's query and return ONLY a single, valid JSON object with ALL the following keys: "task_type", "search_context", "specific_question", "aggregation_instruction", "objects_vi", "objects_en".

        **Analysis Steps & Rules:**

        1.  **Determine `task_type` FIRST (Strict Priority):**
            - **TRACK_VQA:** Does it ask to aggregate (count, list, summarize) info about a COLLECTION of items across scenes? (e.g., "đếm số lân", "liệt kê tất cả các xe"). If yes, `task_type` is "TRACK_VQA".
            - **TRAKE:** If not, does it ask for a SEQUENCE of distinct actions? (e.g., "đứng lên rồi đi ra"). If yes, `task_type` is "TRAKE".
            - **QNA:** If not, is it a DIRECT QUESTION expecting a factual answer about a single scene? (e.g., "ai là người...", "màu gì?", "đang làm gì?"). If yes, `task_type` is "QNA".
            - **KIS:** Otherwise, it's a descriptive search. `task_type` is "KIS".

        2.  **Fill other keys based on `task_type`:**
            - `search_context`: The general scene to search for. ALWAYS FILL THIS.
            - `specific_question`: The specific question for a vision model. ONLY for QNA and TRACK_VQA.
            - `aggregation_instruction`: The final goal. ONLY for TRACK_VQA.
            - `objects_vi` & `objects_en`: Key nouns/entities from the query.

        **Example 1 (QNA):**
        Query: "ai là người đàn ông đội mũ đỏ đang phát biểu ở mỹ"
        JSON: {{"task_type": "QNA", "search_context": "cảnh người đàn ông đội mũ đỏ đang phát biểu ở Mỹ", "specific_question": "ai là người đàn ông này?", "aggregation_instruction": "", "objects_vi": ["người đàn ông", "mũ đỏ", "phát biểu", "Mỹ"], "objects_en": ["man", "red hat", "speaking", "USA"]}}

        **Example 2 (KIS):**
        Query: "cảnh người đàn ông đội mũ đỏ phát biểu ở mỹ"
        JSON: {{"task_type": "KIS", "search_context": "cảnh người đàn ông đội mũ đỏ phát biểu ở Mỹ", "specific_question": "", "aggregation_instruction": "", "objects_vi": ["người đàn ông", "mũ đỏ", "phát biểu", "Mỹ"], "objects_en": ["man", "red hat", "speaking", "USA"]}}
        ---
        **Your Task:**
        Analyze the query below and generate the required JSON object.

        **Query:** "{query}"
        **JSON:**
        """
        try:
            response = self._gemini_text_call(prompt)
            raw_text = response.text
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            json_string = match.group(1) if match else raw_text
            
            result = json.loads(json_string)
            if 'task_type' in result and 'search_context' in result:
                return result
            # Nếu JSON hợp lệ nhưng thiếu key, trả về fallback nhưng vẫn giữ lại những gì có
            return {**fallback_result, **result}

        except Exception as e:
            print(f"Lỗi Gemini analyze_query_fully: {e}. Response: '{getattr(response, 'text', 'N/A')}'")
            return fallback_result

    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Phân tích và trích xuất thông tin truy vấn bằng Gemini."""
        fallback_result = {
            'search_context': query, 'specific_question': "", 'aggregation_instruction': "",
            'objects_vi': [], 'objects_en': []
        }
        prompt = f"""
        Analyze a Vietnamese user query for a video search system. **Return ONLY a valid JSON object** with five keys: "search_context", "specific_question", "aggregation_instruction", "objects_vi", and "objects_en".

        **Rules:**
        1.  `search_context`: A Vietnamese phrase for finding the general scene. This is used for vector search.
        2.  `specific_question`: The specific question to ask the Vision model for EACH individual frame.
        3.  `aggregation_instruction`: The final instruction for the AI to synthesize all individual answers. This should reflect the user's ultimate goal (e.g., counting, listing, summarizing).
        4.  `objects_vi`: A list of Vietnamese nouns/entities.
        5.  `objects_en`: The English translation for EACH item in `objects_vi`.

        **Example (VQA):**
        Query: "Trong video quay cảnh bữa tiệc, người phụ nữ mặc váy đỏ đang cầm ly màu gì?"
        JSON: {{"search_context": "cảnh bữa tiệc có người phụ nữ mặc váy đỏ", "specific_question": "cô ấy đang cầm ly màu gì?", "aggregation_instruction": "trả lời câu hỏi người phụ nữ cầm ly màu gì", "objects_vi": ["bữa tiệc", "người phụ nữ", "váy đỏ"], "objects_en": ["party", "woman", "red dress"]}}

        **Example (Track-VQA):**
        Query: "đếm xem có bao nhiêu con lân trong buổi biểu diễn"
        JSON: {{"search_context": "buổi biểu diễn múa lân", "specific_question": "có con lân nào trong ảnh này không và màu gì?", "aggregation_instruction": "từ các quan sát, đếm tổng số lân và liệt kê màu sắc của chúng", "objects_vi": ["con lân", "buổi biểu diễn"], "objects_en": ["lion dance", "performance"]}}

        **Your Task:**
        Analyze the query below and generate the JSON.

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
        
    def load_known_entities(self, known_entities: Set[str]):
        """
        Chuẩn bị và cache lại phần prompt chứa từ điển đối tượng.
        Chỉ cần gọi một lần khi MasterSearcher khởi tạo.
        """
        if not known_entities:
            print("--- ⚠️ Từ điển đối tượng rỗng. Semantic Grounding sẽ không hoạt động. ---")
            return
        
        # Sắp xếp để đảm bảo prompt nhất quán giữa các lần chạy
        sorted_entities = sorted(list(known_entities))
        # Định dạng thành chuỗi JSON để nhúng vào prompt
        self.known_entities_prompt_segment = json.dumps(sorted_entities)
        print(f"--- ✅ GeminiTextHandler: Đã nạp {len(sorted_entities)} thực thể vào bộ nhớ prompt. ---")

    def perform_semantic_grounding(self, user_objects: List[str]) -> List[str]:
        """
        Sử dụng Gemini để "dịch" các đối tượng của người dùng sang các thực thể đã biết.
        """
        if not self.known_entities_prompt_segment or not user_objects:
            return user_objects # Trả về như cũ nếu không có từ điển hoặc không có object

        prompt = f"""
        You are an AI assistant for a video search engine. Your task is to map a list of user-provided objects to a predefined list of "known entities".

        Rules:
        1. For each object in the user's list, find the BEST, most specific, single corresponding entity from the "Known Entities" list.
        2. If an object is a specific type of a known entity, map it to the general entity (e.g., "poodle" -> "dog", "lamborghini" -> "car").
        3. If an object is already a known entity, keep it as is.
        4. If an object has NO reasonable mapping in the known list (e.g., "love", "happiness"), discard it.
        5. The final list should not have duplicate entities.
        6. Return ONLY a valid JSON array containing the final, mapped entities. Your entire response must be a single JSON array.

        **Known Entities:**
        {self.known_entities_prompt_segment}

        **Example 1:**
        User Objects: ["lamborghini", "a woman in a red shirt", "poodle"]
        Expected JSON Result: ["car", "woman", "shirt", "dog"]

        **Example 2:**
        User Objects: ["happiness", "a big cat"]
        Expected JSON Result: ["cat"]
        ---
        **Your Task:**
        User Objects: {json.dumps(user_objects)}
        Analyze and produce the JSON Result:
        """
        
        try:
            response = self._gemini_text_call(prompt)
            # Vì đã yêu cầu response_mime_type="application/json", chúng ta có thể parse trực tiếp
            # response.text sẽ là một chuỗi JSON
            # Thêm một lớp bảo vệ để trích xuất JSON từ markdown block nếu có
            raw_text = response.text
            match = re.search(r"```json\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
            json_string = match.group(1) if match else raw_text

            mapped_objects = json.loads(json_string)
            
            if isinstance(mapped_objects, list):
                # Kiểm tra thêm để đảm bảo các phần tử là string
                if all(isinstance(item, str) for item in mapped_objects):
                    return mapped_objects
            
            print(f"--- ⚠️ Semantic Grounding: Kết quả trả về không phải là một list of strings. Fallback. Got: {mapped_objects} ---")
            return user_objects

        except (json.JSONDecodeError, ValueError) as e:
             print(f"--- ⚠️ Semantic Grounding: Không thể parse JSON từ Gemini. Lỗi: {e}. Response: '{response.text}' ---")
             return user_objects
        except Exception as e:
            print(f"--- ❌ Lỗi không xác định trong lúc thực hiện Semantic Grounding: {e} ---")
            return user_objects