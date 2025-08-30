# ==============================================================================
# OPENAI HANDLER - PHIÊN BẢN NÂNG CẤP V2 (DỰA TRÊN BẢN GỐC + CONTEXT-AWARE VQA)
# ==============================================================================
import openai
import json
import re
import base64
from typing import Dict, Any, List, Optional
import io
from PIL import Image
from utils import api_retrier

class OpenAIHandler:
    """
    Một class "adapter" để đóng gói tất cả các lệnh gọi API đến OpenAI.
    Che giấu sự phức tạp của việc gọi API và cung cấp các phương thức
    rõ ràng cho các tác vụ cụ thể (phân tích, VQA, etc.).
    """
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        # --- KHÔNG THAY ĐỔI ---
        print(f"--- 🤖 Khởi tạo OpenAI Handler với model mặc định: {model} ---")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.vision_model = "gpt-4o"
        
    @api_retrier(max_retries=2, initial_delay=1)
    def check_api_health(self) -> bool:
        # --- KHÔNG THAY ĐỔI ---
        print("--- 🩺 Đang thực hiện kiểm tra trạng thái API OpenAI... ---")
        try:
            self.client.embeddings.create(input="kiểm tra", model="text-embedding-3-small")
            print("--- ✅ Trạng thái API OpenAI: OK ---")
            return True
        except openai.AuthenticationError as e:
            print(f"--- ❌ Lỗi OpenAI API: Authentication Error. API Key có thể không hợp lệ. Lỗi: {e} ---")
            return False
        except Exception as e:
            print(f"--- ❌ Lỗi OpenAI API: Không thể kết nối đến OpenAI. Lỗi: {e} ---")
            return False

    @api_retrier(max_retries=3, initial_delay=2)
    def _openai_vision_call(self, messages: List[Dict], is_json: bool = True, is_vision: bool = False) -> str:
        # --- KHÔNG THAY ĐỔI ---
        model_to_use = self.vision_model if is_vision else self.model
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        
        response = self.client.chat.completions.create(
            model=model_to_use, messages=messages, response_format=response_format,
            temperature=0.1, max_tokens=1024
        )
        
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            return content if content is not None else "" 
        
        return ""

    def _preprocess_and_encode_image(
        self, 
        image_path: str, 
        max_size: int = 1024, 
        quality: int = 90
    ) -> str:
        """
        Tiền xử lý ảnh (resize, nén) và mã hóa sang Base64.
        Đây là bước tối quan trọng để đảm bảo độ tin cậy và hiệu suất.
        """
        try:
            with Image.open(image_path) as img:
                # 1. Chuẩn hóa: Chuyển sang định dạng RGB để loại bỏ các kênh màu phức tạp như RGBA hoặc CMYK.
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 2. Resize: Giảm kích thước ảnh xuống một ngưỡng hợp lý trong khi giữ nguyên tỷ lệ.
                #    GPT-4o xử lý tốt nhất với ảnh ~1024px.
                img.thumbnail((max_size, max_size))

                # 3. Nén & Lưu vào bộ nhớ đệm (in-memory buffer)
                #    Không cần lưu ra file tạm, giúp tăng tốc độ.
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                img_bytes = buffer.getvalue()

                # 4. Mã hóa Base64
                return base64.b64encode(img_bytes).decode('utf-8')
        except FileNotFoundError:
            print(f"--- ⚠️ Lỗi khi xử lý ảnh: File không tồn tại tại '{image_path}' ---")
            return ""
        except Exception as e:
            print(f"--- ⚠️ Lỗi khi xử lý ảnh {image_path}: {e} ---")
            return ""

    # === HÀM ĐÃ ĐƯỢC NÂNG CẤP ===
    def perform_vqa(self, image_path: str, question: str, context_text: Optional[str] = None) -> Dict[str, any]:
        """
        Thực hiện VQA sử dụng GPT-4o, có thể nhận thêm bối cảnh từ transcript.
        *** PHIÊN BẢN CÓ XỬ LÝ LỖI TỐT HƠN VÀ BỐI CẢNH MỞ RỘNG ***
        """
        base64_image = self._preprocess_and_encode_image(image_path)
        if not base64_image:
            return {"answer": "Lỗi: Không thể xử lý ảnh", "confidence": 0.0}

        # *** BẮT ĐẦU NÂNG CẤP TẠI ĐÂY ***
        
        context_prompt_part = ""
        has_context = False
        if context_text and context_text.strip():
            has_context = True
            truncated_context = (context_text[:500] + '...') if len(context_text) > 503 else context_text
            context_prompt_part = f"""
        **Additional Context from Transcript (What was said around this moment):**
        ---
        "{truncated_context}"
        ---
        """
        
        # Thêm log mới
        if has_context:
            print(f"   -> 🧠 Thực hiện VQA với Context: '{question}'")
        
        prompt = f"""
        Analyze the image and use the provided transcript context (if any) to answer the question in Vietnamese.
        Return a JSON object with two keys: "answer" (string) and "confidence" (float).
        {context_prompt_part}
        Question: "{question}"
        """
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]}
        ]

        try:
            response_content = self._openai_vision_call(messages, is_json=True, is_vision=True)
            
            if not response_content:
                print("--- ⚠️ OpenAI VQA không trả về nội dung. ---")
                return {"answer": "Không thể phân tích hình ảnh", "confidence": 0.1}

            result = json.loads(response_content)
            return {
                "answer": result.get("answer", "Không có câu trả lời"),
                "confidence": float(result.get("confidence", 0.5))
            }
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Lỗi OpenAI perform_vqa (JSON parsing): {e}. Response nhận được: '{response_content}'")
            return {"answer": "Lỗi định dạng phản hồi", "confidence": 0.0}
        except Exception as e:
            print(f"Lỗi không xác định trong OpenAI perform_vqa: {e}")
            return {"answer": "Lỗi xử lý VQA", "confidence": 0.0}

    def decompose_trake_query(self, query: str) -> List[str]:
        # --- KHÔNG THAY ĐỔI ---
        prompt = f"""
        Decompose the Vietnamese query...
        ...
        """
        try:
            response_content = self._openai_vision_call([{"role": "user", "content": prompt}], is_json=True)
            result = json.loads(response_content)
            if isinstance(result, list):
                return result
            return [query]
        except Exception as e:
            print(f"Lỗi OpenAI decompose_trake_query: {e}")
            return [query]