import re
from enum import Enum
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional

# Định nghĩa các loại nhiệm vụ bằng Enum để code rõ ràng và ít lỗi hơn
class TaskType(Enum):
    """
    Enum để đại diện cho các loại nhiệm vụ tìm kiếm khác nhau.
    """
    KIS = "Textual KIS"
    QNA = "Question Answering"
    TRAKE = "Action Keyframe Tracking"

def analyze_query_heuristic(query: str) -> TaskType:
    """
    Phân loại truy vấn bằng phương pháp heuristic dựa trên regex và từ khóa.

    Đây là phương pháp nhanh, không cần API call, và hoạt động tốt cho các trường hợp rõ ràng.
    Nó đóng vai trò là một cơ chế dự phòng tin cậy.

    Args:
        query (str): Câu truy vấn của người dùng.

    Returns:
        TaskType: Loại nhiệm vụ được phân loại (KIS, QNA, hoặc TRAKE).
    """
    if not isinstance(query, str) or not query:
        return TaskType.KIS # Mặc định cho input không hợp lệ

    query_lower = query.lower().strip()

    # --- Heuristics cho Nhiệm vụ Q&A (Question Answering) ---
    # Các từ để hỏi thường xuất hiện ở đầu câu
    qna_start_keywords = [
        'màu gì', 'ai là', 'ai đang', 'ở đâu', 'khi nào', 'tại sao',
        'cái gì', 'bao nhiêu', 'có bao nhiêu', 'hành động gì', 'đang làm gì'
    ]
    # Dấu hỏi "?" là một dấu hiệu mạnh mẽ
    if '?' in query or any(query_lower.startswith(k) for k in qna_start_keywords):
        return TaskType.QNA

    # --- Heuristics cho Nhiệm vụ TRAKE (Action Keyframe Tracking) ---
    # Các dấu hiệu của một chuỗi hành động: có đánh số, hoặc các từ khóa về chuỗi
    trake_keywords = ['tìm các khoảnh khắc', 'tìm những khoảnh khắc', 'chuỗi hành động', 'các bước']
    # Regex để tìm các mẫu như (1), (2) hoặc 1., 2.
    trake_pattern = r'\(\d+\)|bước \d+|\d\.'
    if any(k in query_lower for k in trake_keywords) or re.search(trake_pattern, query_lower):
        return TaskType.TRAKE

    # --- Mặc định là KIS (Knowledge Intensive Search) ---
    # Nếu không rơi vào các trường hợp trên, đây là một truy vấn tìm kiếm khoảnh khắc đơn lẻ
    return TaskType.KIS

def analyze_query_gemini(query: str, model: Optional[genai.GenerativeModel] = None) -> TaskType:
    """
    Phân loại truy vấn bằng mô hình Gemini để có độ chính xác cao hơn với các câu phức tạp.

    Args:
        query (str): Câu truy vấn của người dùng.
        model (genai.GenerativeModel, optional): Instance của Gemini model đã được khởi tạo.
                                                 Nếu là None, sẽ fallback về heuristic.

    Returns:
        TaskType: Loại nhiệm vụ được phân loại.
    """
    if not model:
        return analyze_query_heuristic(query)
    
    # Prompt được thiết kế để yêu cầu Gemini trả về một câu trả lời ngắn gọn, có cấu trúc.
    # Việc đưa ra ví dụ (few-shot prompting) giúp mô hình hiểu rõ yêu cầu hơn.
    prompt = f"""
    Analyze the following Vietnamese user query for a video search system. Classify it into one of three types:
    1.  "KIS": The user is looking for a single, specific moment or scene. (e.g., "a man opening a laptop", "a red car on the street")
    2.  "QNA": The user is asking a direct question about a scene that needs an answer. (e.g., "What color is the woman's dress?", "Who is speaking on the stage?")
    3.  "TRAKE": The user is looking for a sequence of multiple distinct moments in order. (e.g., "Find the moments of: (1) jumping, (2) landing", "a person stands up, walks, and then sits down")

    Return ONLY the type as a single word: KIS, QNA, or TRAKE.

    Query: "{query}"
    Type:
    """
    
    # Cấu hình an toàn để tránh bị block bởi các bộ lọc không cần thiết
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        # .strip() để loại bỏ khoảng trắng thừa, .upper() để so sánh không phân biệt hoa thường
        result = response.text.strip().upper()
        
        if "QNA" in result:
            return TaskType.QNA
        if "TRAKE" in result:
            return TaskType.TRAKE
        # Kể cả khi Gemini trả về một câu dài, nếu có chứa từ khóa thì vẫn nhận dạng được
        # Nếu không, mặc định là KIS.
        return TaskType.KIS
        
    except Exception as e:
        print(f"--- ⚠️ Lỗi khi gọi API phân loại của Gemini: {e}. Sử dụng fallback heuristic. ---")
        # Nếu có bất kỳ lỗi nào xảy ra với API, hệ thống vẫn hoạt động nhờ heuristic
        return analyze_query_heuristic(query)
