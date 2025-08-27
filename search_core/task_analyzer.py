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

# --- Block để kiểm thử nhanh (sẽ được chuyển vào file test riêng) ---
if __name__ == '__main__':
    print("--- Chạy kiểm thử cho Module Phân loại Truy vấn ---")

    test_queries = {
        # KIS Cases
        "một người đàn ông mặc áo xanh đang đi bộ": TaskType.KIS,
        "cảnh hoàng hôn trên bãi biển": TaskType.KIS,
        "xe cứu hỏa đang phun nước": TaskType.KIS,
        
        # QNA Cases
        "Người phụ nữ trong video đang mặc váy màu gì?": TaskType.QNA,
        "Có bao nhiêu người trên sân khấu": TaskType.QNA,
        "What color is the bus?": TaskType.QNA, # Test với câu hỏi tiếng Anh
        "Ai đang phát biểu vậy": TaskType.QNA,

        # TRAKE Cases
        "Tìm các khoảnh khắc: (1) một người chạy tới, (2) nhảy lên, (3) tiếp đất": TaskType.TRAKE,
        "Một chiếc ô tô bắt đầu di chuyển, tăng tốc, và sau đó dừng lại": TaskType.TRAKE,
        "Tìm chuỗi hành động của đầu bếp: bước 1. thái rau, bước 2. cho vào chảo": TaskType.TRAKE
    }

    print("\n--- Kiểm thử với Phương pháp Heuristic ---")
    correct_heuristic = 0
    for query, expected in test_queries.items():
        result = analyze_query_heuristic(query)
        is_correct = result == expected
        if is_correct:
            correct_heuristic += 1
        print(f"Query: '{query}'\n -> Expected: {expected.name}, Got: {result.name} {'✅' if is_correct else '❌'}")
    
    print(f"\n>>> Kết quả Heuristic: {correct_heuristic}/{len(test_queries)} đúng.")

    # --- Kiểm thử với Gemini (chỉ chạy nếu có API Key) ---
    # Bạn cần thay thế bằng cách khởi tạo model thật trong môi trường Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model_for_test = genai.GenerativeModel('gemini-2.5-flash')
        
        print("\n--- Kiểm thử với Phương pháp Gemini (cần API) ---")
        correct_gemini = 0
        for query, expected in test_queries.items():
            result = analyze_query_gemini(query, model=gemini_model_for_test)
            is_correct = result == expected
            if is_correct:
                correct_gemini += 1
            print(f"Query: '{query}'\n -> Expected: {expected.name}, Got: {result.name} {'✅' if is_correct else '❌'}")

        print(f"\n>>> Kết quả Gemini: {correct_gemini}/{len(test_queries)} đúng.")

    except Exception as e:
        print(f"\n--- Bỏ qua kiểm thử Gemini: Không thể khởi tạo model. Lỗi: {e} ---")
