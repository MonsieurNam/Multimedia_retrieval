import time
import random
from functools import wraps

def api_retrier(max_retries=5, initial_delay=1, backoff_factor=2, jitter=0.1):
    """
    Một decorator để tự động thử lại các lệnh gọi API Gemini khi gặp lỗi 429.

    Sử dụng thuật toán Exponential Backoff with Jitter để tránh làm quá tải API.

    Args:
        max_retries (int): Số lần thử lại tối đa.
        initial_delay (int): Thời gian chờ ban đầu (giây).
        backoff_factor (float): Hệ số nhân cho thời gian chờ sau mỗi lần thất bại.
        jitter (float): Hệ số ngẫu nhiên để tránh các client thử lại cùng lúc.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Kiểm tra xem có phải lỗi Resource Exhausted không
                    # Lỗi này thường có mã 429 hoặc chứa chuỗi 'Resource has been exhausted'
                    # Hoặc 'rate limit'
                    error_str = str(e).lower()
                    if 'resource has been exhausted' in error_str or 'rate limit' in error_str or '429' in error_str:
                        if i == max_retries - 1:
                            print(f"--- ❌ API Call Thất bại sau {max_retries} lần thử. Bỏ qua. Lỗi: {e} ---")
                            # Trả về một giá trị fallback hợp lý tùy theo hàm gốc
                            # Ở đây ta re-raise lỗi để hàm gọi có thể xử lý fallback
                            raise e

                        # Tính toán thời gian chờ ngẫu nhiên
                        jitter_value = delay * jitter * random.uniform(-1, 1)
                        wait_time = delay + jitter_value
                        
                        print(f"--- ⚠️ API Rate Limit. Thử lại lần {i+1}/{max_retries} sau {wait_time:.2f} giây... ---")
                        time.sleep(wait_time)
                        
                        # Tăng thời gian chờ cho lần thử tiếp theo
                        delay *= backoff_factor
                    else:
                        # Nếu là lỗi khác, không phải rate limit, thì re-raise ngay lập tức
                        raise e
            return None # Fallback nếu vòng lặp kết thúc mà không có return
        return wrapper
    return decorator