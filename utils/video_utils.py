import ffmpeg
import os
import time
import shutil
from typing import Optional

def create_video_segment(
    video_path: Optional[str],
    timestamp: float,
    duration: int = 30,
    output_dir: str = "/kaggle/working/temp_clips"
) -> Optional[str]:
    """
    Cắt một đoạn video ngắn từ một file video lớn tại một timestamp cho trước.

    Hàm này sẽ cố gắng cắt video bằng cách sao chép codec (rất nhanh) trước.
    Nếu thất bại (thường do keyframe không align), nó sẽ tự động thử lại bằng cách
    re-encode video (chậm hơn nhưng đáng tin cậy hơn).

    Args:
        video_path (Optional[str]): Đường dẫn đầy đủ đến file video nguồn.
        timestamp (float): Thời điểm (tính bằng giây) làm trung tâm của đoạn clip.
        duration (int): Thời lượng của đoạn clip cắt ra (tính bằng giây).
        output_dir (str): Thư mục để lưu các file clip tạm thời.

    Returns:
        Optional[str]: Đường dẫn đến file video clip đã được tạo, hoặc None nếu có lỗi.
    """
    # --- Bước 1: Validate Input ---
    if not video_path or not isinstance(video_path, str):
        print(f"--- ⚠️ Lỗi Cắt Video: Đường dẫn video không hợp lệ (giá trị là {video_path}). ---")
        return None
        
    if not os.path.exists(video_path):
        print(f"--- ⚠️ Lỗi Cắt Video: File video không tồn tại tại '{video_path}'. ---")
        return None
    
    # --- Bước 2: Chuẩn bị đường dẫn và dọn dẹp ---
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Dọn dẹp các clip cũ để tránh đầy bộ nhớ Kaggle
    cleanup_old_clips(output_dir, max_age_seconds=600) # Xóa clip cũ hơn 10 phút

    # Tính toán thời điểm bắt đầu, đảm bảo không âm
    start_time = max(0, timestamp - (duration / 2))
    
    # Tạo tên file output độc nhất để tránh xung đột
    timestamp_ms = int(time.time() * 1000)
    output_filename = f"clip_{os.path.basename(video_path)}_{timestamp_ms}.mp4"
    output_clip_path = os.path.join(output_dir, output_filename)
    
    print(f"--- 🎬 Bắt đầu tạo clip: Nguồn='{os.path.basename(video_path)}', Time={timestamp:.2f}s, Output='{output_filename}' ---")

    # --- Bước 3: Cố gắng cắt nhanh (Stream Copy) ---
    try:
        print("   -> Thử phương pháp cắt nhanh (copy codec)...")
        (
            ffmpeg
            .input(video_path, ss=start_time)
            .output(output_clip_path, t=duration, c='copy', y=None) # c='copy' là mấu chốt
            .run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
        )
        print("   -> ✅ Cắt nhanh thành công!")
        return output_clip_path
    except ffmpeg.Error as e:
        print(f"   -> ⚠️ Cắt nhanh thất bại. Lỗi FFMPEG: {e.stderr.decode('utf8')}")
        # Lỗi có thể xảy ra nếu start_time không phải là một I-frame.
        # Chúng ta sẽ thử lại bằng cách re-encode.
    
    # --- Bước 4: Fallback - Cắt bằng cách Re-encode ---
    try:
        print("   -> Thử phương pháp cắt bằng re-encode (đáng tin cậy hơn)...")
        (
            ffmpeg
            .input(video_path, ss=start_time)
            .output(output_clip_path, t=duration, y=None) # Bỏ 'c=copy' để ffmpeg tự chọn codec và re-encode
            .run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
        )
        print("   -> ✅ Re-encode thành công!")
        return output_clip_path
    except ffmpeg.Error as e:
        print(f"--- ❌ Lỗi Cắt Video: Cả hai phương pháp đều thất bại. Lỗi FFMPEG cuối cùng: {e.stderr.decode('utf8')} ---")
        return None

def cleanup_old_clips(directory: str, max_age_seconds: int):
    """
    Dọn dẹp các file clip cũ trong một thư mục để giải phóng dung lượng.

    Args:
        directory (str): Thư mục chứa các file clip.
        max_age_seconds (int): Tuổi tối đa của một file (tính bằng giây) trước khi bị xóa.
    """
    try:
        current_time = time.time()
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    print(f"   -> 🧹 Dọn dẹp clip cũ: {filename}")
                    os.remove(file_path)
    except Exception as e:
        print(f"--- ⚠️ Lỗi khi dọn dẹp clip cũ: {e} ---")
