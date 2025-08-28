
import pandas as pd
from typing import List, Dict, Any
import os

# Import TaskType để sử dụng trong type hinting
from search_core.task_analyzer import TaskType

def format_results_for_gallery(response: Dict[str, Any]) -> List[tuple]:
    """
    Định dạng kết quả thô thành định dạng phù hợp cho gr.Gallery.
    *** PHIÊN BẢN MỚI: Xử lý kết quả tổng hợp TRACK_VQA ***
    """
    task_type = response.get("task_type")
    results = response.get("results", [])
    
    formatted_gallery = []

    if not results:
        return []
        
    # Duyệt qua tất cả các kết quả trả về
    for res in results:
        # --- LOGIC XỬ LÝ MỚI ---
        if res.get("is_aggregated_result"):
            # Nếu đây là kết quả tổng hợp của TRACK_VQA
            final_answer = res.get("final_answer", "Không có câu trả lời.")
            short_answer = (final_answer[:100] + '...') if len(final_answer) > 103 else final_answer
            caption = (f"💡 **Kết quả Phân tích Tổng hợp**\n{short_answer}")
            
            keyframe_path = res.get("keyframe_path") # Lấy đường dẫn ảnh đại diện
            if keyframe_path and os.path.isfile(keyframe_path):
                formatted_gallery.append((keyframe_path, caption))
            else:
                # Nếu không có ảnh bằng chứng, không hiển thị gì trong gallery
                # Thông tin sẽ được hiển thị ở status_output hoặc detailed_info
                pass
        
        elif res.get("video_id"): # Xử lý cho KIS, QNA, TRAKE (kết quả frame đơn lẻ)
            scores = res.get('scores', {})
            final_score = res.get('final_score', 0)
            
            answer_text = ""
            if task_type == TaskType.QNA and "answer" in res:
                answer = res.get('answer', '...')
                short_answer = (answer[:30] + '...') if len(answer) > 33 else answer
                answer_text = f"\n💬 Đáp: {short_answer}"

            caption = (
                f"📹 {res.get('video_id', 'N/A')}\n"
                f"⏰ {res.get('timestamp', 0):.1f}s | 🏆 {final_score:.3f}"
                f"{answer_text}"
            )

            keyframe_path = res.get('keyframe_path')
            if keyframe_path and os.path.isfile(keyframe_path):
                formatted_gallery.append((keyframe_path, caption))

    return formatted_gallery

def format_for_submission(response: Dict[str, Any], max_results: int = 100) -> pd.DataFrame:
    """
    Định dạng kết quả thô thành một DataFrame sẵn sàng để lưu ra file CSV nộp bài.

    Args:
        response (Dict[str, Any]): Dictionary kết quả trả về từ MasterSearcher.search().
        max_results (int): Số lượng dòng tối đa trong file nộp bài.

    Returns:
        pd.DataFrame: DataFrame có các cột phù hợp với yêu cầu của ban tổ chức.
    """
    task_type = response.get("task_type")
    results = response.get("results", [])
    
    submission_data = []

    if task_type == TaskType.KIS:
        for res in results:
            try:
                frame_index = int(res.get('keyframe_id', '').split('_')[-1])
                submission_data.append({
                    'video_id': res.get('video_id'),
                    'frame_index': frame_index
                })
            except (ValueError, IndexError):
                continue

    elif task_type == TaskType.QNA:
        for res in results:
            try:
                frame_index = int(res.get('keyframe_id', '').split('_')[-1])
                submission_data.append({
                    'video_id': res.get('video_id'),
                    'frame_index': frame_index,
                    'answer': res.get('answer', '')
                })
            except (ValueError, IndexError):
                continue
    
    elif task_type == TaskType.TRAKE:
        for seq_res in results:
            sequence = seq_res.get('sequence', [])
            if not sequence:
                continue
            
            row = {'video_id': seq_res.get('video_id')}
            for i, frame in enumerate(sequence):
                try:
                    frame_index = int(frame.get('keyframe_id', '').split('_')[-1])
                    row[f'frame_moment_{i+1}'] = frame_index
                except (ValueError, IndexError):
                    row[f'frame_moment_{i+1}'] = -1 
            submission_data.append(row)

    if not submission_data:
        return pd.DataFrame() # Trả về DF rỗng nếu không có kết quả

    df = pd.DataFrame(submission_data)
    
    return df.head(max_results)

def generate_submission_file(df: pd.DataFrame, query_id: str, output_dir: str = "/kaggle/working/submissions") -> str:
    """
    Lưu DataFrame thành file CSV theo đúng định dạng tên file.

    Args:
        df (pd.DataFrame): DataFrame đã được định dạng để nộp bài.
        query_id (str): ID của câu truy vấn (ví dụ: 'query_01').
        output_dir (str): Thư mục để lưu file.

    Returns:
        str: Đường dẫn đến file CSV đã được tạo.
    """
    if df.empty:
        return "Không có dữ liệu để tạo file."

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{query_id}_submission.csv")
    
    df.to_csv(file_path, header=False, index=False)
    
    print(f"--- ✅ Đã tạo file nộp bài tại: {file_path} ---")
    return file_path
