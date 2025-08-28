
import pandas as pd
from typing import List, Dict, Any
import os

# Import TaskType để sử dụng trong type hinting
from search_core.task_analyzer import TaskType

def format_results_for_gallery(response: Dict[str, Any]) -> List[tuple]:
    """
    Định dạng kết quả thô thành định dạng phù hợp cho gr.Gallery.
    *** PHIÊN BẢN CÓ HIỂN THỊ VQA TỰ ĐỘNG ***
    """
    task_type = response.get("task_type")
    results = response.get("results", [])
    
    formatted_gallery = []

    if not results:
        return []

    if task_type == TaskType.KIS or task_type == TaskType.QNA:
        for res in results:
            scores = res.get('scores', {})
            final_score = res.get('final_score', 0)
            
            answer_text = ""
            if task_type == TaskType.QNA:
                answer = res.get('answer', '...')
                short_answer = (answer[:30] + '...') if len(answer) > 33 else answer
                answer_text = f"\n💬 Đáp: {short_answer}"
            
            caption = (
                f"📹 {res.get('video_id', 'N/A')}\n"
                f"⏰ {res.get('timestamp', 0):.1f}s | 🏆 {final_score:.3f}"
                f"{answer_text}" # Thêm câu trả lời vào đây
            )
            formatted_gallery.append((res.get('keyframe_path', ''), caption))

    elif task_type == TaskType.TRAKE:
        for i, seq_res in enumerate(results):
            sequence = seq_res.get('sequence', [])
            if not sequence:
                continue
            
            first_frame = sequence[0]
            final_score = seq_res.get('final_score', 0)
            
            caption = (
                f"🎬 Chuỗi #{i+1} | Video: {seq_res.get('video_id', 'N/A')}\n"
                f"🔢 {len(sequence)} bước | 🏆 Điểm TB: {final_score:.3f}\n"
                f"➡️ Bắt đầu lúc: {first_frame.get('timestamp', 0):.1f}s"
            )
            formatted_gallery.append((first_frame.get('keyframe_path', ''), caption))
        
    elif task_type == TaskType.TRACK_VQA:
        # Kết quả của TRACK_VQA chỉ có một item
        agg_result = results[0] if results else None
        if agg_result and agg_result.get("is_aggregated_result"):
            final_answer = agg_result.get("final_answer", "Không có câu trả lời.")
            # Rút gọn câu trả lời dài
            short_answer = (final_answer[:100] + '...') if len(final_answer) > 103 else final_answer
            
            caption = (
                f"💡 **Kết quả Tổng hợp**\n"
                f"{short_answer}"
            )
            # Dùng ảnh bằng chứng đầu tiên làm ảnh đại diện
            keyframe_path = agg_result.get("keyframe_path", "")
            formatted_gallery.append((keyframe_path, caption))
            return formatted_gallery # Trả về ngay lập tức

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
