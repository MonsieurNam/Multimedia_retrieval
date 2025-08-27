
import pandas as pd
from typing import List, Dict, Any
import os

# Import TaskType để sử dụng trong type hinting
from ..search_core.task_analyzer import TaskType

def format_results_for_gallery(response: Dict[str, Any]) -> List[tuple]:
    """
    Định dạng kết quả thô từ MasterSearcher thành định dạng phù hợp cho gr.Gallery.

    Args:
        response (Dict[str, Any]): Dictionary kết quả trả về từ MasterSearcher.search().

    Returns:
        List[tuple]: Danh sách các tuple, mỗi tuple chứa (đường_dẫn_ảnh, caption).
    """
    task_type = response.get("task_type")
    results = response.get("results", [])
    
    formatted_gallery = []

    if not results:
        return []

    if task_type == TaskType.KIS or task_type == TaskType.QNA:
        # Đối với KIS và QNA, mỗi kết quả là một keyframe đơn lẻ
        for res in results:
            scores = res.get('scores', {})
            final_score = res.get('final_score', 0)
            
            # Thêm câu trả lời của VQA vào caption nếu có
            answer_text = f"\n💬 Trả lời: {res.get('answer', 'N/A')}" if task_type == TaskType.QNA else ""
            
            caption = (
                f"📹 {res.get('video_id', 'N/A')}\n"
                f"⏰ {res.get('timestamp', 0):.1f}s | 🏆 {final_score:.3f}\n"
                f"📊 C:{scores.get('clip',0):.2f} O:{scores.get('object',0):.2f} S:{scores.get('semantic',0):.2f}"
                f"{answer_text}"
            )
            formatted_gallery.append((res.get('keyframe_path', ''), caption))

    elif task_type == TaskType.TRAKE:
        # Đối với TRAKE, mỗi kết quả là một chuỗi. Ta sẽ hiển thị keyframe đầu tiên của chuỗi.
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
        # Định dạng: <Tên video>,<Số thứ tự của khung hình>
        for res in results:
            # Giả sử `keyframe_id` có dạng 'Lxx_Vxxx_yyy' và yyy là frame_index
            # Hoặc cần một cột `frame_index` riêng trong metadata nếu tên file không phải là index
            try:
                # Cần đảm bảo 'keyframe_id' chứa thông tin frame index
                frame_index = int(res.get('keyframe_id', '').split('_')[-1])
                submission_data.append({
                    'video_id': res.get('video_id'),
                    'frame_index': frame_index
                })
            except (ValueError, IndexError):
                # Bỏ qua nếu keyframe_id không đúng định dạng
                continue

    elif task_type == TaskType.QNA:
        # Định dạng: <Tên video>,<Số thứ tự của khung hình>,<Câu trả lời của bạn>
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
        # Định dạng: <Tên video>,<Khung hình cho khoảnh khắc 1>,<Khung hình cho khoảnh khắc 2>,...
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
                    # Nếu một frame trong chuỗi bị lỗi, đánh dấu là không hợp lệ
                    row[f'frame_moment_{i+1}'] = -1 
            submission_data.append(row)

    # Tạo DataFrame và đảm bảo có đúng số lượng dòng
    if not submission_data:
        return pd.DataFrame() # Trả về DF rỗng nếu không có kết quả

    df = pd.DataFrame(submission_data)
    
    # Nếu kết quả ít hơn max_results, ta cần điền thêm các dòng rỗng hoặc lặp lại kết quả cuối
    # Cách tiếp cận an toàn là chỉ trả về những gì tìm được.
    # Ban tổ chức thường sẽ xử lý các file nộp có ít hơn 100 dòng.
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
    
    # Nộp bài không cần header và index
    df.to_csv(file_path, header=False, index=False)
    
    print(f"--- ✅ Đã tạo file nộp bài tại: {file_path} ---")
    return file_path


# --- Block để kiểm thử nhanh ---
if __name__ == '__main__':
    print("--- Chạy kiểm thử cho Module Formatting ---")

    # --- Dữ liệu giả ---
    mock_kis_response = {
        "task_type": TaskType.KIS,
        "results": [
            {'video_id': 'L01_V001', 'keyframe_id': 'L01_V001_505', 'timestamp': 50.5, 'final_score': 0.9},
            {'video_id': 'L01_V002', 'keyframe_id': 'L01_V002_123', 'timestamp': 12.3, 'final_score': 0.8}
        ]
    }
    mock_qna_response = {
        "task_type": TaskType.QNA,
        "results": [
            {'video_id': 'L05_V005', 'keyframe_id': 'L05_V005_888', 'timestamp': 88.8, 'final_score': 0.95, 'answer': 'màu xanh'}
        ]
    }
    mock_trake_response = {
        "task_type": TaskType.TRAKE,
        "results": [{
            'video_id': 'L10_V010',
            'final_score': 0.88,
            'sequence': [
                {'keyframe_id': 'L10_V010_101', 'timestamp': 10.1},
                {'keyframe_id': 'L10_V010_156', 'timestamp': 15.6},
                {'keyframe_id': 'L10_V010_203', 'timestamp': 20.3},
            ]
        }]
    }

    print("\n--- 1. Kiểm thử format_for_submission ---")
    
    df_kis = format_for_submission(mock_kis_response)
    print("KIS DataFrame:\n", df_kis)
    assert df_kis.shape == (2, 2)
    assert list(df_kis.columns) == ['video_id', 'frame_index']
    assert df_kis.iloc[0]['frame_index'] == 505

    df_qna = format_for_submission(mock_qna_response)
    print("\nQNA DataFrame:\n", df_qna)
    assert df_qna.shape == (1, 3)
    assert df_qna.iloc[0]['answer'] == 'màu xanh'

    df_trake = format_for_submission(mock_trake_response)
    print("\nTRAKE DataFrame:\n", df_trake)
    assert df_trake.shape == (1, 4) # video_id + 3 moments
    assert df_trake.iloc[0]['frame_moment_2'] == 156

    print("\n--- 2. Kiểm thử generate_submission_file ---")
    file_path = generate_submission_file(df_kis, query_id="test_query_01")
    assert os.path.exists(file_path)
    print(f" -> Đã tạo file: {file_path}")

    print("\n✅ Kiểm thử Module Formatting thành công!")
