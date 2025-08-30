
import pandas as pd
from typing import List, Dict, Any
import os

# Import TaskType để sử dụng trong type hinting
from search_core.task_analyzer import TaskType

def format_results_for_gallery(response: Dict[str, Any]) -> List[str]:
    """
    Định dạng kết quả thô thành định dạng cho gr.Gallery (chỉ trả về đường dẫn ảnh).
    PHIÊN BẢN "COCKPIT V3.3"
    """
    results = response.get("results", [])
    task_type = response.get("task_type")
    
    # Logic mới: Chỉ trả về đường dẫn ảnh để UI load nhanh
    gallery_paths = []
    if not results:
        return []

    for res in results:
        keyframe_path = None
        # Đối với TRAKE, lấy ảnh đại diện là frame đầu tiên của chuỗi
        if task_type == TaskType.TRAKE:
            sequence = res.get('sequence', [])
            if sequence:
                keyframe_path = sequence[0].get('keyframe_path')
        # Đối với KIS và QNA, lấy trực tiếp
        else:
            keyframe_path = res.get('keyframe_path')

        if keyframe_path and os.path.isfile(keyframe_path):
            gallery_paths.append(keyframe_path)
            
    return gallery_paths

def format_results_for_mute_gallery(response: Dict[str, Any]) -> List[str]:
    """
    Định dạng kết quả thô CHỈ LẤY ĐƯỜNG DẪN ẢNH cho "Lưới ảnh câm" (Cockpit v3.3).
    """
    # ==============================================================================
    # === DEBUG LOG: KIỂM TRA INPUT ==============================================
    # ==============================================================================
    print("\n" + "="*20 + " DEBUG LOG: format_results_for_mute_gallery " + "="*20)
    print(f"-> Nhận được response với các key: {response.keys() if isinstance(response, dict) else 'Không phải dict'}")
    results = response.get("results", [])
    task_type = response.get("task_type")
    print(f"-> Task Type: {task_type}")
    print(f"-> Số lượng 'results' nhận được: {len(results)}")
    if results:
        print(f"-> Cấu trúc của result đầu tiên: {results[0].keys() if isinstance(results[0], dict) else 'Không phải dict'}")
        # Kiểm tra sự tồn tại của key 'keyframe_path'
        if 'keyframe_path' in results[0]:
             print(f"  -> Key 'keyframe_path' tồn tại. Giá trị: {results[0]['keyframe_path']}")
        else:
             print("  -> 🚨 CẢNH BÁO: Key 'keyframe_path' KHÔNG TỒN TẠI trong result đầu tiên!")
    print("="*75 + "\n")
    # ==============================================================================

    if not results:
        return []
        
    task_type = response.get("task_type")
    
    keyframe_paths = []

    # Với TRAKE, mỗi kết quả là một chuỗi. Ảnh đại diện là frame ĐẦU TIÊN của chuỗi.
    if task_type == TaskType.TRAKE:
        for sequence_result in results:
            sequence = sequence_result.get('sequence', [])
            if sequence: # Đảm bảo chuỗi không rỗng
                first_frame = sequence[0]
                path = first_frame.get('keyframe_path')
                if path and os.path.isfile(path):
                    keyframe_paths.append(path)
    
    # Với KIS và QNA, mỗi kết quả là một frame đơn lẻ.
    else: # Bao gồm KIS, QNA
        for single_frame_result in results:
            path = single_frame_result.get('keyframe_path')
            if path and os.path.isfile(path):
                keyframe_paths.append(path)

    return keyframe_paths

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

def format_list_for_submission(submission_list: List[Dict], max_results: int = 100) -> pd.DataFrame:
    """
    Định dạng một danh sách các dictionary kết quả (đã được người dùng sắp xếp)
    thành DataFrame để nộp bài.
    """
    if not submission_list:
        return pd.DataFrame()
        
    submission_data = []
    # Giả sử task type là giống nhau cho tất cả, lấy của item đầu tiên
    # Điều này hợp lý vì một truy vấn chỉ thuộc một loại
    task_type = submission_list[0].get('task_type')
    
    if task_type == TaskType.KIS:
        for res in submission_list:
            try:
                # keyframe_id có dạng 'Lxx_Vyyy_zzz'
                frame_index = int(res.get('keyframe_id', '').split('_')[-1])
                submission_data.append({
                    'video_id': res.get('video_id'),
                    'frame_index': frame_index
                })
            except (ValueError, IndexError):
                continue

    elif task_type == TaskType.QNA:
        for res in submission_list:
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
        for seq_res in submission_list:
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
        return pd.DataFrame()

    df = pd.DataFrame(submission_data)
    return df.head(max_results)