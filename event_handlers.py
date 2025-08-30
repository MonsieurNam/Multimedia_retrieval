# /AIC25_Video_Search_Engine/event_handlers.py

import gradio as gr
import pandas as pd
import numpy as np
import time
import os
import traceback
from typing import Dict, Any, List

# Local imports
from config import ITEMS_PER_PAGE, MAX_SUBMISSION_RESULTS
from ui_helpers import create_detailed_info_html, format_submission_list_for_display, encode_image_to_base64
from search_core.task_analyzer import TaskType
from utils.formatting import format_list_for_submission, format_results_for_mute_gallery
from utils import create_video_segment, generate_submission_file

# --- HÀM XỬ LÝ SỰ KIỆN TÌM KIẾM CHÍNH ---

def perform_search(
    # Inputs từ UI
    query_text: str, num_results: int, kis_retrieval: int, vqa_candidates: int,
    vqa_retrieval: int, trake_candidates_per_step: int, trake_max_sequences: int,
    w_clip: float, w_obj: float, w_semantic: float, lambda_mmr: float,
    # Instance backend được truyền vào
    master_searcher
):
    """
    Hàm chính xử lý sự kiện tìm kiếm. Đây là một generator để cập nhật UI từng bước.
    """
    # === BƯỚC 1: VALIDATE INPUT & KHỞI TẠO BIẾN ===
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        status_msg = "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn và bấm Tìm kiếm.</div>"
        return [], status_msg, None, "", "", [], 1, "Trang 1 / 1"

    # === BƯỚC 2: YIELD TRẠNG THÁI "ĐANG XỬ LÝ" ===
    loading_html = """
    <div style="display: flex; align-items: center; gap: 15px; padding: 10px; background-color: #e0e7ff; border-radius: 8px;">
        <span style="font-weight: 500; color: #4338ca;">⏳ Đang xử lý... AI đang phân tích và tìm kiếm.</span>
    </div>
    """
    yield ([], loading_html, None, "", "", [], 1, "Trang 1 / 1")
    
    # === BƯỚC 3: GỌI BACKEND & XỬ LÝ LỖI ===
    try:
        config = {
            "top_k_final": int(num_results), "kis_retrieval": int(kis_retrieval),
            "vqa_candidates": int(vqa_candidates), "vqa_retrieval": int(vqa_retrieval),
            "trake_candidates_per_step": int(trake_candidates_per_step),
            "trake_max_sequences": int(trake_max_sequences), "w_clip": w_clip,
            "w_obj": w_obj, "w_semantic": w_semantic, "lambda_mmr": lambda_mmr
        }
        start_time = time.time()
        full_response = master_searcher.search(query=query_text, config=config)
        search_time = time.time() - start_time
    except Exception as e:
        traceback.print_exc()
        status_msg = f"<div style='color: red;'>🔥 Đã xảy ra lỗi backend nghiêm trọng: {e}</div>"
        return [], status_msg, None, "", "", [], 1, "Trang 1 / 1"

    # === BƯỚC 4: ĐỊNH DẠNG KẾT QUẢ & CẬP NHẬT UI CUỐI CÙNG ===
    gallery_paths = format_results_for_mute_gallery(full_response)
    response_state = full_response
    num_found = len(gallery_paths)
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    
    if num_found == 0:
        status_msg = f"<div style='color: #d97706;'>😔 **{task_type_msg}** | Không tìm thấy kết quả nào ({search_time:.2f}s).</div>"
    else:
        status_msg = f"<div style='color: #166534;'>✅ **{task_type_msg}** | Tìm thấy {num_found} kết quả ({search_time:.2f}s).</div>"

    query_analysis = full_response.get('query_analysis', {})
    analysis_html = f"<div><strong>Phân tích AI:</strong> <em>{query_analysis.get('search_context', 'N/A')}</em></div>"
    stats_info_html = f"<div><strong>Thời gian:</strong> {search_time:.2f}s | <strong>Kết quả:</strong> {num_found}</div>"
    
    initial_gallery_view = gallery_paths[:ITEMS_PER_PAGE]
    total_pages = int(np.ceil(num_found / ITEMS_PER_PAGE)) or 1
    page_info = f"Trang 1 / {total_pages}"
    
    yield (
        initial_gallery_view, status_msg, response_state, analysis_html,
        stats_info_html, gallery_paths, 1, page_info
    )

# --- CÁC HÀM XỬ LÝ PHÂN TRANG & GALLERY ---

def update_gallery_page(gallery_items: List, current_page: int, direction: str):
    """Cập nhật trang hiển thị của gallery kết quả."""
    if not gallery_items:
        return [], 1, "Trang 1 / 1"

    total_items = len(gallery_items)
    total_pages = int(np.ceil(total_items / ITEMS_PER_PAGE)) or 1
    
    new_page = current_page
    if direction == "▶️ Trang sau":
        new_page = min(total_pages, current_page + 1)
    elif direction == "◀️ Trang trước":
        new_page = max(1, current_page - 1)

    start_index = (new_page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    
    new_gallery_view = gallery_items[start_index:end_index]
    page_info = f"Trang {new_page} / {total_pages}"
    
    return new_gallery_view, new_page, page_info

def on_gallery_select(response_state: Dict[str, Any], current_page: int, evt: gr.SelectData):
    """Xử lý khi người dùng click vào một ảnh trong gallery chính."""
    if not response_state or evt is None:
        gr.Warning("Vui lòng thực hiện tìm kiếm trước khi chọn.")
        return None, None, pd.DataFrame(), "", "", None, "", ""

    results = response_state.get("results", [])
    task_type = response_state.get("task_type")
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    
    if not results or global_index >= len(results):
        gr.Error("Lỗi dữ liệu không đồng bộ. Vui lòng thử tìm kiếm lại.")
        return None, None, pd.DataFrame(), "", "", None, "", ""

    selected_result = results[global_index]
    
    # Nhánh 1: Xử lý kết quả chuỗi TRAKE
    if task_type == TaskType.TRAKE:
        sequence = selected_result.get('sequence', [])
        if not sequence: return None, None, pd.DataFrame(), "", "", None, "Lỗi: Chuỗi TRAKE rỗng.", ""
        
        target_frame = sequence[0] # Lấy frame đầu làm đại diện
        html_output = f"<h4>Chuỗi hành động ({len(sequence)} bước)</h4>"
        # (code tạo HTML chi tiết cho chuỗi TRAKE có thể thêm ở đây)
        video_clip_path = create_video_segment(target_frame.get('video_path'), target_frame.get('timestamp'))
        clip_info_html = "Clip đại diện cho chuỗi hành động."
        return target_frame.get('keyframe_path'), video_clip_path, pd.DataFrame(), "", "", selected_result, html_output, clip_info_html

    # Nhánh 2: Xử lý kết quả đơn lẻ (KIS và QNA)
    else:
        video_path = selected_result.get('video_path')
        timestamp = selected_result.get('timestamp')
        
        selected_image_path = selected_result.get('keyframe_path')
        video_clip_path = create_video_segment(video_path, timestamp)

        scores = selected_result.get('scores', {})
        scores_data = {"Metric": list(scores.keys()), "Value": list(scores.values())}
        scores_df = pd.DataFrame(scores_data)
        
        vqa_answer = selected_result.get('answer', "") if task_type == TaskType.QNA else "Không áp dụng cho tác vụ KIS."
        transcript = selected_result.get('transcript_text', "Không có transcript.")
        detailed_info_html = create_detailed_info_html(selected_result, task_type)
        clip_info_html = f"Clip 10 giây từ <strong>{os.path.basename(video_path or 'N/A')}</strong>"

        return (selected_image_path, video_clip_path, scores_df, vqa_answer, transcript, 
                selected_result, detailed_info_html, clip_info_html)

# --- CÁC HÀM XỬ LÝ VÙNG NỘP BÀI ---

def add_to_submission_list(
    submission_list: list, candidate: Dict[str, Any], 
    response_state: Dict[str, Any], position: str
):
    """Thêm một ứng viên vào danh sách nộp bài."""
    if not candidate:
        gr.Warning("Chưa có ứng viên nào được chọn để thêm!")
        choices = [f"{i+1}. ..." for i, item in enumerate(submission_list)]
        # === FIX HERE: Bỏ `.update` ===
        return format_submission_list_for_display(submission_list), submission_list, gr.Dropdown(choices=choices)

    task_type = response_state.get("task_type")
    item_to_add = candidate.copy()
    item_to_add['task_type'] = task_type
    
    if len(submission_list) >= MAX_SUBMISSION_RESULTS:
        gr.Warning(f"Danh sách đã đạt giới hạn {MAX_SUBMISSION_RESULTS} kết quả.")
        submission_list = submission_list[:MAX_SUBMISSION_RESULTS]
    else:
        if position == 'top':
            submission_list.insert(0, item_to_add)
        else:
            submission_list.append(item_to_add)
        gr.Success(f"Đã thêm kết quả vào {'đầu' if position == 'top' else 'cuối'} danh sách!")

    new_choices = [
        f"{i+1}. {item.get('keyframe_id') or 'TRAKE (' + str(item.get('video_id')) + ')'}" 
        for i, item in enumerate(submission_list)
    ]
    # === FIX HERE: Bỏ `.update` ===
    return format_submission_list_for_display(submission_list), submission_list, gr.Dropdown(choices=new_choices, value=None)

def clear_submission_list():
    """Xóa toàn bộ danh sách nộp bài."""
    gr.Info("Đã xóa danh sách nộp bài.")
    # === FIX HERE: Bỏ `.update` ===
    return "Chưa có kết quả nào được thêm vào.", [], gr.Dropdown(choices=[], value=None)

def modify_submission_list(
    submission_list: list, selected_item_index_str: str, action: str
):
    """Sửa đổi danh sách nộp bài (di chuyển, xóa)."""
    if not selected_item_index_str:
        gr.Warning("Vui lòng chọn một mục từ danh sách để thao tác.")
        return format_submission_list_for_display(submission_list), submission_list, selected_item_index_str

    try:
        index = int(selected_item_index_str.split('.')[0]) - 1
        if not (0 <= index < len(submission_list)): raise ValueError("Index out of bounds")
    except:
        gr.Error("Lựa chọn không hợp lệ.")
        return format_submission_list_for_display(submission_list), submission_list, None

    if action == 'move_up' and index > 0:
        submission_list[index], submission_list[index-1] = submission_list[index-1], submission_list[index]
    elif action == 'move_down' and index < len(submission_list) - 1:
        submission_list[index], submission_list[index+1] = submission_list[index+1], submission_list[index]
    elif action == 'remove':
        submission_list.pop(index)

    new_choices = [
        f"{i+1}. {item.get('keyframe_id') or 'TRAKE (' + str(item.get('video_id')) + ')'}" 
        for i, item in enumerate(submission_list)
    ]
    # === FIX HERE: Bỏ `.update` ===
    return format_submission_list_for_display(submission_list), submission_list, gr.Dropdown(choices=new_choices, value=None)

def handle_submission(submission_list: list, query_id: str):
    """Tạo và cung cấp file nộp bài định dạng CSV."""
    if not submission_list:
        gr.Warning("Danh sách nộp bài đang trống.")
        return None
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
        
    submission_df = format_list_for_submission(submission_list, max_results=MAX_SUBMISSION_RESULTS)
    if submission_df.empty:
        gr.Warning("Không thể định dạng kết quả để nộp bài.")
        return None
              
    file_path = generate_submission_file(submission_df, query_id=query_id)
    gr.Info(f"Đã tạo file nộp bài thành công: {os.path.basename(file_path)}")
    return file_path

# --- HÀM TIỆN ÍCH KHÁC ---

def clear_all():
    """Reset toàn bộ giao diện về trạng thái ban đầu."""
    return (
        # Cột trái
        [],                                         # results_gallery
        "",                                         # status_output
        None,                                       # response_state
        "",                                         # gemini_analysis
        "",                                         # stats_info
        [],                                         # gallery_items_state
        1,                                          # current_page_state
        "Trang 1 / 1",                              # page_info_display
        # Cột phải - Trạm phân tích
        None,                                       # selected_image_display
        None,                                       # video_player
        pd.DataFrame(),                             # scores_display
        "",                                         # vqa_answer_display
        "",                                         # transcript_display
        None,                                       # selected_candidate_for_submission
        "",                                         # detailed_info
        "",                                         # clip_info
        # Cột phải - Vùng nộp bài
        "Chưa có kết quả nào được thêm vào.",       # submission_list_display
        [],                                         # submission_list_state
        # === FIX HERE: Bỏ `.update` ===
        gr.Dropdown(choices=[], value=None),        # submission_list_selector
        "",                                         # query_id_input
        None,                                       # submission_file_output
    )