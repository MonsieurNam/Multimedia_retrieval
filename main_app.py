print("--- 🚀 Bắt đầu khởi chạy AIC25 Video Search Engine ---")
print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")

import sys
import os
import zipfile

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import gradio as gr
import pandas as pd
import glob
import time
import numpy as np
from kaggle_secrets import UserSecretsClient
from typing import Dict, Any

from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.master_searcher import MasterSearcher
from search_core.task_analyzer import TaskType
from utils.formatting import format_list_for_submission, format_results_for_mute_gallery

from utils import (
    create_video_segment,
    format_results_for_gallery,
    format_for_submission,
    generate_submission_file
)
import base64

def _create_detailed_info_html(result: Dict[str, Any], task_type: TaskType) -> str:
    """
    Hàm phụ trợ tạo mã HTML chi tiết cho một kết quả được chọn.
    *** PHIÊN BẢN CẢI TIẾN ***
    """
    # ... (code tạo progress bar không đổi) ...
    def create_progress_bar(score, color):
        percentage = max(0, min(100, score * 100))
        return f"""<div style='background: #e9ecef; border-radius: 5px; overflow: hidden;'><div style='background: {color}; width: {percentage}%; height: 10px; border-radius: 5px;'></div></div>"""

    video_id = result.get('video_id', 'N/A')
    keyframe_id = result.get('keyframe_id', 'N/A')
    timestamp = result.get('timestamp', 0)
    final_score = result.get('final_score', 0)
    scores = result.get('scores', {})

    # Bảng thông tin cơ bản
    info_html = f"""
    <div style='font-size: 14px; line-height: 1.6; background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>
        <p style='margin: 0;'><strong>📹 Video ID:</strong> <code>{video_id}</code></p>
        <p style='margin: 5px 0 0 0;'><strong>🖼️ Keyframe ID:</strong> <code>{keyframe_id}</code></p>
        <p style='margin: 5px 0 0 0;'><strong>⏰ Timestamp:</strong> <code>{timestamp:.2f}s</code></p>
    </div>
    """

    # Bảng điểm số chi tiết
    scores_html = f"""
    <div style='background-color: #f3f4f6; padding: 15px; border-radius: 8px; margin-top: 15px;'>
        <h4 style='margin: 0 0 15px 0; color: #111827;'>🏆 Bảng điểm</h4>
        <div style='margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                <span><strong>📊 Điểm tổng:</strong></span>
                <span style='font-weight: bold; font-size: 16px;'>{final_score:.4f}</span>
            </div>
            {create_progress_bar(final_score, '#10b981')}
        </div>
        """
    # Thêm các điểm thành phần nếu có
    score_items = [('CLIP', 'clip', '#3b82f6'), ('Object', 'object', '#f97316'), ('Semantic', 'semantic', '#8b5cf6')]
    for name, key, color in score_items:
        if key in scores:
            scores_html += f"""
            <div style='margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                    <span>{name} Score:</span><span>{scores[key]:.3f}</span>
                </div>
                {create_progress_bar(scores[key], color)}
            </div>
            """
    scores_html += "</div>"
    
    return info_html + scores_html

def encode_image_to_base64(image_path: str) -> str:
    """Mã hóa một file ảnh thành chuỗi base64 để nhúng vào HTML."""
    if not image_path or not os.path.isfile(image_path):
        return ""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"--- ⚠️ Lỗi khi mã hóa ảnh {image_path}: {e} ---")
        return ""
    
print("--- Giai đoạn 2/4: Đang cấu hình và khởi tạo Backend...")

def _normalize_item_to_path(item):
    """Gallery item can be 'path' or (path, caption)."""
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    return item

def _build_selected_preview(gallery_items, selected_indices):
    """Build preview list for 'Ảnh đã chọn' from gallery items + selected indices."""
    imgs = []
    for i in sorted(selected_indices or []):
        item = gallery_items[i] if 0 <= i < len(gallery_items or []) else None
        if not item:
            continue
        path = _normalize_item_to_path(item)
        if path:
            imgs.append(path)
    return imgs

try:
    user_secrets = UserSecretsClient()
    OPENAI_API_KEY = user_secrets.get_secret("OPENAI_API_KEY") # <-- Đổi tên biến
    print("--- ✅ Cấu hình OpenAI API Key thành công! ---")
except Exception as e:
    OPENAI_API_KEY = None
    print(f"--- ⚠️ Không tìm thấy OpenAI API Key. Lỗi: {e} ---")
try:
    GEMINI_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    print("--- ✅ Cấu hình GEMINI API Key thành công! ---")
except Exception as e:
    GEMINI_API_KEY = None
    print(f"--- ⚠️ Không tìm thấy GEMINI API Key. Lỗi: {e} ---")

CLIP_FEATURES_PATH = '/kaggle/input/stage1/features.npy' 
FAISS_INDEX_PATH = '/kaggle/input/stage1/faiss.index'
RERANK_METADATA_PATH = '/kaggle/input/stage1/rerank_metadata_ultimate_v5.parquet'
VIDEO_BASE_PATH = "/kaggle/input/aic2025-batch-1-video/"
ALL_ENTITIES_PATH = "/kaggle/input/stage1/all_detection_entities.json"

def initialize_backend():
    """
    Hàm khởi tạo toàn bộ backend theo chuỗi phụ thuộc của OpenAI.
    """
    print("--- Đang khởi tạo các model (quá trình này chỉ chạy một lần)... ---")
    
    # Load video path map
    all_video_files = glob.glob(os.path.join(VIDEO_BASE_PATH, "**", "*.mp4"), recursive=True)
    video_path_map = {os.path.basename(f).replace('.mp4', ''): f for f in all_video_files}

    # Bước 1: Khởi tạo BasicSearcher (không đổi)
    print("   -> 1/2: Khởi tạo BasicSearcher...")
    basic_searcher = BasicSearcher(FAISS_INDEX_PATH, RERANK_METADATA_PATH, video_path_map, clip_features_path=CLIP_FEATURES_PATH)
    
    # Bước 2: Khởi tạo MasterSearcher phiên bản OpenAI
    # MasterSearcher giờ sẽ tự quản lý SemanticSearcher và OpenAIHandler bên trong
    print("   -> 2/2: Khởi tạo MasterSearcher (OpenAI Edition and GEMINI)...")
    master_searcher = MasterSearcher(
            basic_searcher=basic_searcher,
            openai_api_key=OPENAI_API_KEY,
            gemini_api_key=GEMINI_API_KEY,
            entities_path=ALL_ENTITIES_PATH,
            clip_features_path=CLIP_FEATURES_PATH
        )    
    if not master_searcher.mmr_builder:
         print("   -> ⚠️ Cảnh báo: MMR Builder chưa được kích hoạt. Kết quả sẽ không có tính đa dạng.")
    print("--- ✅ Backend đã khởi tạo thành công! ---")
    return master_searcher

master_searcher = initialize_backend()

print("--- Giai đoạn 3/4: Đang định nghĩa các hàm logic cho giao diện...")

ITEMS_PER_PAGE = 20 # 5 cột x 4 hàng

def update_gallery_page(gallery_items, current_page, direction):
    """
    Cập nhật trang hiển thị của gallery.
    """
    if not gallery_items:
        return [], 1, "Trang 1 / 1"

    total_items = len(gallery_items)
    # Tính tổng số trang, đảm bảo ít nhất là 1 trang
    total_pages = int(np.ceil(total_items / ITEMS_PER_PAGE)) or 1
    
    new_page = current_page
    if direction == "▶️ Trang sau":
        new_page = min(total_pages, current_page + 1)
    elif direction == "◀️ Trang trước":
        new_page = max(1, current_page - 1)

    # Tính toán index để cắt danh sách
    start_index = (new_page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    
    new_gallery_view = gallery_items[start_index:end_index]
    
    page_info = f"Trang {new_page} / {total_pages}"
    
    return new_gallery_view, new_page, page_info

def perform_search(
    # --- Inputs từ UI ---
    query_text: str, 
    num_results: int,
    kis_retrieval: int,
    vqa_candidates: int,
    vqa_retrieval: int,
    trake_candidates_per_step: int,
    trake_max_sequences: int,
    w_clip: float, 
    w_obj: float, 
    w_semantic: float,
    lambda_mmr: float
):
    """
    Hàm chính xử lý sự kiện tìm kiếm, phiên bản hoàn thiện và bền bỉ.
    Nó điều phối việc gọi backend, xử lý lỗi, định dạng kết quả, và cập nhật toàn bộ UI.
    *** PHIÊN BẢN FULL FIXED (UnboundLocalError & ValueError) ***
    """
    
    # ==============================================================================
    # === BƯỚC 1: KHỞI TẠO BIẾN & VALIDATE INPUT =================================
    # ==============================================================================
    
    gallery_paths = []
    status_msg = ""
    response_state = None
    analysis_html = ""
    stats_info_html = ""
    gallery_items_state = []
    selected_indices_state = []
    selected_count_md = "Đã chọn: 0"
    selected_preview = []
    current_page = 1
    page_info = "Trang 1 / 1"

    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        status_msg = "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn và bấm Tìm kiếm.</div>"
        # --- SỬA ĐỔI 1: Trả về tuple 11 giá trị ---
        return (gallery_paths, status_msg, response_state, analysis_html, stats_info_html, 
                gallery_items_state, selected_indices_state, selected_count_md, selected_preview,
                current_page, page_info)

    # ==============================================================================
    # === BƯỚC 2: YIELD TRẠNG THÁI "ĐANG XỬ LÝ" ===================================
    # ==============================================================================
    
    status_update = """
    <div style="display: flex; align-items: center; gap: 15px; padding: 10px; background-color: #e0e7ff; border-radius: 8px;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="animation: spin 1s linear infinite;"><path d="M12 2V6" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 18V22" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M4.93 4.93L7.76 7.76" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16.24 16.24L19.07 19.07" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 12H6" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M18 12H22" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M4.93 19.07L7.76 16.24" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16.24 7.76L19.07 4.93" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span style="font-weight: 500; color: #4338ca;">Đang xử lý... AI đang phân tích và tìm kiếm. Quá trình này có thể mất một chút thời gian.</span>
    </div>
    """
    
    yield (gallery_paths, status_update, response_state, analysis_html, stats_info_html, 
           gallery_items_state, selected_indices_state, selected_count_md, selected_preview,
           current_page, page_info)
    
    # ==============================================================================
    # === BƯỚC 3: GỌI BACKEND & XỬ LÝ LỖI ========================================
    # ==============================================================================
    
    try:
        # Tạo dictionary config để truyền vào backend
        config = {
            "top_k_final": int(num_results),
            "kis_retrieval": int(kis_retrieval),
            "vqa_candidates": int(vqa_candidates),
            "vqa_retrieval": int(vqa_retrieval),
            "trake_candidates_per_step": int(trake_candidates_per_step),
            "trake_max_sequences": int(trake_max_sequences),
            "w_clip": w_clip,
            "w_obj": w_obj,
            "w_semantic": w_semantic,
            "lambda_mmr": lambda_mmr
        }
        
        start_time = time.time()
        full_response = master_searcher.search(query=query_text, config=config)
        search_time = time.time() - start_time

    except Exception as e:
        print(f"--- ❌ LỖI NGHIÊM TRỌNG TRONG PIPELINE TÌM KIẾM: {e} ---")
        import traceback
        traceback.print_exc()
        status_msg = f"<div style='color: red;'>🔥 Đã xảy ra lỗi backend: {e}</div>"
        # Trả về trạng thái lỗi và các giá trị rỗng
        return (gallery_paths, status_msg, response_state, analysis_html, stats_info_html, 
                gallery_items_state, selected_indices_state, selected_count_md, selected_preview,
                current_page, page_info)

    # ==============================================================================
    # === BƯỚC 4: ĐỊNH DẠNG KẾT QUẢ & CẬP NHẬT UI CUỐI CÙNG ======================
    # ==============================================================================

    gallery_paths = format_results_for_mute_gallery(full_response)
    response_state = full_response
    
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    num_found = len(gallery_paths)
    
    if num_found == 0:
        status_msg = f"<div style='color: #d97706;'>😔 **{task_type_msg}** | Không tìm thấy kết quả nào trong {search_time:.2f} giây.</div>"
    else:
        status_msg = f"<div style='color: #166534;'>✅ **{task_type_msg}** | Tìm thấy {num_found} kết quả trong {search_time:.2f} giây.</div>"

    query_analysis = full_response.get('query_analysis', {})
    analysis_html = f"""
    <div style="background-color: #f3f4f6; border-radius: 8px; padding: 15px;">
        <h4 style="margin: 0 0 10px 0; color: #111827;">🧠 Phân tích Truy vấn AI</h4>
        <div style="font-size: 14px; line-height: 1.6;">
            <strong>Bối cảnh Tìm kiếm:</strong> <em>{query_analysis.get('search_context', 'N/A')}</em><br>
            <strong>Đối tượng (EN):</strong> <code>{', '.join(query_analysis.get('objects_en', []))}</code><br>
            <strong>Câu hỏi VQA (nếu có):</strong> <em>{query_analysis.get('specific_question', 'Không có')}</em>
        </div>
    </div>
    """

    stats_info_html = f"""
    <div style="background-color: #f3f4f6; border-radius: 8px; padding: 15px;">
        <h4 style="margin: 0 0 10px 0; color: #111827;">📊 Thống kê Nhanh</h4>
        <div style="font-size: 14px; line-height: 1.6;">
            <strong>Thời gian:</strong> {search_time:.2f} giây<br>
            <strong>Kết quả:</strong> {num_found}
        </div>
    </div>
    """
    initial_gallery_view = gallery_paths[:ITEMS_PER_PAGE]
    
    current_page = 1
    total_pages = int(np.ceil(len(gallery_paths) / ITEMS_PER_PAGE)) or 1
    page_info = f"Trang {current_page} / {total_pages}"
    
    yield (
        initial_gallery_view,   # 1. results_gallery (chỉ 20 ảnh đầu)
        status_msg,             # 2. status_output
        response_state,         # 3. response_state
        analysis_html,          # 4. gemini_analysis
        stats_info_html,        # 5. stats_info
        gallery_paths,          # 6. gallery_items_state (toàn bộ 100 đường dẫn)
        [],                     # 7. selected_indices_state (reset)
        "Đã chọn: 0",           # 8. selected_count_md (reset)
        [],                     # 9. selected_preview (reset)
        current_page,           # 10. current_page_state (reset về 1)
        page_info               # 11. page_info_display
    )


def _create_detailed_info_html(result: Dict[str, Any], task_type: TaskType) -> str:
    """
    Hàm phụ trợ tạo mã HTML chi tiết cho một kết quả được chọn.
    """
    def create_progress_bar(score, color):
        percentage = max(0, min(100, score * 100))
        return f"""
        <div style="background: #e0e0e0; border-radius: 10px; height: 8px; margin: 5px 0;">
            <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 10px; transition: width 0.3s ease;"></div>
        </div>
        """

    video_id = result.get('video_id', 'N/A')
    keyframe_id = result.get('keyframe_id', 'N/A')
    timestamp = result.get('timestamp', 0)
    final_score = result.get('final_score', 0)
    scores = result.get('scores', {})
    
    html = f"""
    <div style="background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%); padding: 20px; border-radius: 12px; color: white;">
        <h3 style="margin: 0; color: white;">🎬 Chi tiết Keyframe</h3>
        
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div><strong>📹 Video:</strong><br><code ...>{video_id}</code></div>
                <div><strong>⏰ Thời điểm:</strong><br><code ...>{timestamp:.2f}s</code></div>
            </div>
        </div>
    """

    if task_type == TaskType.QNA:
        answer = result.get('answer', 'N/A')
        vqa_conf = scores.get('vqa_confidence', 0)
        html += f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4 style="margin: 0 0 10px 0; color: white;">💬 Câu trả lời (VQA)</h4>
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">{answer}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 14px;">
                <span>Độ tự tin:</span><span>{vqa_conf:.2f}</span>
            </div>
            {create_progress_bar(vqa_conf, '#8e44ad')}
        </div>
        """

    html += f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4 style="margin: 0 0 15px 0; color: white;">🏆 Điểm số chi tiết</h4>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span><strong>📊 Điểm tổng:</strong></span>
                    <span style="font-weight: bold; font-size: 18px;">{final_score:.4f}</span>
                </div>
                {create_progress_bar(final_score, '#00b894')}
            </div>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>🖼️ CLIP Score:</span><span>{scores.get('clip', 0):.3f}</span>
                </div>
                {create_progress_bar(scores.get('clip', 0), '#0984e3')}
            </div>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>🎯 Object Score:</span><span>{scores.get('object', 0):.3f}</span>
                </div>
                {create_progress_bar(scores.get('object', 0), '#e17055')}
            </div>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>🧠 Semantic Score:</span><span>{scores.get('semantic', 0):.3f}</span>
                </div>
                {create_progress_bar(scores.get('semantic', 0), '#a29bfe')}
            </div>
        </div>
    """

    detected_objects = result.get('objects_detected', [])
    objects_html = "".join([f'<span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 14px;">{obj}</span>' for obj in detected_objects])
    html += f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
            <h4 style="margin: 0 0 10px 0; color: white;">🔍 Đối tượng phát hiện</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                {objects_html if objects_html else "Không có đối tượng nổi bật."}
            </div>
        </div>
    </div>
    """
    return html

def on_gallery_select(
    # --- Inputs MỚI ---
    response_state: Dict[str, Any], 
    current_page: int,
    evt: gr.SelectData
):
    """
    Khi click 1 ảnh trong gallery: Cập nhật toàn bộ Trạm Phân tích ở cột phải.
    *** PHIÊN BẢN NÂNG CẤP TỪ CODE GỐC CỦA BẠN ***
    """
    # --- Bước 1: Validate & Tính toán Index Toàn cục ---
    if not response_state or evt is None:
        gr.Warning("Vui lòng thực hiện tìm kiếm trước.")
        # Trả về giá trị rỗng cho tất cả outputs của Trạm Phân tích
        return None, None, pd.DataFrame(), "", "", None, "", ""

    results = response_state.get("results", [])
    task_type = response_state.get("task_type")
    
    # Tính index toàn cục dựa trên trang hiện tại
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    
    if not results or global_index >= len(results):
        gr.Error("Lỗi: Dữ liệu không đồng bộ.")
        return None, None, pd.DataFrame(), "", "", None, "", ""

    # --- Bước 2: Lấy dữ liệu của ứng viên được chọn ---
    selected_result = results[global_index]

    # --- Nhánh 1: Xử lý kết quả chuỗi TRAKE ---
    if task_type == TaskType.TRAKE:
        sequence = selected_result.get('sequence', [])
        if not sequence:
            return None, "Lỗi: Chuỗi TRAKE rỗng.", pd.DataFrame(), "", "", None
        
        # Lấy frame đầu tiên của chuỗi làm đại diện
        target_frame = sequence[0]
        
        # Tạo HTML đặc biệt cho TRAKE
        html_output = f"<div style='padding: 15px; background-color: #f3f4f6; border-radius: 8px;'>"
        html_output += f"<h4 style='margin-top:0;'>Chuỗi hành động ({len(sequence)} bước)</h4>"
        html_output += f"<p><strong>Video:</strong> <code>{selected_result.get('video_id')}</code> | <strong>Điểm trung bình:</strong> {selected_result.get('final_score', 0):.3f}</p>"
        html_output += "<div style='display: flex; gap: 10px; overflow-x: auto; padding-bottom: 10px;'>"
        for i, frame in enumerate(sequence):
            b64_img = encode_image_to_base64(frame.get('keyframe_path'))
            html_output += f"<div style='text-align: center; flex-shrink: 0;'><p style='margin:0;font-weight:bold;'>Bước {i+1}</p><img src='{b64_img}' style='width:120px; border-radius: 4px; border: 2px solid #ddd;'><p style='font-size:12px;margin:2px 0;'>@{frame.get('timestamp',0):.1f}s</p></div>"
        html_output += "</div></div>"
        
        video_path = target_frame.get('video_path')
        timestamp = target_frame.get('timestamp')
        video_clip_path = create_video_segment(video_path, timestamp)
        
        clip_info_html = f"<div style='text-align: center;'>Clip đại diện cho chuỗi</div>"

        # --- SỬA ĐỔI: Trả về 8 giá trị ---
        return (target_frame.get('keyframe_path'), video_clip_path, pd.DataFrame(),
                "", "", selected_result, html_output, clip_info_html)

    # --- Nhánh 2: Xử lý kết quả đơn lẻ (KIS và QNA) ---
    else:
        video_path = selected_result.get('video_path')
        timestamp = selected_result.get('timestamp')
        
        # Chuẩn bị dữ liệu
        selected_image_path = selected_result.get('keyframe_path')
        video_clip_path = create_video_segment(video_path, timestamp)

        # Bảng điểm
        scores = selected_result.get('scores', {})
        scores_data = {"Metric": [], "Value": []}
        # Thêm các điểm thành phần một cách linh hoạt
        score_map = {
            "🏆 Final Score": selected_result.get('final_score', 0),
            "🖼️ CLIP Score": scores.get('clip', None),
            "🎯 Object Score": scores.get('object', None),
            "🧠 Semantic Score": scores.get('semantic', None),
            "💬 VQA Confidence": scores.get('vqa_confidence', None)
        }
        for name, value in score_map.items():
            if value is not None:
                scores_data["Metric"].append(name)
                scores_data["Value"].append(value)
        scores_df = pd.DataFrame(scores_data)

        # Câu trả lời VQA
        vqa_answer = selected_result.get('answer', "") if task_type == TaskType.QNA else ""

        # Transcript
        transcript = selected_result.get('transcript_text', "Không có transcript.")

        detailed_info_html = _create_detailed_info_html(selected_result, task_type)
        
        clip_info_html = f"""
        <div style="text-align: center; margin-top: 10px; font-size: 14px; padding: 8px; background-color: #f3f4f6; border-radius: 8px;">
            Clip 10 giây từ <strong>{os.path.basename(video_path or "N/A")}</strong>
        </div>
        """

        # --- SỬA ĐỔI: Trả về 8 giá trị ---
        return (selected_image_path, 
                video_clip_path, 
                scores_df, 
                vqa_answer, 
                transcript, 
                selected_result, 
                detailed_info_html, 
                clip_info_html)

def select_all_items(gallery_items):
    """Chọn tất cả các item trong gallery hiện tại."""
    idxs = list(range(len(gallery_items or [])))
    return idxs, f"Đã chọn: {len(idxs)}", _build_selected_preview(gallery_items, idxs)

def clear_selection():
    """Bỏ chọn tất cả."""
    return [], "Đã chọn: 0", []

def deselect_from_selected_preview(gallery_items, selected_indices, evt: gr.SelectData):
    """Khi click một thumbnail trong 'Ảnh đã chọn', bỏ chọn nó."""
    if evt is None or not selected_indices: return selected_indices, f"Đã chọn: {len(selected_indices or [])}", _build_selected_preview(gallery_items, selected_indices)
    
    k = int(evt.index)
    current_selection = list(selected_indices)
    if 0 <= k < len(current_selection):
        item_to_remove = current_selection.pop(k)
    return current_selection, f"Đã chọn: {len(current_selection)}", _build_selected_preview(gallery_items, current_selection)

def download_selected_zip(gallery_items, selected_indices):
    """Tạo và trả về file ZIP của các ảnh đã chọn."""
    if not selected_indices:
        gr.Warning("Chưa có ảnh nào được chọn để tải về.")
        return None
    out_path = "/kaggle/working/selected_images.zip"
    if os.path.exists(out_path): os.remove(out_path)
    
    with zipfile.ZipFile(out_path, "w") as zf:
        for i in sorted(selected_indices):
            path = _normalize_item_to_path(gallery_items[i])
            if path and os.path.isfile(path): zf.write(path, arcname=os.path.basename(path))
    return out_path

def handle_submission(response_state: Dict[str, Any], query_id: str):
    """
    Tạo và cung cấp file nộp bài.
    Hàm này không bị ảnh hưởng vì `task_type` và `results` vẫn có trong state đã được dọn dẹp.
    """
    if not response_state or not response_state.get('results'):
        gr.Warning("Không có kết quả để tạo file nộp bài.")
        return None
    
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
        
    # Hàm này vẫn hoạt động bình thường với `cleaned_response_for_state`
    submission_df = format_for_submission(response_state, max_results=100)
    
    if submission_df.empty:
        gr.Warning("Không thể định dạng kết quả để nộp bài.")
        return None
              
    file_path = generate_submission_file(submission_df, query_id=query_id)
    return file_path

def clear_all():
    """Nâng cấp để xóa tất cả các output và state mới."""
    return (
        [], "", None, "", "", None, "", "", "", None, # Outputs cũ
        "Đã chọn: 0", [], [], None, [] # Outputs mới (selection)
    )
    
def _format_submission_list_for_display(submission_list: list) -> str:
    """Hàm phụ trợ để biến danh sách submission thành một chuỗi text đẹp mắt."""
    if not submission_list:
        return "Chưa có kết quả nào được thêm vào."
    
    display_text = ""
    for i, item in enumerate(submission_list):
        task_type = item.get('task_type')
        item_info = ""
        if task_type == TaskType.TRAKE:
            item_info = f"TRAKE Seq | Vid: {item.get('video_id')} | Score: {item.get('final_score', 0):.3f}"
        else: # KIS, QNA
            item_info = f"Frame | {item.get('keyframe_id')} | Score: {item.get('final_score', 0):.3f}"
        
        display_text += f"{i+1:02d}. {item_info}\n" # Thêm số thứ tự 2 chữ số
    return display_text

def add_to_submission_list(
    submission_list: list, 
    candidate: Dict[str, Any], 
    response_state: Dict[str, Any], # Cần response_state để lấy task_type
    position: str
):
    """Thêm một ứng viên vào danh sách nộp bài."""
    if not candidate:
        gr.Warning("Chưa có ứng viên nào được chọn để thêm!")
        return _format_submission_list_for_display(submission_list), submission_list

    task_type = response_state.get("task_type")
    
    # Tạo một bản sao sạch của ứng viên để lưu trữ
    item_to_add = candidate.copy()
    item_to_add['task_type'] = task_type # Gắn loại nhiệm vụ vào item

    # Kiểm tra trùng lặp
    existing_ids = {item.get('keyframe_id') for item in submission_list if item.get('keyframe_id')}
    if task_type != TaskType.TRAKE and item_to_add.get('keyframe_id') in existing_ids:
        gr.Info("Frame này đã có trong danh sách nộp bài.")
        return _format_submission_list_for_display(submission_list), submission_list
        
    # Thêm vào vị trí mong muốn
    if position == 'top':
        submission_list.insert(0, item_to_add)
    else: # bottom
        submission_list.append(item_to_add)
        
    # Giới hạn danh sách ở 100
    if len(submission_list) > 100:
        if position == 'top':
             submission_list = submission_list[:100]
        else: # Nếu thêm vào cuối, loại bỏ phần tử đầu
             submission_list = submission_list[-100:]
        gr.Info("Danh sách đã đạt 100 kết quả.")

    gr.Success(f"Đã thêm kết quả vào {'đầu' if position == 'top' else 'cuối'} danh sách!")
    return _format_submission_list_for_display(submission_list), submission_list

def clear_submission_list():
    """Xóa toàn bộ danh sách nộp bài."""
    gr.Info("Đã xóa danh sách nộp bài.")
    return "Chưa có kết quả nào được thêm vào.", []

# Cập nhật hàm handle_submission
def handle_submission(submission_list: list, query_id: str):
    if not submission_list:
        gr.Warning("Danh sách nộp bài đang trống.")
        return None
    
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
        
    # Gọi hàm format mới
    submission_df = format_list_for_submission(submission_list, max_results=100)
    
    if submission_df.empty:
        gr.Warning("Không thể định dạng kết quả để nộp bài.")
        return None
              
    file_path = generate_submission_file(submission_df, query_id=query_id)
    return file_path

print("--- Giai đoạn 4/4: Đang xây dựng giao diện người dùng...")

custom_css = """
/* Ẩn footer */
footer {display: none !important}

/* Custom styling cho gallery */
.gallery {
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

/* === THÊM VÀO ĐỂ SỬA LỖI CUỘN GALLERY CHÍNH === */
/* 
  Nhắm chính xác vào khu vực chứa ảnh bên trong gallery chính
  và buộc nó phải có thanh cuộn dọc khi nội dung vượt quá chiều cao.
*/
#results-gallery > .gradio-gallery { 
    height: 700px !important; 
    overflow-y: auto !important;
}
/* ============================================== */

/* Animation cho buttons */
.gradio-button {
    transition: all 0.3s ease !important;
    border-radius: 25px !important;
    font-weight: 600 !important;
}

.gradio-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
}

/* Custom textbox styling */
.gradio-textbox {
    border-radius: 12px !important;
    border: 2px solid #e0e0e0 !important;
    transition: all 0.3s ease !important;
}

.gradio-textbox:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Video player styling */
video {
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

/* Gallery items hover effect */
.gallery img {
    transition: transform 0.3s ease !important;
    border-radius: 8px !important;
}

.gallery img:hover {
    transform: scale(1.05) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}
"""
app_header_html = """
        <div style="text-align: center; max-width: 1200px; margin: 0 auto 30px auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; border-radius: 20px; color: white; box-shadow: 0 10px 40px rgba(0,0,0,0.1);">
                <h1 style="margin: 0 0 15px 0; font-size: 3em; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🚀 AIC25 Video Search Engine
                </h1>
                <p style="margin: 0; font-size: 1.3em; opacity: 0.9; font-weight: 300;">
                    Hệ thống tìm kiếm video thông minh với AI - Nhập mô tả bằng tiếng Việt
                </p>
                <div style="margin-top: 20px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                    <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 15px; font-size: 0.9em;">
                        ⚡ Semantic Search
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 15px; font-size: 0.9em;">
                        🎯 Object Detection
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 15px; font-size: 0.9em;">
                        🧠 Gemini AI
                    </div>
                </div>
            </div>
        </div>
    """

app_footer_html = """
    <div style="text-align: center; margin-top: 40px; padding: 20px; background: linear-gradient(135deg, #636e72 0%, #2d3436 100%); border-radius: 12px; color: white;">
        <p style="margin: 0; opacity: 0.8;">
            🚀 AIC25 Video Search Engine - Powered by AI & Computer Vision
        </p>
        <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.6;">
            Sử dụng Semantic Search, Object Detection và Gemini AI để tìm kiếm video thông minh
        </p>
    </div>
    """


print("--- Giai đoạn 4/4: Đang xây dựng giao diện người dùng (Ultimate Battle Station)...")

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🚀 AIC25 Video Search") as app:
    
    gr.HTML(app_header_html)

    # --- Khai báo States ---
    response_state = gr.State()
    gallery_items_state = gr.State([])
    selected_indices_state = gr.State([])
    current_page_state = gr.State(1) 
    submission_list_state = gr.State([])
    selected_candidate_for_submission = gr.State()

    # --- BỐ CỤC CHÍNH 2 CỘT ---
    with gr.Row(variant='panel'):
        # --- CỘT TRÁI (2/3 không gian): TÌM KIẾM, KẾT QUẢ, THU THẬP ---
        with gr.Column(scale=2):
            
            # --- 1. Khu vực Nhập liệu & Điều khiển chính ---
            gr.Markdown("### 1. Nhập truy vấn")
            query_input = gr.Textbox(
                label="🔍 Nhập mô tả chi tiết bằng tiếng Việt",
                placeholder="Ví dụ: một người phụ nữ mặc váy đỏ đang nói về việc bảo tồn rùa biển...",
                lines=2,
                autofocus=True
            )
            with gr.Row():
                search_button = gr.Button("🚀 Tìm kiếm", variant="primary", size="lg")
                num_results = gr.Slider(
                    minimum=20, maximum=100, value=100, step=10,
                    label="📊 Số kết quả tối đa",
                    interactive=True
                )
                clear_button = gr.Button("🗑️ Xóa tất cả", variant="secondary", size="lg")

            # --- 2. Khu vực Tinh chỉnh Nâng cao ---
            with gr.Accordion("⚙️ Tùy chỉnh Nâng cao", open=False):
                with gr.Tabs():
                    # *** HOÀN THIỆN ĐỊNH NGHĨA CÁC SLIDER TẠI ĐÂY ***
                    with gr.TabItem("KIS / Chung"):
                        kis_retrieval_slider = gr.Slider(
                            minimum=50, maximum=500, value=100, step=25,
                            label="Số ứng viên KIS ban đầu (Retrieval)",
                            info="Lấy bao nhiêu ứng viên từ FAISS trước khi rerank cho KIS."
                        )
                    with gr.TabItem("VQA"):
                        vqa_candidates_slider = gr.Slider(
                            minimum=3, maximum=30, value=8, step=1,
                            label="Số ứng viên VQA",
                            info="Hỏi đáp AI trên bao nhiêu ứng viên có bối cảnh tốt nhất."
                        )
                        vqa_retrieval_slider = gr.Slider(
                            minimum=50, maximum=500, value=200, step=25,
                            label="Số ứng viên VQA ban đầu (Retrieval)",
                            info="Lấy bao nhiêu ứng viên từ FAISS để tìm bối cảnh cho VQA."
                        )
                    with gr.TabItem("TRAKE"):
                        trake_candidates_per_step_slider = gr.Slider(
                            minimum=5, maximum=30, value=15, step=1,
                            label="Số ứng viên mỗi bước (TRAKE)",
                            info="Với mỗi bước trong chuỗi, lấy bao nhiêu ứng viên."
                        )
                        trake_max_sequences_slider = gr.Slider(
                            minimum=10, maximum=100, value=50, step=5,
                            label="Số chuỗi kết quả tối đa (TRAKE)",
                            info="Số lượng chuỗi tối đa sẽ được trả về."
                        )
                        
                    with gr.TabItem("⚖️ Trọng số & Đa dạng"):
                        gr.Markdown("Điều chỉnh tầm quan trọng của các yếu tố khi tính điểm cuối cùng.")
                        w_clip_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="w_clip (Thị giác Tổng thể)")
                        w_obj_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_obj (Đối tượng)")
                        w_semantic_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_semantic (Ngữ nghĩa)")
                        
                        # --- THÊM WIDGET BỊ THIẾU VÀO ĐÂY ---
                        gr.Markdown("---") # Thêm một đường kẻ ngang để phân tách
                        gr.Markdown("Điều chỉnh sự cân bằng giữa Độ liên quan và Sự đa dạng của kết quả.")
                        lambda_mmr_slider = gr.Slider(
                            minimum=0.0, 
                            maximum=1.0, 
                            value=0.7, 
                            step=0.05, 
                            label="λ - MMR (0.0=Đa dạng nhất, 1.0=Liên quan nhất)"
                        )
            # --- 3. Khu vực Trạng thái & Phân tích ---
            status_output = gr.HTML()
            with gr.Row():
                gemini_analysis = gr.HTML()
                stats_info = gr.HTML()
            # --- 4. Khu vực Kết quả chính ---
            gr.Markdown("### 2. Kết quả tìm kiếm")
            
            # --- THÊM MỚI: Bảng điều khiển phân trang ---
            with gr.Row(equal_height=True, variant='compact'):
                prev_page_button = gr.Button("◀️ Trang trước")
                page_info_display = gr.Markdown("Trang 1 / 1", elem_id="page-info")
                next_page_button = gr.Button("▶️ Trang sau")
                
            results_gallery = gr.Gallery(
                label="Click vào một ảnh để phân tích sâu",
                show_label=True,
                elem_id="results-gallery",
                columns=5, # Giữ nguyên mật độ cao
                object_fit="contain",
                height=580, # Chiều cao cố định, không cần cuộn
                allow_preview=False
            )

        # --- CỘT PHẢI (1/3 không gian): XEM CHI TIẾT & NỘP BÀI ---
        with gr.Column(scale=1):
            
            gr.Markdown("### 3. Trạm Phân tích")
            
            # --- KHAI BÁO CÁC COMPONENT BỊ THIẾU Ở ĐÂY ---
            selected_image_display = gr.Image(label="Ảnh Keyframe Được chọn", type="filepath")
            video_player = gr.Video(label="🎬 Clip 10 giây", autoplay=True)
            
            with gr.Tabs():
                with gr.TabItem("📊 Phân tích & Điểm số"):
                    # Component này sẽ nhận HTML từ on_gallery_select
                    detailed_info = gr.HTML() 
                    scores_display = gr.DataFrame(headers=["Metric", "Value"], label="Bảng điểm")
                    
                with gr.TabItem("💬 VQA & Transcript"):
                    vqa_answer_display = gr.Textbox(label="Câu trả lời VQA", interactive=False, lines=5)
                    transcript_display = gr.Textbox(label="📝 Transcript", lines=8, interactive=False)
                    
            clip_info = gr.HTML() 

            gr.Markdown("### 4. Vùng Nộp bài")
            with gr.Row():
                add_top_button = gr.Button("➕ Thêm vào Top 1", variant="primary")
                add_bottom_button = gr.Button("➕ Thêm vào cuối")
            
            with gr.Tabs():
                with gr.TabItem("📋 Danh sách Nộp bài"):
                    submission_list_display = gr.Textbox(
                        label="Thứ tự Nộp bài (Top 1 ở trên cùng)",
                        lines=15,
                        interactive=False,
                        value="Chưa có kết quả nào."
                    )
                    clear_submission_button = gr.Button("🗑️ Xóa toàn bộ danh sách")

                with gr.TabItem("💾 Xuất File"):
                    query_id_input = gr.Textbox(label="Nhập Query ID", placeholder="Ví dụ: query_01")
                    submission_button = gr.Button("💾 Tạo File CSV")
                    submission_file_output = gr.File(label="Tải file nộp bài tại đây")

    gr.HTML(app_footer_html)
    
    search_inputs = [
        query_input, num_results, kis_retrieval_slider, vqa_candidates_slider,
        vqa_retrieval_slider, trake_candidates_per_step_slider, trake_max_sequences_slider,
        w_clip_slider, w_obj_slider, w_semantic_slider, lambda_mmr_slider 
    ]
    search_outputs = [
        results_gallery, status_output, response_state, gemini_analysis, stats_info,
        gallery_items_state, selected_indices_state, current_page_state, page_info_display 
    ]
    search_button.click(fn=perform_search, inputs=search_inputs, outputs=search_outputs)
    query_input.submit(fn=perform_search, inputs=search_inputs, outputs=search_outputs)

    prev_page_button.click(
        fn=update_gallery_page,
        inputs=[gallery_items_state, current_page_state, gr.Textbox("◀️ Trang trước", visible=False)],
        outputs=[results_gallery, current_page_state, page_info_display]
    )
    
    next_page_button.click(
        fn=update_gallery_page,
        inputs=[gallery_items_state, current_page_state, gr.Textbox("▶️ Trang sau", visible=False)],
        outputs=[results_gallery, current_page_state, page_info_display]
    )
    # 1. Định nghĩa outputs cho sự kiện select
    analysis_outputs = [
        selected_image_display,
        video_player,
        scores_display,
        vqa_answer_display,
        transcript_display,
        selected_candidate_for_submission,
        detailed_info, # `detailed_info` giờ đã được định nghĩa
        clip_info      # `clip_info` giờ đã được định nghĩa
    ]
    results_gallery.select(
        fn=on_gallery_select,
        inputs=[response_state, current_page_state], 
        outputs=analysis_outputs
    )

    # 4. Sự kiện Nộp bài
    add_top_button.click(
        fn=add_to_submission_list,
        inputs=[submission_list_state, selected_candidate_for_submission, response_state, gr.Textbox("top", visible=False)],
        outputs=[submission_list_display, submission_list_state]
    )
    
    add_bottom_button.click(
        fn=add_to_submission_list,
        inputs=[submission_list_state, selected_candidate_for_submission, response_state, gr.Textbox("bottom", visible=False)],
        outputs=[submission_list_display, submission_list_state]
    )
    
    clear_submission_button.click(
        fn=clear_submission_list,
        inputs=[],
        outputs=[submission_list_display, submission_list_state]
    )
    
    submission_button.click(
        fn=handle_submission,
        inputs=[submission_list_state, query_id_input],
        outputs=[submission_file_output]
    )

    # 5. Sự kiện Xóa tất cả
    clear_outputs = [
        results_gallery, status_output, response_state, gemini_analysis, stats_info,
        video_player, detailed_info, clip_info, query_id_input, submission_file_output,
        selected_indices_state, gallery_items_state
    ]
    clear_button.click(fn=clear_all, inputs=None, outputs=clear_outputs)

if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server ---")
    app.launch(
        share=True,
        allowed_paths=["/kaggle/input/", "/kaggle/working/"], 
        debug=True, # Bật debug để xem lỗi chi tiết trên console
        show_error=True # Hiển thị lỗi trên giao diện
    )