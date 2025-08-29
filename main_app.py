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

from utils import (
    create_video_segment,
    format_results_for_gallery,
    format_for_submission,
    generate_submission_file
)
import base64

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
    basic_searcher = BasicSearcher(FAISS_INDEX_PATH, RERANK_METADATA_PATH, video_path_map)
    
    # Bước 2: Khởi tạo MasterSearcher phiên bản OpenAI
    # MasterSearcher giờ sẽ tự quản lý SemanticSearcher và OpenAIHandler bên trong
    print("   -> 2/2: Khởi tạo MasterSearcher (OpenAI Edition and GEMINI)...")
    master_searcher = MasterSearcher(
            basic_searcher=basic_searcher,
            openai_api_key=OPENAI_API_KEY,
            gemini_api_key=GEMINI_API_KEY,
            entities_path=ALL_ENTITIES_PATH
        )    
    print("--- ✅ Backend đã khởi tạo thành công! ---")
    return master_searcher

master_searcher = initialize_backend()

print("--- Giai đoạn 3/4: Đang định nghĩa các hàm logic cho giao diện...")

def perform_search(query_text: str, 
        num_results: int,
        kis_retrieval: int,
        vqa_candidates: int,
        vqa_retrieval: int,
        trake_candidates_per_step: int,
        trake_max_sequences: int,
        track_vqa_retrieval: int, # <-- Thêm tham số
        track_vqa_candidates: int,  # <-- Thêm tham số
        w_clip: float, 
        w_obj: float, 
        w_semantic: float
    ):
    """
    Hàm chính xử lý sự kiện tìm kiếm. Gọi MasterSearcher và định dạng kết quả.
    """
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "⚠️ Vui lòng nhập truy vấn và bấm Tìm kiếm.", None, "", ""
    
    config = {
        "top_k_final": int(num_results),
        "kis_retrieval": int(kis_retrieval),
        "vqa_candidates": int(vqa_candidates),
        "vqa_retrieval": int(vqa_retrieval),
        "trake_candidates_per_step": int(trake_candidates_per_step),
        "trake_max_sequences": int(trake_max_sequences),
        "track_vqa_retrieval": int(track_vqa_retrieval), 
        "track_vqa_candidates": int(track_vqa_candidates),
        "w_clip": w_clip,
        "w_obj": w_obj,
        "w_semantic": w_semantic
    }
    
    start_time = time.time()
    
    full_response = master_searcher.search(query=query_text, config=config)
    
    search_time = time.time() - start_time
    
    formatted_gallery = format_results_for_gallery(full_response)
    if isinstance(formatted_gallery, list):
        formatted_gallery = formatted_gallery[:100]
        
    query_analysis = full_response.get('query_analysis', {})
    gemini_analysis_html = f"""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 12px; color: white; margin: 10px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; color: white; display: flex; align-items: center;">
            🧠 Phân tích truy vấn AI
        </h3>
        <div style="margin-top: 15px;">
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 8px 0;">
                <strong>🎯 Đối tượng (VI):</strong> <code style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px;">{', '.join(full_response['query_analysis'].get('objects_vi', []))}</code>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 8px 0;">
                <strong>🌍 Đối tượng (EN):</strong> <code style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px;">{', '.join(full_response['query_analysis'].get('objects_en', []))}</code>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 8px 0;">
                <strong>📝 Bối cảnh:</strong> <em>"{full_response['query_analysis'].get('context_vi', '')}"</em>
            </div>
        </div>
    </div>
    """
        
    stats_info_html =  f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; color: white; margin: 10px 0;">
        <h3 style="margin: 0; color: white;">🔄 Đang xử lý truy vấn...</h3>
        <p style="margin: 10px 0 0 0; opacity: 0.9;"> Số kết quả: <strong>{num_results}</strong></p>
    </div>
    """
    cleaned_response_for_state = {"task_type": full_response.get("task_type"), "results": full_response.get("results")}
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    status_msg_html = f"✅ Tìm kiếm hoàn tất trong {search_time:.2f}s. Chế độ: {task_type_msg}"
    
    return (
        formatted_gallery, status_msg_html, cleaned_response_for_state, 
        gemini_analysis_html, stats_info_html, formatted_gallery, 
        [], "Đã chọn: 0", []
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

def on_gallery_select(response_state: Dict[str, Any], gallery_items, selected_indices, evt: gr.SelectData):
    """
    Khi click 1 ảnh trong gallery: hiển thị preview, toggle chọn/bỏ chọn, cập nhật 'Ảnh đã chọn'.
    """
    if not response_state or evt is None:
        current_selection = selected_indices or []
        return None, "", "", current_selection, f"Đã chọn: {len(current_selection)}", _build_selected_preview(gallery_items, current_selection)

    results = response_state.get("results", [])
    if not results or evt.index >= len(results):
        gr.Error("Lỗi: Không tìm thấy kết quả tương ứng.")
        current_selection = selected_indices or []
        return None, "Lỗi: Dữ liệu không đồng bộ.", "", current_selection, f"Đã chọn: {len(current_selection)}", _build_selected_preview(gallery_items, current_selection)

    selected_result = results[evt.index]; task_type = response_state.get('task_type')

    # --- Nhánh 1: Xử lý kết quả tổng hợp TRACK_VQA ---
    if selected_result.get("is_aggregated_result"):
        final_answer = selected_result.get("final_answer", "N/A")
        evidence_paths = selected_result.get("evidence_paths", [])
        evidence_captions = selected_result.get("evidence_captions", [])
        
        evidence_html = ""
        if evidence_paths:
            evidence_html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; margin-top: 15px;">'
            for path, caption in zip(evidence_paths, evidence_captions):
                image_base64_src = encode_image_to_base64(path)
                evidence_html += f"""
                <div style="text-align: center;">
                    <img src="{image_base64_src}" style="width: 100%; height: auto; border-radius: 8px; border: 2px solid #ddd;" alt="Evidence Frame">
                    <p style="font-size: 12px; margin: 5px 0 0 0; color: #333;">{caption}</p>
                </div>
                """
            evidence_html += '</div>'
        else:
            evidence_html = "<p>Không có hình ảnh bằng chứng nào được tìm thấy.</p>"
            
        detailed_info_html = f"""
        <div style="padding: 20px; border-radius: 12px; background-color: #f8f9fa;">
            <h3 style="margin: 0 0 15px 0; border-bottom: 2px solid #dee2e6; padding-bottom: 10px;">💡 Kết quả Phân tích Tổng hợp</h3>
            <div style="background-color: #e9ecef; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <p style="font-size: 16px; margin: 0; line-height: 1.6;">{final_answer}</p>
            </div>
            <h4 style="margin: 0 0 10px 0;">🖼️ Các hình ảnh bằng chứng:</h4>
            {evidence_html}
        </div>
        """
        
        return None, detailed_info_html, "Thông tin tổng hợp cho truy vấn của bạn."

    # --- Nhánh 2: Xử lý kết quả chuỗi TRAKE ---
    elif task_type == TaskType.TRAKE:
        sequence = selected_result.get('sequence', [])
        if not sequence:
             return None, "Lỗi: Chuỗi TRAKE rỗng.", ""
        
        # Lấy frame đầu tiên để tạo clip và làm thông tin chính
        target_frame = sequence[0]
        video_path = target_frame.get('video_path')
        timestamp = target_frame.get('timestamp')
        
        # Tạo HTML chi tiết cho cả chuỗi
        seq_html = f"""...""" # Dán code tạo HTML cho TRAKE vào đây
        detailed_info_html = seq_html

    # --- Nhánh 3: Xử lý kết quả đơn lẻ KIS và QNA ---
    else:
        target_frame = selected_result
        video_path = target_frame.get('video_path')
        timestamp = target_frame.get('timestamp')
        # Gọi hàm phụ trợ để tạo HTML chi tiết
        detailed_info_html = _create_detailed_info_html(target_frame, task_type)

    # --- Logic chung cho Nhánh 2 và 3 (TRAKE, KIS, QNA) ---
    # Chỉ thực thi nếu không phải là TRACK_VQA
    video_clip_path = create_video_segment(video_path, timestamp)
    
    clip_info_html = f"""
    <div style="background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); padding: 15px; border-radius: 12px; color: white; text-align: center; margin-top: 10px;">
        <h4 style="margin: 0;">🎥 Video Clip (10 giây)</h4>
        <p style="margin: 8px 0 0 0; opacity: 0.9;">
            Từ ~{max(0, timestamp - 5):.1f}s đến ~{timestamp + 5:.1f}s
        </p>
    </div>
    """
    
    s = set(selected_indices or [])
    if evt.index is not None:
        if evt.index in s: s.remove(evt.index)
        else: s.add(evt.index)
    s_list = sorted(list(s))
    
    return video_clip_path, detailed_info_html, clip_info_html, s_list, f"Đã chọn: {len(s_list)}", _build_selected_preview(gallery_items, s_list)

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

print("--- Giai đoạn 4/4: Đang xây dựng giao diện người dùng...")

custom_css = """
/* Ẩn footer */
footer {display: none !important}

/* Custom styling cho gallery */
.gallery {
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

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
usage_guide_html = """
        <div style="padding: 20px; background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); border-radius: 12px; color: white;">
            <h3 style="margin-top: 0; color: white;">Cách sử dụng hệ thống:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 10px 0; color: white;">🔍 Tìm kiếm</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Nhập mô tả chi tiết bằng tiếng Việt</li>
                        <li>Sử dụng từ ngữ cụ thể về đối tượng, hành động, địa điểm</li>
                        <li>Chọn chế độ Semantic Search để có kết quả tốt nhất</li>
                    </ul>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 10px 0; color: white;">🎬 Xem video</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Click vào bất kỳ ảnh nào trong kết quả</li>
                        <li>Video clip 10 giây sẽ được tạo tự động</li>
                        <li>Xem thông tin chi tiết về điểm số và đối tượng</li>
                    </ul>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                    <h4 style="margin: 0 0 10px 0; color: white;">⚙️ Tùy chỉnh</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Điều chỉnh số lượng kết quả (6-24)</li>
                        <li>So sánh giữa Basic CLIP và Semantic Search</li>
                        <li>Xem phân tích AI từ Gemini</li>
                    </ul>
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
                    with gr.TabItem("KIS / Chung"):
                        kis_retrieval_slider = gr.Slider(...)
                    with gr.TabItem("VQA"):
                        vqa_candidates_slider = gr.Slider(...)
                        vqa_retrieval_slider = gr.Slider(...)
                    with gr.TabItem("TRAKE"):
                        trake_candidates_per_step_slider = gr.Slider(...)
                        trake_max_sequences_slider = gr.Slider(...)
                    with gr.TabItem("Track-VQA"):
                        track_vqa_retrieval_slider = gr.Slider(...)
                        track_vqa_candidates_slider = gr.Slider(...)
                    # *** GIỮ LẠI TAB QUAN TRỌNG NÀY ***
                    with gr.TabItem("⚖️ Trọng số Rerank"):
                        gr.Markdown("Điều chỉnh tầm quan trọng của các yếu tố khi tính điểm cuối cùng.")
                        w_clip_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="w_clip (Thị giác Tổng thể)")
                        w_obj_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_obj (Đối tượng)")
                        w_semantic_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_semantic (Ngữ nghĩa)")

            # --- 3. Khu vực Trạng thái & Phân tích ---
            status_output = gr.HTML()
            with gr.Row():
                gemini_analysis = gr.HTML(scale=1)
                stats_info = gr.HTML(scale=1)

            # --- 4. Khu vực Kết quả chính ---
            gr.Markdown("### 2. Kết quả tìm kiếm")
            results_gallery = gr.Gallery(
                label="Click vào ảnh để xem chi tiết và để CHỌN/BỎ CHỌN",
                show_label=True,
                elem_id="results-gallery",
                columns=5,
                object_fit="cover",
                height=700,
                allow_preview=False,
                preview=True
            )

            # --- 5. Khu vực Thu thập & Tải về ---
            gr.Markdown("### 3. Thu thập & Tải về")
            selected_count_md = gr.Markdown("Đã chọn: 0")
            selected_preview = gr.Gallery(
                label="Ảnh đã chọn (Click để bỏ chọn)",
                show_label=True,
                columns=8,
                rows=2,
                height=220,
                object_fit="cover"
            )
            with gr.Row():
                btn_select_all = gr.Button("Chọn tất cả")
                btn_clear_sel = gr.Button("Bỏ chọn tất cả")
                btn_download = gr.Button("Tải ZIP các ảnh đã chọn", variant="primary")
            zip_file_out = gr.File(label="Tải tệp ZIP của bạn tại đây")

        # --- CỘT PHẢI (1/3 không gian): XEM CHI TIẾT & NỘP BÀI ---
        with gr.Column(scale=1):
            
            # --- 1. Khu vực Xem Video & Chi tiết ---
            gr.Markdown("### Chi tiết Kết quả")
            video_player = gr.Video(label="🎬 Video Clip (10 giây)", autoplay=True)
            clip_info = gr.HTML()
            detailed_info = gr.HTML()

            # --- 2. Khu vực Nộp bài ---
            with gr.Accordion("💾 Tạo File Nộp Bài", open=True):
                query_id_input = gr.Textbox(label="Nhập Query ID", placeholder="Ví dụ: query_01")
                submission_button = gr.Button("Tạo File")
                submission_file_output = gr.File(label="Tải file nộp bài")

    gr.HTML(usage_guide_html)
    gr.HTML(app_footer_html)
    
    search_inputs = [
        query_input, num_results, kis_retrieval_slider, vqa_candidates_slider,
        vqa_retrieval_slider, trake_candidates_per_step_slider, trake_max_sequences_slider,
        track_vqa_retrieval_slider, track_vqa_candidates_slider,
        w_clip_slider, w_obj_slider, w_semantic_slider
    ]
    search_outputs = [
        results_gallery, status_output, response_state, gemini_analysis, stats_info,
        gallery_items_state, selected_indices_state, selected_count_md, selected_preview
    ]
    search_button.click(fn=perform_search, inputs=search_inputs, outputs=search_outputs)
    query_input.submit(fn=perform_search, inputs=search_inputs, outputs=search_outputs)

    # 2. Sự kiện Lựa chọn trong Gallery chính
    results_gallery.select(
        fn=on_gallery_select,
        inputs=[response_state, gallery_items_state, selected_indices_state],
        outputs=[
            video_player, detailed_info, clip_info, 
            selected_indices_state, selected_count_md, selected_preview
        ]
    )

    # 3. Sự kiện cho các nút Chọn/Bỏ chọn/Tải về
    btn_select_all.click(
        fn=select_all_items,
        inputs=[gallery_items_state],
        outputs=[selected_indices_state, selected_count_md, selected_preview]
    )
    btn_clear_sel.click(
        fn=clear_selection,
        inputs=[],
        outputs=[selected_indices_state, selected_count_md, selected_preview]
    )
    selected_preview.select(
        fn=deselect_from_selected_preview,
        inputs=[gallery_items_state, selected_indices_state],
        outputs=[selected_indices_state, selected_count_md, selected_preview]
    )
    btn_download.click(
        fn=download_selected_zip,
        inputs=[gallery_items_state, selected_indices_state],
        outputs=[zip_file_out]
    )

    # 4. Sự kiện Nộp bài
    submission_button.click(
        fn=handle_submission,
        inputs=[response_state, query_id_input],
        outputs=[submission_file_output]
    )

    # 5. Sự kiện Xóa tất cả
    clear_outputs = [
        results_gallery, status_output, response_state, gemini_analysis, stats_info,
        video_player, detailed_info, clip_info, query_id_input, submission_file_output,
        selected_count_md, selected_indices_state, gallery_items_state, zip_file_out, selected_preview
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