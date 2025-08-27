# /kaggle/working/main_app.py

# ==============================================================================
# === PHẦN 1: SETUP VÀ IMPORTS ===
# ==============================================================================
print("--- 🚀 Bắt đầu khởi chạy AIC25 Video Search Engine ---")
print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")

import gradio as gr
import pandas as pd
import os
import glob
import time
import numpy as np
from kaggle_secrets import UserSecretsClient
from typing import Dict, Any

# Import các module cốt lõi từ cấu trúc dự án của chúng ta
# Giả sử cấu trúc thư mục đã được thiết lập đúng
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.master_searcher import MasterSearcher
from search_core.task_analyzer import TaskType

# Import các hàm tiện ích đã được đóng gói
from utils import (
    create_video_segment,
    format_results_for_gallery,
    format_for_submission,
    generate_submission_file
)

# ==============================================================================
# === PHẦN 2: KHỞI TẠO BACKEND (SINGLETON PATTERN) ===
# ==============================================================================
print("--- Giai đoạn 2/4: Đang cấu hình và khởi tạo Backend...")

# --- Cấu hình API Key ---
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    print("--- ✅ Cấu hình Google API Key thành công! ---")
except Exception as e:
    GOOGLE_API_KEY = None
    print(f"--- ⚠️ Không tìm thấy Google API Key. Lỗi: {e} ---")

# --- Đường dẫn Dữ liệu (Constants) ---
FAISS_INDEX_PATH = '/kaggle/input/stage1/faiss.index'
RERANK_METADATA_PATH = '/kaggle/input/stage1/rerank_metadata.parquet'
VIDEO_BASE_PATH = "/kaggle/input/aic2025-batch-1-video/"
ALL_ENTITIES_PATH = "/kaggle/input/stage1/all_detection_entities.json"

# --- Sử dụng @gr.cache để đảm bảo model chỉ được load MỘT LẦN ---
@gr.cache
def get_master_searcher():
    """
    Hàm khởi tạo singleton cho toàn bộ backend.
    Gradio sẽ cache kết quả của hàm này, tránh việc load lại model mỗi khi reload UI.
    """
    print("--- Đang khởi tạo các model AI (quá trình này chỉ chạy một lần)... ---")
    all_video_files = glob.glob(os.path.join(VIDEO_BASE_PATH, "**", "*.mp4"), recursive=True)
    video_path_map = {os.path.basename(f).replace('.mp4', ''): f for f in all_video_files}

    # MasterSearcher sẽ tự quản lý việc khởi tạo các thành phần con
    basic_searcher = BasicSearcher(FAISS_INDEX_PATH, RERANK_METADATA_PATH, video_path_map)
    master_searcher = MasterSearcher(basic_searcher=basic_searcher, gemini_api_key=GOOGLE_API_KEY)
    
    return master_searcher

# Load các model ngay khi app khởi động và lưu vào biến toàn cục
master_searcher = get_master_searcher()

# ==============================================================================
# === PHẦN 3: CÁC HÀM LOGIC CHO GIAO DIỆN (UI LOGIC) ===
# ==============================================================================
print("--- Giai đoạn 3/4: Đang định nghĩa các hàm logic cho giao diện...")

def perform_search(query_text: str, num_results: int):
    """
    Hàm chính xử lý sự kiện tìm kiếm. Gọi MasterSearcher và định dạng kết quả.
    """
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "⚠️ Vui lòng nhập truy vấn và bấm Tìm kiếm.", None, "", ""

    start_time = time.time()
    
    # Gọi hàm search chính của backend
    response = master_searcher.search(query=query_text, top_k=num_results)
    
    search_time = time.time() - start_time
    
    # Định dạng kết quả cho gallery
    formatted_gallery = format_results_for_gallery(response)
    
    # Lấy thông tin phân tích
    query_analysis = response.get('query_analysis', {})
    gemini_analysis_html = f"""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 12px; color: white; margin: 10px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; color: white; display: flex; align-items: center;">
            🧠 Phân tích truy vấn AI
        </h3>
        <div style="margin-top: 15px;">
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 8px 0;">
                <strong>🎯 Đối tượng (VI):</strong> <code style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px;">{', '.join(response['query_analysis'].get('objects_vi', []))}</code>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 8px 0;">
                <strong>🌍 Đối tượng (EN):</strong> <code style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px;">{', '.join(response['query_analysis'].get('objects_en', []))}</code>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 8px 0;">
                <strong>📝 Bối cảnh:</strong> <em>"{response['query_analysis'].get('context_vi', '')}"</em>
            </div>
        </div>
    </div>
    """
        
    # Tạo thông tin thống kê
    stats_info_html =  f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; color: white; margin: 10px 0;">
        <h3 style="margin: 0; color: white;">🔄 Đang xử lý truy vấn...</h3>
        <p style="margin: 10px 0 0 0; opacity: 0.9;"> Số kết quả: <strong>{num_results}</strong></p>
    </div>
    """
    
    # Tạo thông báo trạng thái
    task_type_msg = response.get('task_type', TaskType.KIS).value
    status_msg_html = f"✅ Tìm kiếm hoàn tất trong {search_time:.2f}s. Chế độ: {task_type_msg}"
    
    return formatted_gallery, status_msg_html, response, gemini_analysis_html, stats_info_html

def on_gallery_select(evt: gr.SelectData, response_state: dict):
    """
    Xử lý khi người dùng chọn một ảnh trong gallery.
    """
    if not response_state:
        gr.Warning("Vui lòng thực hiện tìm kiếm trước khi chọn ảnh.")
        return None, "⚠️ Vui lòng thực hiện tìm kiếm trước.", ""

    task_type = response_state.get('task_type')
    results = response_state.get('results', [])
    selected_result = results[evt.index]

    if task_type == TaskType.TRAKE:
        # Xử lý hiển thị cho TRAKE (hiển thị tất cả các frame trong chuỗi)
        # TODO: Implement a more sophisticated display for TRAKE sequences
        first_frame = selected_result['sequence'][0]
        video_path = first_frame.get('video_path')
        timestamp = first_frame.get('timestamp')
        detailed_info_html = "Đây là kết quả của một chuỗi hành động..."
    else: # KIS và QNA
        video_path = selected_result.get('video_path')
        timestamp = selected_result.get('timestamp')
        detailed_info_html = "Chi tiết Keyframe..." # Copy từ bản nháp trước
    
    video_clip_path = create_video_segment(video_path, timestamp)
    clip_info_html = f"🎥 Video Clip (10 giây) từ {max(0, timestamp - 5):.1f}s"
    
    return video_clip_path, detailed_info_html, clip_info_html

def handle_submission(response_state: dict, query_id: str):
    """
    Tạo và cung cấp file nộp bài.
    """
    if not response_state or not response_state.get('results'):
        gr.Warning("Không có kết quả để tạo file nộp bài.")
        return None
    
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
        
    submission_df = format_for_submission(response_state, max_results=100)
    
    if submission_df.empty:
        gr.Warning("Không thể định dạng kết quả để nộp bài.")
        return None
              
    file_path = generate_submission_file(submission_df, query_id=query_id)
    return file_path

def clear_all():
    """
    Xóa tất cả các output trên giao diện.
    """
    return (
        [],                  # results_gallery
        "",                  # status_output
        None,                # response_state
        "",                  # gemini_analysis
        "",                  # stats_info
        None,                # video_player
        "",                  # detailed_info
        "",                  # clip_info
        "",                  # query_id_input
        None                 # submission_file_output
    )

# ==============================================================================
# === PHẦN 4: GIAO DIỆN GRADIO (UI LAYOUT) ===
# ==============================================================================
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

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🚀 AIC25 Video Search") as app:
    
    gr.HTML(app_header_html)
    
    # State để lưu trữ kết quả tìm kiếm đầy đủ giữa các lần tương tác
    response_state = gr.State()
    
     # --- Bố cục Phần Nhập liệu ---
    with gr.Row():
        with gr.Column(scale=8):
            query_input = gr.Textbox(
                label="🔍 Nhập truy vấn tìm kiếm",
                placeholder="Ví dụ: một người đàn ông mặc áo xanh đang nói chuyện điện thoại trong công viên...",
                lines=2,
                autofocus=True,
                elem_classes=["search-input"]
            )
        with gr.Column(scale=2):
            search_mode = gr.Dropdown(
                choices=["Semantic Search", "Basic CLIP Search"],
                value="Semantic Search",
                label="🎛️ Chế độ tìm kiếm",
                interactive=True
            )
    
    with gr.Row():
        with gr.Column(scale=2):
            search_button = gr.Button(
                "🚀 Tìm kiếm",
                variant="primary",
                size="lg",
                elem_classes=["search-button"]
            )
        with gr.Column(scale=2):
            num_results = gr.Slider(
                minimum=6,
                maximum=24,
                value=12,
                step=3,
                label="📊 Số kết quả",
                interactive=True
            )
        with gr.Column(scale=4):
            clear_button = gr.Button(
                "🗑️ Xóa kết quả",
                variant="secondary",
                size="lg"
            )
    
        
    # --- Bố cục Phần Hiển thị Trạng thái và Phân tích ---
    status_output = gr.HTML()
    with gr.Row():
        gemini_analysis = gr.HTML()
        stats_info = gr.HTML()
    
    # --- Bố cục Phần Kết quả ---
    with gr.Row():
        with gr.Column(scale=2):
            results_gallery = gr.Gallery(
                label="🖼️ Kết quả tìm kiếm",
                show_label=True,
                elem_id="results-gallery",
                columns=3,
                rows=4,
                object_fit="cover",
                height=600,
                allow_preview=True,
                preview=True
            )
        
        with gr.Column(scale=1):
            with gr.Row():
                video_player = gr.Video(
                    label="🎬 Video Clip",
                    height=300,
                    autoplay=True,
                    show_share_button=False
                )
            clip_info = gr.HTML()
            detailed_info = gr.HTML()

    # --- Bố cục Phần Nộp bài ---
    with gr.Accordion("💾 Tạo File Nộp Bài", open=False):
        with gr.Row():
            query_id_input = gr.Textbox(label="Nhập Query ID", placeholder="Ví dụ: query_01")
            submission_button = gr.Button("Tạo File")
        submission_file_output = gr.File(label="Tải file nộp bài của bạn")

    gr.HTML(usage_guide_html)
    gr.HTML(app_footer_html)

    # --- Liên kết các sự kiện (Event Listeners) ---
    # (Dán toàn bộ phần liên kết sự kiện từ bản nháp trước)
    search_inputs = [query_input, num_results]
    search_outputs = [results_gallery, status_output, response_state, gemini_analysis, stats_info]
    
    search_button.click(fn=perform_search, inputs=search_inputs, outputs=search_outputs)
    query_input.submit(fn=perform_search, inputs=search_inputs, outputs=search_outputs)
    
    results_gallery.select(
        fn=on_gallery_select,
        inputs=[response_state],
        outputs=[video_player, detailed_info, clip_info]
    )
    
    submission_button.click(
        fn=handle_submission,
        inputs=[response_state, query_id_input],
        outputs=[submission_file_output]
    )

    clear_outputs = [
        results_gallery, status_output, response_state, gemini_analysis, stats_info,
        video_player, detailed_info, clip_info, query_id_input, submission_file_output
    ]
    clear_button.click(fn=clear_all, inputs=[], outputs=clear_outputs)

# ==============================================================================
# === LAUNCH ỨNG DỤNG ===
# ==============================================================================
if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server ---")
    app.launch(
        share=True,
        # allowed_paths cho phép Gradio truy cập các file trong thư mục này,
        # cực kỳ quan trọng để hiển thị ảnh keyframe và video clip.
        allowed_paths=["/kaggle/input/", "/kaggle/working/"],
        debug=True, # Bật debug để xem lỗi chi tiết trên console
        show_error=True # Hiển thị lỗi trên giao diện
    )