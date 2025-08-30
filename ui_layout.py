# /AIC25_Video_Search_Engine/ui_layout.py

import gradio as gr

# ... (custom_css, app_header_html, app_footer_html giữ nguyên như cũ) ...
custom_css = """
/* Ẩn footer mặc định của Gradio */
footer {display: none !important}
/* Custom styling cho gallery */
.gallery { border-radius: 12px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.05) !important; }
/* Đảm bảo gallery chính có thể cuộn được */
#results-gallery > .gradio-gallery { height: 700px !important; overflow-y: auto !important; }
/* Animation cho buttons */
.gradio-button { transition: all 0.2s ease !important; border-radius: 20px !important; font-weight: 600 !important; }
.gradio-button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; }
/* Custom textbox styling */
.gradio-textbox { border-radius: 10px !important; border: 1px solid #e0e0e0 !important; transition: all 0.2s ease !important; }
.gradio-textbox:focus { border-color: #667eea !important; box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important; }
/* Video player styling */
video { border-radius: 12px !important; }
/* Hiệu ứng hover cho ảnh trong gallery */
.gallery img { transition: transform 0.2s ease !important; border-radius: 8px !important; }
.gallery img:hover { transform: scale(1.04) !important; }
/* Tùy chỉnh thanh cuộn */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); }
"""

app_header_html = """
<div style="text-align: center; max-width: 1200px; margin: 0 auto 25px auto;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px 20px; border-radius: 20px; color: white; box-shadow: 0 8px 30px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">🚀 AIC25 Video Search Engine</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">Hệ thống tìm kiếm video thông minh, vận hành bởi AI</p>
    </div>
</div>
"""

app_footer_html = """
<div style="text-align: center; margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 12px;">
    <p style="margin: 0; color: #6c757d;">AIC25 Video Search Engine - Powered by Semantic Search, Object Detection & Generative AI</p>
</div>
"""
# === KEY CHANGE 1: Hàm build_ui bây giờ chấp nhận một tham số là một hàm khác ===
def build_ui(connect_events_fn):
    """
    Xây dựng toàn bộ giao diện người dùng và kết nối các sự kiện.

    Args:
        connect_events_fn (function): Một hàm nhận vào từ điển `components`
                                     và thực hiện việc kết nối sự kiện (.click, .select...).
    
    Returns:
        (gr.Blocks): Đối tượng ứng dụng Gradio đã sẵn sàng để launch.
    """
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🚀 AIC25 Video Search") as app:
        
        # --- Khai báo States ---
        response_state = gr.State()
        gallery_items_state = gr.State([])
        current_page_state = gr.State(1)
        submission_list_state = gr.State([])
        selected_candidate_for_submission = gr.State()

        # --- BỐ CỤC CHÍNH (giữ nguyên) ---
        gr.HTML(app_header_html)
        
        with gr.Row(variant='panel'):
            # --- CỘT TRÁI ---
            with gr.Column(scale=2):
                # ... (Toàn bộ layout cột trái giữ nguyên)
                gr.Markdown("### 1. Nhập truy vấn")
                query_input = gr.Textbox(label="🔍 Nhập mô tả chi tiết bằng tiếng Việt", placeholder="Ví dụ: một người phụ nữ mặc váy đỏ đang nói về việc bảo tồn rùa biển...", lines=2, autofocus=True)
                with gr.Row():
                    search_button = gr.Button("🚀 Tìm kiếm", variant="primary", size="lg")
                    clear_button = gr.Button("🗑️ Xóa tất cả", variant="secondary", size="lg")
                num_results = gr.Slider(minimum=20, maximum=100, value=100, step=10, label="📊 Số kết quả tối đa")

                with gr.Accordion("⚙️ Tùy chỉnh Nâng cao", open=False):
                    with gr.Tabs():
                        with gr.TabItem("KIS / Chung"):
                            kis_retrieval_slider = gr.Slider(minimum=50, maximum=500, value=100, step=25, label="Số ứng viên KIS ban đầu")
                        with gr.TabItem("VQA"):
                            vqa_retrieval_slider = gr.Slider(minimum=50, maximum=500, value=200, step=25, label="Số ứng viên VQA ban đầu")
                            vqa_candidates_slider = gr.Slider(minimum=3, maximum=30, value=8, step=1, label="Số ứng viên VQA để hỏi đáp")
                        with gr.TabItem("TRAKE"):
                            trake_candidates_per_step_slider = gr.Slider(minimum=5, maximum=30, value=15, step=1, label="Số ứng viên mỗi bước (TRAKE)")
                            trake_max_sequences_slider = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="Số chuỗi kết quả tối đa (TRAKE)")
                        with gr.TabItem("⚖️ Trọng số & Đa dạng (MMR)"):
                            w_clip_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="w_clip (Thị giác)")
                            w_obj_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_obj (Đối tượng)")
                            w_semantic_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_semantic (Ngữ nghĩa)")
                            lambda_mmr_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="λ - MMR (0.0=Đa dạng, 1.0=Liên quan)")

                status_output = gr.HTML()
                with gr.Row():
                    gemini_analysis = gr.HTML()
                    stats_info = gr.HTML()

                gr.Markdown("### 2. Kết quả tìm kiếm")
                with gr.Row(equal_height=True, variant='compact'):
                    prev_page_button = gr.Button("◀️ Trang trước")
                    page_info_display = gr.Markdown("Trang 1 / 1", elem_id="page-info")
                    next_page_button = gr.Button("▶️ Trang sau")
                results_gallery = gr.Gallery(label="Click vào một ảnh để phân tích", show_label=True, elem_id="results-gallery", columns=5, object_fit="contain", height=700, allow_preview=False)
            
            # --- CỘT PHẢI ---
            with gr.Column(scale=1):
                # ... (Toàn bộ layout cột phải giữ nguyên)
                gr.Markdown("### 3. Trạm Phân tích")
                selected_image_display = gr.Image(label="Ảnh Keyframe Được chọn", type="filepath")
                video_player = gr.Video(label="🎬 Clip 10 giây", autoplay=True)
                with gr.Tabs():
                    with gr.TabItem("📊 Phân tích & Điểm số"):
                        detailed_info = gr.HTML() 
                        scores_display = gr.DataFrame(headers=["Metric", "Value"], label="Bảng điểm chi tiết")
                    with gr.TabItem("💬 VQA & Transcript"):
                        vqa_answer_display = gr.Textbox(label="Câu trả lời VQA", interactive=False, lines=5)
                        transcript_display = gr.Textbox(label="📝 Transcript", lines=8, interactive=False)
                clip_info = gr.HTML() 

                gr.Markdown("### 4. Vùng Nộp bài")
                with gr.Row():
                    add_top_button = gr.Button("➕ Thêm vào Top 1", variant="primary")
                    add_bottom_button = gr.Button("➕ Thêm vào cuối")
                with gr.Tabs():
                    with gr.TabItem("📋 Danh sách & Tinh chỉnh"):
                        submission_list_display = gr.Textbox(label="Thứ tự Nộp bài (Top 1 ở trên cùng)", lines=12, interactive=False, value="Chưa có kết quả nào.")
                        submission_list_selector = gr.Dropdown(label="Chọn mục để thao tác", choices=[], interactive=True)
                        with gr.Row():
                            move_up_button = gr.Button("⬆️ Lên")
                            move_down_button = gr.Button("⬇️ Xuống")
                            remove_button = gr.Button("🗑️ Xóa", variant="stop")
                        clear_submission_button = gr.Button("💥 Xóa toàn bộ danh sách")
                    with gr.TabItem("💾 Xuất File"):
                        query_id_input = gr.Textbox(label="Nhập Query ID", placeholder="Ví dụ: query_01")
                        submission_button = gr.Button("💾 Tạo File CSV Nộp bài")
                        submission_file_output = gr.File(label="Tải file nộp bài tại đây")

        gr.HTML(app_footer_html)
        
        # Gom tất cả components vào một dictionary
        components = {
            # States
            "response_state": response_state, "gallery_items_state": gallery_items_state,
            "current_page_state": current_page_state, "submission_list_state": submission_list_state,
            "selected_candidate_for_submission": selected_candidate_for_submission,
            # Cột Trái - Inputs & Controls
            "query_input": query_input, "search_button": search_button, "num_results": num_results,
            "clear_button": clear_button, "kis_retrieval_slider": kis_retrieval_slider,
            "vqa_retrieval_slider": vqa_retrieval_slider, "vqa_candidates_slider": vqa_candidates_slider,
            "trake_candidates_per_step_slider": trake_candidates_per_step_slider,
            "trake_max_sequences_slider": trake_max_sequences_slider, "w_clip_slider": w_clip_slider,
            "w_obj_slider": w_obj_slider, "w_semantic_slider": w_semantic_slider, "lambda_mmr_slider": lambda_mmr_slider,
            # Cột Trái - Outputs & Display
            "status_output": status_output, "gemini_analysis": gemini_analysis, "stats_info": stats_info,
            "prev_page_button": prev_page_button, "page_info_display": page_info_display,
            "next_page_button": next_page_button, "results_gallery": results_gallery,
            # Cột Phải - Trạm Phân tích
            "selected_image_display": selected_image_display, "video_player": video_player,
            "detailed_info": detailed_info, "scores_display": scores_display,
            "vqa_answer_display": vqa_answer_display, "transcript_display": transcript_display, "clip_info": clip_info,
            # Cột Phải - Vùng Nộp bài
            "add_top_button": add_top_button, "add_bottom_button": add_bottom_button,
            "submission_list_display": submission_list_display, "submission_list_selector": submission_list_selector,
            "move_up_button": move_up_button, "move_down_button": move_down_button, "remove_button": remove_button,
            "clear_submission_button": clear_submission_button, "query_id_input": query_id_input,
            "submission_button": submission_button, "submission_file_output": submission_file_output,
        }

        # === KEY CHANGE 2: Gọi hàm kết nối sự kiện được truyền vào, ngay bên trong context "with" ===
        connect_events_fn(components)

    # Hàm bây giờ chỉ trả về đối tượng app
    return app