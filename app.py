# /AIC25_Video_Search_Engine/app.py

print("--- 🚀 Bắt đầu khởi chạy AIC25 Video Search Engine ---")
print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")

import gradio as gr
from functools import partial

# Local imports from our refactored files
from backend_loader import initialize_backend
from ui_layout import build_ui
import event_handlers as handlers

# --- Giai đoạn 2: Khởi tạo Backend ---
# Việc này chỉ chạy một lần khi ứng dụng bắt đầu
master_searcher = initialize_backend()

# --- Giai đoạn 3: Xây dựng Giao diện & Kết nối Logic ---
print("--- Giai đoạn 3/4: Đang xây dựng giao diện và kết nối sự kiện...")

# Xây dựng UI và lấy về một từ điển chứa tất cả các components
app, ui = build_ui()

# Sử dụng `partial` để "gói" hàm `perform_search` cùng với `master_searcher`.
# Điều này cho phép chúng ta truyền instance backend vào hàm xử lý sự kiện
# một cách gọn gàng mà không cần dùng biến global.
search_with_backend = partial(handlers.perform_search, master_searcher=master_searcher)

# --- KẾT NỐI SỰ KIỆN (WIRING) ---
# Phần này kết nối các tương tác trên UI (click, submit) với các hàm xử lý logic.

# 1. Sự kiện Tìm kiếm chính
# Tập hợp tất cả các input cần thiết cho hàm tìm kiếm
search_inputs = [
    ui["query_input"],
    ui["num_results"],
    ui["kis_retrieval_slider"],
    ui["vqa_candidates_slider"],
    ui["vqa_retrieval_slider"],
    ui["trake_candidates_per_step_slider"],
    ui["trake_max_sequences_slider"],
    ui["w_clip_slider"],
    ui["w_obj_slider"],
    ui["w_semantic_slider"],
    ui["lambda_mmr_slider"],
]
# Tập hợp tất cả các output mà hàm tìm kiếm sẽ cập nhật
search_outputs = [
    ui["results_gallery"],
    ui["status_output"],
    ui["response_state"],
    ui["gemini_analysis"],
    ui["stats_info"],
    ui["gallery_items_state"],
    ui["current_page_state"],
    ui["page_info_display"],
]
ui["search_button"].click(fn=search_with_backend, inputs=search_inputs, outputs=search_outputs)
ui["query_input"].submit(fn=search_with_backend, inputs=search_inputs, outputs=search_outputs)

# 2. Sự kiện Phân trang
page_outputs = [ui["results_gallery"], ui["current_page_state"], ui["page_info_display"]]
ui["prev_page_button"].click(
    fn=handlers.update_gallery_page,
    inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("◀️ Trang trước", visible=False)],
    outputs=page_outputs
)
ui["next_page_button"].click(
    fn=handlers.update_gallery_page,
    inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("▶️ Trang sau", visible=False)],
    outputs=page_outputs
)

# 3. Sự kiện Chọn một ảnh trong Gallery để phân tích
analysis_outputs = [
    ui["selected_image_display"],
    ui["video_player"],
    ui["scores_display"],
    ui["vqa_answer_display"],
    ui["transcript_display"],
    ui["selected_candidate_for_submission"],
    ui["detailed_info"],
    ui["clip_info"],
]
ui["results_gallery"].select(
    fn=handlers.on_gallery_select,
    inputs=[ui["response_state"], ui["current_page_state"]],
    outputs=analysis_outputs
)

# 4. Sự kiện trong Vùng Nộp bài
submission_list_outputs = [
    ui["submission_list_display"],
    ui["submission_list_state"],
    ui["submission_list_selector"],
]
add_inputs = [
    ui["submission_list_state"],
    ui["selected_candidate_for_submission"],
    ui["response_state"],
]
ui["add_top_button"].click(
    fn=handlers.add_to_submission_list,
    inputs=add_inputs + [gr.Textbox("top", visible=False)],
    outputs=submission_list_outputs
)
ui["add_bottom_button"].click(
    fn=handlers.add_to_submission_list,
    inputs=add_inputs + [gr.Textbox("bottom", visible=False)],
    outputs=submission_list_outputs
)
ui["clear_submission_button"].click(
    fn=handlers.clear_submission_list,
    inputs=[],
    outputs=submission_list_outputs
)

# Sự kiện tinh chỉnh danh sách nộp bài
modify_inputs = [ui["submission_list_state"], ui["submission_list_selector"]]
ui["move_up_button"].click(
    fn=handlers.modify_submission_list,
    inputs=modify_inputs + [gr.Textbox("move_up", visible=False)],
    outputs=submission_list_outputs
)
ui["move_down_button"].click(
    fn=handlers.modify_submission_list,
    inputs=modify_inputs + [gr.Textbox("move_down", visible=False)],
    outputs=submission_list_outputs
)
ui["remove_button"].click(
    fn=handlers.modify_submission_list,
    inputs=modify_inputs + [gr.Textbox("remove", visible=False)],
    outputs=submission_list_outputs
)

# Sự kiện tạo file nộp bài
ui["submission_button"].click(
    fn=handlers.handle_submission,
    inputs=[ui["submission_list_state"], ui["query_id_input"]],
    outputs=[ui["submission_file_output"]]
)

# 5. Sự kiện Xóa tất cả (Clear All)
# Định nghĩa rõ ràng danh sách các component cần được reset
clear_outputs = [
    # Cột trái
    ui["results_gallery"], ui["status_output"], ui["response_state"],
    ui["gemini_analysis"], ui["stats_info"], ui["gallery_items_state"],
    ui["current_page_state"], ui["page_info_display"],
    # Cột phải - Trạm phân tích
    ui["selected_image_display"], ui["video_player"], ui["scores_display"],
    ui["vqa_answer_display"], ui["transcript_display"],
    ui["selected_candidate_for_submission"], ui["detailed_info"], ui["clip_info"],
    # Cột phải - Vùng nộp bài
    ui["submission_list_display"], ui["submission_list_state"],
    ui["submission_list_selector"], ui["query_id_input"], ui["submission_file_output"]
]
ui["clear_button"].click(fn=handlers.clear_all, inputs=None, outputs=clear_outputs)


# --- Giai đoạn 4: Khởi chạy App Server ---
if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server ---")
    app.launch(
        share=True,
        allowed_paths=["/kaggle/input/", "/kaggle/working/"],
        debug=True, # Bật debug để xem lỗi chi tiết trên console
        show_error=True # Hiển thị lỗi trên giao diện
    )