# /AIC25_Video_Search_Engine/app.py

print("--- 🚀 Bắt đầu khởi chạy AIC25 Video Search Engine ---")
print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")

import gradio as gr
from functools import partial

# Local imports
from backend_loader import initialize_backend
from ui_layout import build_ui
import event_handlers as handlers

# --- Giai đoạn 2: Khởi tạo Backend ---
master_searcher = initialize_backend()

# --- Giai đoạn 3: Xây dựng Giao diện & Kết nối Logic ---
print("--- Giai đoạn 3/4: Đang xây dựng giao diện và kết nối sự kiện...")

# Sử dụng `partial` để gói hàm tìm kiếm với instance backend
search_with_backend = partial(handlers.perform_search, master_searcher=master_searcher)


# === KEY CHANGE 1: Toàn bộ logic kết nối sự kiện được chuyển vào hàm này ===
def connect_event_listeners(ui_components):
    """
    Kết nối tất cả các sự kiện của component UI với các hàm xử lý tương ứng.
    Hàm này sẽ được truyền vào `build_ui` để được gọi trong context của gr.Blocks.
    
    Args:
        ui_components (dict): Từ điển chứa tất cả các component của UI.
    """
    ui = ui_components # Đổi tên cho ngắn gọn

    # 1. Sự kiện Tìm kiếm chính
    search_inputs = [
        ui["query_input"], ui["num_results"], ui["kis_retrieval_slider"],
        ui["vqa_candidates_slider"], ui["vqa_retrieval_slider"], ui["trake_candidates_per_step_slider"],
        ui["trake_max_sequences_slider"], ui["w_clip_slider"], ui["w_obj_slider"],
        ui["w_semantic_slider"], ui["lambda_mmr_slider"],
    ]
    search_outputs = [
        ui["results_gallery"], ui["status_output"], ui["response_state"],
        ui["gemini_analysis"], ui["stats_info"], ui["gallery_items_state"],
        ui["current_page_state"], ui["page_info_display"],
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

    # 3. Sự kiện Chọn một ảnh trong Gallery
    analysis_outputs = [
        ui["selected_image_display"], ui["video_player"], ui["scores_display"],
        ui["vqa_answer_display"], ui["transcript_display"], ui["selected_candidate_for_submission"],
        ui["detailed_info"], ui["clip_info"],
    ]
    ui["results_gallery"].select(
        fn=handlers.on_gallery_select,
        inputs=[ui["response_state"], ui["current_page_state"]],
        outputs=analysis_outputs
    )

    # 4. Sự kiện trong Vùng Nộp bài
    submission_list_outputs = [
        ui["submission_list_display"], ui["submission_list_state"], ui["submission_list_selector"],
    ]
    add_inputs = [
        ui["submission_list_state"], ui["selected_candidate_for_submission"], ui["response_state"],
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
    ui["clear_submission_button"].click(fn=handlers.clear_submission_list, inputs=[], outputs=submission_list_outputs)

    modify_inputs = [ui["submission_list_state"], ui["submission_list_selector"]]
    ui["move_up_button"].click(fn=handlers.modify_submission_list, inputs=modify_inputs + [gr.Textbox("move_up", visible=False)], outputs=submission_list_outputs)
    ui["move_down_button"].click(fn=handlers.modify_submission_list, inputs=modify_inputs + [gr.Textbox("move_down", visible=False)], outputs=submission_list_outputs)
    ui["remove_button"].click(fn=handlers.modify_submission_list, inputs=modify_inputs + [gr.Textbox("remove", visible=False)], outputs=submission_list_outputs)

    ui["submission_button"].click(fn=handlers.handle_submission, inputs=[ui["submission_list_state"], ui["query_id_input"]], outputs=[ui["submission_file_output"]])

    # 5. Sự kiện Xóa tất cả
    clear_outputs = list(ui.values())
    ui["clear_button"].click(fn=handlers.clear_all, inputs=None, outputs=clear_outputs)


# === KEY CHANGE 2: Gọi hàm build_ui và truyền hàm kết nối sự kiện vào đó ===
app = build_ui(connect_event_listeners)


# --- Giai đoạn 4: Khởi chạy App Server ---
if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server ---")
    app.launch(
        share=True,
        allowed_paths=["/kaggle/input/", "/kaggle/working/"],
        debug=True,
        show_error=True
    )