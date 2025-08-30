print("--- 🚀 Bắt đầu khởi chạy AIC25 Battle Station v2 ---")
print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")

import gradio as gr
import pandas as pd
import numpy as np
import time
from enum import Enum

# ==============================================================================
# === PHẦN MOCK BACKEND - Giả lập các class để phát triển UI ===
# ==============================================================================
# Chúng ta sẽ thay thế phần này bằng code backend thật ở các giai đoạn sau.

class MockTaskType(Enum):
    """Enum giả lập cho các loại nhiệm vụ."""
    KIS = "Textual KIS"
    QNA = "Question Answering"
    TRAKE = "Action Keyframe Tracking"

def create_mock_kis_qna_df(num_rows=200):
    """Tạo một DataFrame giả lập cho kết quả KIS/Q&A."""
    data = {
        'keyframe_id': [f'L{np.random.randint(1, 5)}_V00{np.random.randint(1, 9)}_{i:04d}' for i in range(num_rows)],
        'video_id': [f'L{np.random.randint(1, 5)}_V00{np.random.randint(1, 9)}' for _ in range(num_rows)],
        'timestamp': np.random.uniform(10, 1000, num_rows).round(2),
        'clip_score': np.random.uniform(0.7, 0.95, num_rows).round(4),
        'object_score': np.random.uniform(0.1, 0.9, num_rows).round(4),
        'semantic_score': np.random.uniform(0.2, 0.8, num_rows).round(4),
        'final_score': np.random.uniform(0.5, 0.85, num_rows).round(4),
        'answer': [f'Câu trả lời mẫu {i}' for i in range(num_rows)]
    }
    df = pd.DataFrame(data)
    # Sắp xếp theo final_score giảm dần để giả lập kết quả thực tế
    return df.sort_values(by='final_score', ascending=False).reset_index(drop=True)

def create_mock_trake_steps(num_steps=4, num_candidates_per_step=50):
    """Tạo dữ liệu giả lập cho các bước của TRAKE."""
    all_steps = []
    for step in range(num_steps):
        step_candidates = []
        for i in range(num_candidates_per_step):
            step_candidates.append({
                'keyframe_id': f'L{np.random.randint(1, 5)}_V00{np.random.randint(1, 9)}_{i:04d}',
                'video_id': f'L{np.random.randint(1, 5)}_V00{np.random.randint(1, 9)}',
                'timestamp': np.random.uniform(10 + step * 100, 100 + step * 100),
                'final_score': np.random.uniform(0.6, 0.9),
                'thumbnail_path': '/kaggle/input/aic-2024-public-test-data-2nd/keyframes/L01_V001/000000.jpg' # Dùng ảnh placeholder
            })
        all_steps.append(step_candidates)
    return all_steps

class MockMasterSearcher:
    """Class MasterSearcher giả lập."""
    def search(self, query: str, config: dict = None):
        print(f"--- MOCK BACKEND: Nhận truy vấn '{query}' ---")
        time.sleep(2) # Giả lập thời gian xử lý
        if "nhảy" in query or "(1)" in query or "bước" in query:
            print("--- MOCK BACKEND: Phân loại là TRAKE ---")
            return {
                'task_type': MockTaskType.TRAKE,
                'query_analysis': {'task_type': 'TRAKE', 'search_context': query, 'objects_en': ['jump', 'athlete']},
                'kis_qna_candidates': pd.DataFrame(),
                'trake_step_candidates': create_mock_trake_steps(num_steps=4)
            }
        else:
            print("--- MOCK BACKEND: Phân loại là KIS/QNA ---")
            return {
                'task_type': MockTaskType.KIS,
                'query_analysis': {'task_type': 'KIS', 'search_context': query, 'objects_en': ['car', 'street']},
                'kis_qna_candidates': create_mock_kis_qna_df(200),
                'trake_step_candidates': []
            }

mock_master_searcher = MockMasterSearcher()

def perform_search(query_text: str):
    # (Hàm này giữ nguyên logic từ GĐ1)
    response = mock_master_searcher.search(query_text)
    task_type = response['task_type']
    query_analysis = response['query_analysis']
    kis_qna_candidates = response['kis_qna_candidates']
    trake_step_candidates = response['trake_step_candidates']
    analysis_summary = (f"<b>Loại nhiệm vụ:</b> {task_type.value}<br>"
                      f"<b>Bối cảnh tìm kiếm:</b> {query_analysis.get('search_context', 'N/A')}")
    if task_type == MockTaskType.TRAKE:
        return (analysis_summary, response, pd.DataFrame(), trake_step_candidates,
                f"Đã tìm thấy ứng viên cho {len(trake_step_candidates)} bước TRAKE")
    else:
        # TRẢ VỀ THÊM DataFrame để cập nhật State
        return (analysis_summary, response, kis_qna_candidates, [],
                f"Đã tìm thấy {len(kis_qna_candidates)} ứng viên KIS/QNA", kis_qna_candidates)

def on_kis_qna_select(kis_qna_df: pd.DataFrame, evt: gr.SelectData):
    """
    Hàm xử lý khi người dùng chọn một hàng trong bảng KIS/Q&A.
    """
    if evt.index is None or kis_qna_df.empty:
        return None, "Vui lòng chọn một hàng để xem chi tiết."

    # Lấy thông tin của hàng được chọn
    selected_row_index = evt.index[0] # evt.index là một tuple (row_index, col_index)
    selected_row = kis_qna_df.iloc[selected_row_index]
    
    # Tạo video clip (sử dụng mock)
    video_clip = create_mock_video_segment(selected_row['video_path'], selected_row['timestamp'])
    
    # Tạo HTML hiển thị thông tin chi tiết
    detailed_info_html = f"""
    <h4>Thông tin Chi tiết</h4>
    <ul>
        <li><b>Video ID:</b> {selected_row['video_id']}</li>
        <li><b>Keyframe ID:</b> {selected_row['keyframe_id']}</li>
        <li><b>Timestamp:</b> {selected_row['timestamp']:.2f}s</li>
        <li><b>Final Score:</b> {selected_row['final_score']:.4f}</li>
        <hr>
        <li><b>Clip Score:</b> {selected_row['clip_score']:.4f}</li>
        <li><b>Object Score:</b> {selected_row['object_score']:.4f}</li>
        <li><b>Semantic Score:</b> {selected_row['semantic_score']:.4f}</li>
        <hr>
        <li><b>Câu trả lời (VQA):</b> {selected_row['answer']}</li>
    </ul>
    """
    
    return video_clip, detailed_info_html

def update_kis_qna_view(kis_qna_df: pd.DataFrame, sort_by: str, filter_video: str):
    """
    Hàm để lọc và sắp xếp lại bảng KIS/Q&A.
    """
    if kis_qna_df is None or kis_qna_df.empty:
        return pd.DataFrame() # Trả về DF rỗng nếu không có dữ liệu

    # Sao chép để không thay đổi state gốc
    df_processed = kis_qna_df.copy()
    
    # Lọc theo video ID
    if filter_video and filter_video.strip():
        df_processed = df_processed[df_processed['video_id'].str.contains(filter_video.strip(), case=False)]
        
    # Sắp xếp
    if sort_by and sort_by in df_processed.columns:
        # Giả sử điểm cao hơn là tốt hơn
        is_ascending = not ('score' in sort_by)
        df_processed = df_processed.sort_values(by=sort_by, ascending=is_ascending)
        
    return df_processed

def add_to_submission_list(submission_list: pd.DataFrame, kis_qna_df: pd.DataFrame, evt: gr.SelectData):
    """
    Thêm hàng đang được chọn vào danh sách nộp bài.
    """
    if evt.index is None or kis_qna_df.empty:
        gr.Warning("Chưa có ứng viên nào được chọn!")
        return submission_list

    selected_row_index = evt.index[0]
    selected_row = kis_qna_df.iloc[[selected_row_index]] # Lấy dưới dạng DataFrame
    
    if submission_list is None:
        submission_list = pd.DataFrame()

    # Thêm hàng mới vào cuối danh sách
    updated_list = pd.concat([submission_list, selected_row]).reset_index(drop=True)
    gr.Info(f"Đã thêm {selected_row['keyframe_id'].iloc[0]} vào danh sách nộp bài!")
    
    return updated_list

# ==============================================================================
# === BẮT ĐẦU PHẦN GIAO DIỆN GRADIO - PHIÊN BẢN NÂNG CẤP GĐ2 ===
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="AIC25 Battle Station v2") as app:
    
    # --- Khai báo các State ---
    full_response_state = gr.State()
    # **QUAN TRỌNG**: State cho DataFrame gốc, không bị thay đổi bởi lọc/sắp xếp
    kis_qna_df_state = gr.State()
    trake_steps_state = gr.State()
    # **STATE MỚI**: State cho danh sách nộp bài
    submission_list_state = gr.State(pd.DataFrame())

    gr.HTML("<h1>🚀 AIC25 Battle Station v2 - Tối ưu Hiệu suất</h1>")

    with gr.Row(variant='panel'):
        # --- KHU VỰC 1: BẢNG ĐIỀU KHIỂN & TRUY VẤN (CỘT TRÁI) ---
        with gr.Column(scale=2):
            gr.Markdown("### 1. Bảng điều khiển")
            with gr.Group():
                query_input = gr.Textbox(label="Nhập truy vấn", lines=2, placeholder="Ví dụ: một người đang nhảy qua xà...")
                search_button = gr.Button("Phân tích & Truy xuất Sơ bộ", variant="primary")
                analysis_summary_output = gr.HTML(label="Tóm tắt Phân tích AI")

            gr.Markdown("### 2. Không gian Làm việc")
            with gr.Tabs():
                with gr.TabItem("Xác thực Nhanh KIS/Q&A"):
                    status_kis_qna = gr.Markdown("Chưa có dữ liệu.")
                    # **CÁC WIDGET LỌC/SẮP XẾP MỚI**
                    with gr.Row():
                        sort_dropdown = gr.Dropdown(
                            label="Sắp xếp theo",
                            choices=['final_score', 'clip_score', 'object_score', 'semantic_score', 'timestamp'],
                            value='final_score'
                        )
                        filter_textbox = gr.Textbox(label="Lọc theo Video ID")
                    
                    kis_qna_table = gr.DataFrame(
                        label="Top 200 Ứng viên (Click vào hàng để xem chi tiết)",
                        headers=['video_id', 'timestamp', 'final_score', 'clip_score', 'object_score', 'semantic_score'],
                        datatype=['str', 'number', 'number', 'number', 'number', 'number'],
                        row_count=(10, "dynamic"),
                        col_count=(6, "fixed"),
                        interactive=True
                    )

                with gr.TabItem("Bàn Lắp ráp Chuỗi TRAKE"):
                    # ... (giữ nguyên từ GĐ1)
                    trake_workspace_placeholder = gr.HTML("Khu vực này sẽ hiển thị các cột ứng viên cho từng bước TRAKE.")

        # --- KHU VỰC 2 & 3: XẾP HẠNG & CHI TIẾT (CỘT PHẢI) ---
        with gr.Column(scale=1):
            gr.Markdown("### 3. Bảng Xếp hạng & Xem chi tiết")
            with gr.Tabs():
                with gr.TabItem("Xem chi tiết"):
                    # **NÚT THÊM VÀO DANH SÁCH MỚI**
                    add_to_submission_button = gr.Button("➕ Thêm ứng viên này vào Danh sách Nộp bài")
                    video_player = gr.Video(label="Video Clip Preview")
                    detailed_info = gr.HTML("Thông tin chi tiết sẽ hiện ở đây khi bạn chọn một ứng viên.")
                
                with gr.TabItem("Danh sách Nộp bài (Top 100)"):
                    submission_list_table = gr.DataFrame(
                        label="Danh sách này sẽ được sắp xếp lại bằng tay ở GĐ4",
                        interactive=True
                    )
            
            with gr.Group():
                 gr.Markdown("#### Nộp bài")
                 query_id_input = gr.Textbox(label="Query ID", placeholder="query_01")
                 submission_button = gr.Button("Tạo File Nộp bài")

    # ==============================================================================
    # === KẾT NỐI CÁC SỰ KIỆN TƯƠNG TÁC - PHIÊN BẢN GĐ2 ===
    # ==============================================================================
    
    # 1. Sự kiện Tìm kiếm chính (Cập nhật để điền vào state DataFrame gốc)
    search_button.click(
        fn=perform_search,
        inputs=[query_input],
        outputs=[
            analysis_summary_output, full_response_state,
            kis_qna_table, # Cập nhật bảng hiển thị
            trake_steps_state, status_kis_qna,
            kis_qna_df_state # **QUAN TRỌNG**: Lưu DataFrame gốc vào State
        ]
    )

    # 2. Sự kiện Chọn một hàng trong bảng KIS/Q&A
    kis_qna_table.select(
        fn=on_kis_qna_select,
        inputs=[kis_qna_table], # Lấy dữ liệu từ bảng đang hiển thị
        outputs=[video_player, detailed_info]
    )

    # 3. Sự kiện thay đổi các widget lọc hoặc sắp xếp
    sort_dropdown.change(
        fn=update_kis_qna_view,
        inputs=[kis_qna_df_state, sort_dropdown, filter_textbox], # Dùng state gốc để tính toán
        outputs=[kis_qna_table] # Chỉ cập nhật bảng hiển thị
    )
    filter_textbox.submit(
        fn=update_kis_qna_view,
        inputs=[kis_qna_df_state, sort_dropdown, filter_textbox],
        outputs=[kis_qna_table]
    )

    # 4. Sự kiện bấm nút "Thêm vào Danh sách Nộp bài"
    add_to_submission_button.click(
        fn=add_to_submission_list,
        inputs=[submission_list_state, kis_qna_table], # Truyền vào list hiện tại và bảng đang hiển thị
        outputs=[submission_list_table] # Cập nhật bảng danh sách nộp bài
    # `_js` và `evt: gr.SelectData` được Gradio xử lý tự động
    ).then(None, _js="(evt_data) => { return null }", inputs=None, outputs=[kis_qna_table])

if __name__ == "__main__":
    app.launch(debug=True, share=True)