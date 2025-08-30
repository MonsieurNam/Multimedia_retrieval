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
    
def create_mock_trake_steps(num_steps=4, num_candidates_per_step=10):
    """Nâng cấp để trả về DataFrame cho mỗi bước."""
    all_steps_dfs = []
    base_timestamp = 100
    for step in range(num_steps):
        data = {
            'keyframe_id': [f'L01_V001_{step}_{i:03d}' for i in range(num_candidates_per_step)],
            'video_id': ['L01_V001'] * num_candidates_per_step,
            'timestamp': np.round(np.sort(np.random.uniform(base_timestamp, base_timestamp + 50, num_candidates_per_step)), 2),
            'final_score': np.round(np.random.uniform(0.6, 0.9, num_candidates_per_step), 4),
            'video_path': ['/kaggle/input/aic-2024-public-test-data-2nd/videos/L01_V001.mp4'] * num_candidates_per_step
        }
        all_steps_dfs.append(pd.DataFrame(data))
        base_timestamp += 100 # Đảm bảo các bước sau có timestamp lớn hơn
    return all_steps_dfs

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
        time.sleep(1)
        # Luôn trả về kết quả TRAKE để test
        print("--- MOCK BACKEND: Luôn trả về dữ liệu TRAKE để test Giai đoạn 3 ---")
        return {
            'task_type': MockTaskType.TRAKE,
            'query_analysis': {'task_type': 'TRAKE', 'search_context': query, 'sub_queries': ["bước 1", "bước 2", "bước 3", "bước 4"]},
            'kis_qna_candidates': pd.DataFrame(),
            'trake_step_candidates': create_mock_trake_steps(num_steps=4)
        }

mock_master_searcher = MockMasterSearcher()

def create_mock_video_segment(video_path, timestamp):
    return '/kaggle/input/aic-2024-public-test-data-2nd/videos/L01_V001.mp4'

def perform_search(query_text: str):
    """
    Hàm xử lý sự kiện chính: gọi backend và đổ dữ liệu vào các State.
    *** PHIÊN BẢN SỬA LỖI: Đảm bảo trả về đúng 6 giá trị. ***
    """
    response = mock_master_searcher.search(query_text)
    task_type = response['task_type']
    query_analysis = response['query_analysis']
    kis_qna_candidates = response['kis_qna_candidates']
    trake_step_candidates = response['trake_step_candidates']
    analysis_summary = (f"<b>Loại nhiệm vụ:</b> {task_type.value}<br>"
                      f"<b>Bối cảnh tìm kiếm:</b> {query_analysis.get('search_context', 'N/A')}")

    if task_type == MockTaskType.TRAKE:
        status_msg = f"Đã tìm thấy ứng viên cho {len(trake_step_candidates)} bước TRAKE. Bắt đầu lắp ráp chuỗi."
        
        # =======================================================
        # === SỬA LỖI TẠI ĐÂY ===
        # Thêm pd.DataFrame() vào cuối để khớp với 6 outputs
        # =======================================================
        return (
            analysis_summary,           # 1. analysis_summary_output
            response,                   # 2. full_response_state
            pd.DataFrame(),             # 3. kis_qna_table (xóa trắng)
            trake_step_candidates,      # 4. trake_steps_state
            status_msg,                 # 5. status_kis_qna
            pd.DataFrame()              # 6. kis_qna_df_state (xóa trắng)
        )
        # =======================================================

    else: # KIS hoặc QNA
        status_msg = f"Đã tìm thấy {len(kis_qna_candidates)} ứng viên KIS/QNA"
        
        # Nhánh này đã đúng 6 giá trị, giữ nguyên
        return (
            analysis_summary,           # 1. analysis_summary_output
            response,                   # 2. full_response_state
            kis_qna_candidates,         # 3. kis_qna_table
            [],                         # 4. trake_steps_state (xóa trắng)
            status_msg,                 # 5. status_kis_qna
            kis_qna_candidates          # 6. kis_qna_df_state
        )

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
    *** PHIÊN BẢN CẬP NHẬT: Trả về thêm giá trị để xóa lựa chọn (deselect). ***
    """
    if evt.index is None or kis_qna_df.empty:
        gr.Warning("Chưa có ứng viên nào được chọn!")
        # Trả về danh sách gốc và không thay đổi lựa chọn
        return submission_list, None

    selected_row_index = evt.index[0]
    selected_row = kis_qna_df.iloc[[selected_row_index]] # Lấy dưới dạng DataFrame
    
    if submission_list is None:
        submission_list = pd.DataFrame()

    # Thêm hàng mới vào cuối danh sách
    updated_list = pd.concat([submission_list, selected_row]).reset_index(drop=True)
    gr.Info(f"Đã thêm {selected_row['keyframe_id'].iloc[0]} vào danh sách nộp bài!")
    
    # **THAY ĐỔI QUAN TRỌNG**: Trả về 2 giá trị:
    # 1. DataFrame đã cập nhật cho submission_list_table.
    # 2. `None` để xóa lựa chọn trong kis_qna_table.
    return updated_list, None

def build_trake_workspace(trake_steps_data):
    """
    *** HÀM SỬA LỖI 1 ***
    Tạo hoặc cập nhật các giá trị cho các component trong không gian làm việc TRAKE.
    Hàm này sẽ trả về một tuple các giá trị, mỗi giá trị cho một component.
    """
    MAX_STEPS = 6 # Phải khớp với số component đã tạo trong UI
    outputs = []
    
    # Dữ liệu đầu vào là list các DataFrame
    num_steps = len(trake_steps_data) if trake_steps_data else 0

    for i in range(MAX_STEPS):
        if i < num_steps:
            # Nếu có dữ liệu cho bước này
            outputs.append(gr.Markdown(f"<h4>Bước {i+1}</h4>", visible=True))
            outputs.append(gr.DataFrame(trake_steps_data[i], visible=True))
        else:
            # Nếu không, ẩn component đi
            outputs.append(gr.Markdown(visible=False))
            outputs.append(gr.DataFrame(visible=False))
            
    # Trả về một tuple, Gradio sẽ tự động giải nén nó vào các output
    return tuple(outputs)
    
def update_current_sequence(current_sequence: pd.DataFrame, step_index: int, all_steps_data: list, evt: gr.SelectData):
    """
    Hàm chính xử lý logic "Click-to-Add" cho TRAKE.
    """
    if evt.index is None or not all_steps_data or step_index >= len(all_steps_data):
        return current_sequence, "Lỗi: Dữ liệu không hợp lệ."

    selected_row_index = evt.index[0]
    df_step = all_steps_data[step_index] # Đây là list của các DataFrame
    selected_row = df_step.iloc[[selected_row_index]]
    
    if current_sequence is None:
        current_sequence = pd.DataFrame()

    # Thêm cột 'step' để biết frame này thuộc bước nào
    selected_row['step'] = step_index + 1
    
    # Nối vào chuỗi hiện tại và sắp xếp lại theo bước
    updated_sequence = pd.concat([current_sequence, selected_row]).sort_values(by='step').reset_index(drop=True)
    
    # Xác thực chuỗi
    is_valid, validation_msg = validate_sequence(updated_sequence)
    
    return updated_sequence, validation_msg

def validate_sequence(sequence_df: pd.DataFrame):
    """Kiểm tra xem chuỗi có hợp lệ không (cùng video, timestamp tăng dần)."""
    if sequence_df.empty or len(sequence_df) <= 1:
        return True, "✅ Chuỗi hợp lệ (1 bước)."

    # Kiểm tra cùng video
    if sequence_df['video_id'].nunique() > 1:
        return False, "❌ Lỗi: Các bước phải cùng một video!"

    # Kiểm tra timestamp tăng dần
    if not sequence_df['timestamp'].is_monotonic_increasing:
        return False, "❌ Lỗi: Timestamp phải tăng dần!"

    return True, f"✅ Chuỗi hợp lệ ({len(sequence_df)} bước)."

def clear_current_sequence():
    """Xóa chuỗi đang xây dựng."""
    return pd.DataFrame(), "Đã xóa chuỗi hiện tại."
    
def add_sequence_to_submission(submission_list: pd.DataFrame, current_sequence: pd.DataFrame):
    """
    *** HÀM SỬA LỖI 2 ***
    Thêm chuỗi hiện tại (đã được xác thực) vào danh sách nộp bài.
    """
    is_valid, msg = validate_sequence(current_sequence)
    if not is_valid:
        gr.Warning(f"Không thể thêm chuỗi không hợp lệ! {msg}")
        return submission_list
    if current_sequence.empty:
        gr.Warning("Chuỗi đang xây dựng rỗng!")
        return submission_list

    # **SỬA LỖI TẠI ĐÂY**: Chuyển đổi cột 'final_score' sang dạng số, ép lỗi thành NaN
    scores = pd.to_numeric(current_sequence['final_score'], errors='coerce')
    # Tính trung bình, bỏ qua các giá trị NaN
    mean_score = scores.mean()

    submission_row = { 'task_type': ['TRAKE'], 'final_score': [mean_score] }
    submission_row['video_id'] = [current_sequence['video_id'].iloc[0]]
    for i, row in current_sequence.iterrows():
        submission_row[f'frame_moment_{i+1}'] = [row['keyframe_id']]
    
    submission_df_row = pd.DataFrame(submission_row)

    if submission_list is None:
        submission_list = pd.DataFrame()
        
    updated_list = pd.concat([submission_list, submission_df_row]).reset_index(drop=True)
    gr.Info(f"Đã thêm chuỗi video {submission_row['video_id'][0]} vào danh sách nộp bài!")
    
    return updated_list

# ==============================================================================
# === BẮT ĐẦU PHẦN GIAO DIỆN GRADIO - PHIÊN BẢN NÂNG CẤP GĐ2 ===
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="AIC25 Battle Station v2") as app:
    
    # --- Khai báo các State ---
    full_response_state = gr.State()
    kis_qna_df_state = gr.State()
    trake_steps_state = gr.State([])
    current_trake_sequence_state = gr.State(pd.DataFrame())
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
                    status_trake = gr.Markdown("Chưa có dữ liệu. Hãy thực hiện một truy vấn TRAKE.")
                    with gr.Row():
                        # **KHU VỰC LẮP RÁP MỚI**
                        with gr.Column(scale=3):
                             gr.Markdown("#### Chuỗi đang xây dựng")
                             current_sequence_table = gr.DataFrame(label="Click vào ứng viên bên phải để thêm vào đây", headers=['step', 'video_id', 'timestamp', 'final_score'])
                             validation_status = gr.Markdown("...")
                             with gr.Row():
                                 add_seq_to_submission_button = gr.Button("➕ Thêm chuỗi này", variant="primary")
                                 clear_seq_button = gr.Button("🗑️ Xóa chuỗi")
                        
                        # **CÁC CỘT ỨNG VIÊN ĐỘNG**
                        with gr.Column(scale=2):
                            gr.Markdown("#### Ứng viên (Click để thêm)")
                            # Tạo sẵn các component, ban đầu sẽ bị ẩn
                            trake_candidate_headers = []
                            trake_candidate_tables = []
                            MAX_STEPS = 6 # Giả sử truy vấn TRAKE có tối đa 6 bước
                            for i in range(MAX_STEPS):
                                header = gr.Markdown(f"<h4>Bước {i+1}</h4>", visible=False)
                                table = gr.DataFrame(
                                    headers=['keyframe_id', 'timestamp', 'final_score'],
                                    row_count=(5, "dynamic"),
                                    interactive=True,
                                    visible=False
                                )
                                trake_candidate_headers.append(header)
                                trake_candidate_tables.append(table)

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
            kis_qna_table, trake_steps_state, status_kis_qna,
            kis_qna_df_state
        ]
    ).then(
        fn=build_trake_workspace,
        inputs=[trake_steps_state],
        # **SỬA LỖI 1**: Unpack list component ra
        outputs=trake_candidate_headers + trake_candidate_tables
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
        inputs=[submission_list_state, kis_qna_table],
        # **THAY ĐỔI QUAN TRỌNG**: Output giờ là một list gồm 2 component
        outputs=[
            submission_list_table, # Cập nhật bảng danh sách nộp bài
            kis_qna_table          # Truyền `None` vào đây để xóa lựa chọn
        ]
    )
    for i, table in enumerate(trake_candidate_tables):
        table.select(
            fn=update_current_sequence,
            inputs=[current_trake_sequence_state, gr.State(i), trake_steps_state],
            outputs=[current_sequence_table, validation_status]
        )
    clear_seq_button.click(
        fn=clear_current_sequence,
        outputs=[current_sequence_table, validation_status]
    )
    add_seq_to_submission_button.click(
        fn=add_sequence_to_submission,
        inputs=[submission_list_state, current_sequence_table],
        outputs=[submission_list_table] # Sẽ được cập nhật ở GĐ4
    ).then(
        fn=clear_current_sequence, # Tự động xóa chuỗi sau khi thêm thành công
        outputs=[current_sequence_table, validation_status]
    )

if __name__ == "__main__":
    app.launch(debug=True, share=True)