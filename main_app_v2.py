print("--- 🚀 Bắt đầu khởi chạy AIC25 Battle Station v2 ---")
print("--- Tải thư viện cho Giai đoạn 3...")

import gradio as gr
import pandas as pd
import numpy as np
import time
from enum import Enum

# ==============================================================================
# === PHẦN MOCK BACKEND - Giả lập các class để phát triển UI ===
# ==============================================================================

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
        'answer': [f'Câu trả lời mẫu {i}' for i in range(num_rows)],
        'video_path': '/kaggle/input/aic2025-batch-1-video/L01_V001.mp4'
    }
    df = pd.DataFrame(data)
    return df.sort_values(by='final_score', ascending=False).reset_index(drop=True)

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
            'video_path': ['/kaggle/input//kaggle/input/aic2025-batch-1-video/L01_V001.mp4'] * num_candidates_per_step
        }
        all_steps_dfs.append(pd.DataFrame(data))
        base_timestamp += 100 # Đảm bảo các bước sau có timestamp lớn hơn
    return all_steps_dfs

class MockMasterSearcher:
    """Class MasterSearcher giả lập."""
    def search(self, query: str, config: dict = None):
        print(f"--- MOCK BACKEND: Nhận truy vấn '{query}' ---")
        time.sleep(1) # Giả lập thời gian xử lý
        if "nhảy" in query or "(1)" in query or "bước" in query:
            print("--- MOCK BACKEND: Phân loại là TRAKE ---")
            return {
                'task_type': MockTaskType.TRAKE,
                'query_analysis': {'task_type': 'TRAKE', 'search_context': query, 'sub_queries': ["bước 1", "bước 2", "bước 3", "bước 4"]},
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

# Mock hàm tạo video clip
def create_mock_video_segment(video_path, timestamp):
    print(f"--- MOCK VIDEO: Giả lập cắt video '{video_path}' tại {timestamp}s ---")
    return '/kaggle/input//kaggle/input/aic2025-batch-1-video/L01_V001.mp4'


# ==============================================================================
# === CÁC HÀM LOGIC CHO GIAO DIỆN ===
# ==============================================================================
def reorder_submission_list(submission_list_df: pd.DataFrame, direction: str, evt: gr.SelectData):
    """
    Sắp xếp lại danh sách nộp bài bằng cách di chuyển hàng được chọn lên hoặc xuống.
    """
    if submission_list_df is None or submission_list_df.empty or evt.index is None:
        gr.Warning("Vui lòng chọn một hàng trong danh sách nộp bài để di chuyển.")
        return submission_list_df

    selected_index = evt.index[0]
    
    if direction == "up":
        if selected_index == 0: return submission_list_df # Đã ở trên cùng
        swap_with_index = selected_index - 1
    elif direction == "down":
        if selected_index == len(submission_list_df) - 1: return submission_list_df # Đã ở dưới cùng
        swap_with_index = selected_index + 1
    else:
        return submission_list_df

    # Hoán đổi vị trí hai hàng
    b, a = submission_list_df.iloc[selected_index].copy(), submission_list_df.iloc[swap_with_index].copy()
    submission_list_df.iloc[selected_index], submission_list_df.iloc[swap_with_index] = a, b
    
    gr.Info(f"Đã di chuyển hàng {selected_index+1} lên vị trí {swap_with_index+1}" if direction == "up" else f"Đã di chuyển hàng {selected_index+1} xuống vị trí {swap_with_index+1}")
    
    return submission_list_df

def remove_from_submission_list(submission_list_df: pd.DataFrame, evt: gr.SelectData):
    """
    Xóa hàng được chọn khỏi danh sách nộp bài.
    """
    if submission_list_df is None or submission_list_df.empty or evt.index is None:
        gr.Warning("Vui lòng chọn một hàng để xóa.")
        return submission_list_df

    selected_index = evt.index[0]
    
    # Xóa hàng tại vị trí đã chọn
    updated_df = submission_list_df.drop(submission_list_df.index[selected_index]).reset_index(drop=True)
    
    gr.Info(f"Đã xóa hàng {selected_index+1} khỏi danh sách.")
    
    return updated_df

def handle_submission(submission_list_df: pd.DataFrame, query_id: str):
    """
    Tạo và cung cấp file nộp bài từ DataFrame trong State.
    """
    if submission_list_df is None or submission_list_df.empty:
        gr.Warning("Danh sách nộp bài rỗng. Không có gì để tạo file.")
        return None
    
    if not query_id or not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
    
    # Định dạng lại DataFrame lần cuối trước khi lưu
    # Bỏ các cột không cần thiết cho file nộp bài
    submission_cols = [col for col in submission_list_df.columns if col.startswith('video_id') or col.startswith('frame_moment_') or col.startswith('answer')]
    final_df = submission_list_df[submission_cols]

    # Giả sử chúng ta có một hàm format_for_submission thật
    # Ở đây chúng ta chỉ cần lưu file
    output_path = f"/kaggle/working/{query_id.strip()}_submission.csv"
    final_df.to_csv(output_path, index=False, header=True) # Tạm thời để header để dễ debug
    
    gr.Success(f"Đã tạo file nộp bài thành công tại: {output_path}")
    
    return output_path

def handle_search_and_update_workspaces(query_text: str):
    """
    *** HÀM CHÍNH SỬA LỖI: Trả về giá trị gốc thay vì đối tượng gr.update ***
    """
    print(f"--- UI: Bắt đầu tìm kiếm và cập nhật workspace cho '{query_text}' ---")
    
    # 1. Gọi backend để lấy dữ liệu
    response = mock_master_searcher.search(query_text)
    task_type = response['task_type']
    query_analysis = response['query_analysis']
    
    # 2. Chuẩn bị các giá trị trả về chung
    analysis_summary = (f"<b>Loại nhiệm vụ:</b> {task_type.value}<br>"
                      f"<b>Bối cảnh tìm kiếm:</b> {query_analysis.get('search_context', 'N/A')}")
    
    # 3. Chuẩn bị các giá trị trả về mặc định
    kis_qna_df_output = pd.DataFrame()
    kis_qna_df_state_output = pd.DataFrame()
    trake_steps_state_output = []
    status_msg = "Sẵn sàng."
    
    # Chuẩn bị các giá trị cho workspace TRAKE (mặc định là ẩn)
    MAX_STEPS = 6 # Phải khớp với UI
    trake_workspace_updates = []
    # Mỗi bước có 1 header (string, visible) và 1 table (DataFrame, visible)
    for _ in range(MAX_STEPS):
        trake_workspace_updates.append("") # value cho Markdown
        trake_workspace_updates.append(False) # visible cho Markdown
        trake_workspace_updates.append(pd.DataFrame()) # value cho DataFrame
        trake_workspace_updates.append(False) # visible cho DataFrame

    # 4. Xử lý logic dựa trên loại nhiệm vụ
    if task_type == MockTaskType.TRAKE:
        trake_step_candidates = response['trake_step_candidates']
        trake_steps_state_output = trake_step_candidates
        status_msg = f"Đã tìm thấy ứng viên cho {len(trake_step_candidates)} bước TRAKE."
        
        # Tạo các giá trị cập nhật cho workspace TRAKE
        num_steps = len(trake_step_candidates)
        trake_workspace_updates = []
        for i in range(MAX_STEPS):
            if i < num_steps:
                # Giá trị cho Markdown Header
                trake_workspace_updates.append(f"<h4>Bước {i+1}</h4>")
                trake_workspace_updates.append(True)
                # Giá trị cho DataFrame Table
                trake_workspace_updates.append(trake_step_candidates[i])
                trake_workspace_updates.append(True)
            else:
                # Các giá trị để ẩn component
                trake_workspace_updates.append("")
                trake_workspace_updates.append(False)
                trake_workspace_updates.append(pd.DataFrame())
                trake_workspace_updates.append(False)
                
    else: # KIS hoặc QNA
        kis_qna_candidates = response['kis_qna_candidates']
        kis_qna_df_output = kis_qna_candidates
        kis_qna_df_state_output = kis_qna_candidates
        status_msg = f"Đã tìm thấy {len(kis_qna_candidates)} ứng viên KIS/QNA."

    # 5. Trả về một tuple lớn chứa TẤT CẢ các giá trị cập nhật
    # Chúng ta sẽ trả về một gr.update object lớn bao bọc tất cả các giá trị
    
    # Tạo một dictionary để map component tới giá trị mới
    # Đây là cách làm rõ ràng và an toàn nhất
    
    output_dict = {
        analysis_summary_output: analysis_summary,
        full_response_state: response,
        kis_qna_table: kis_qna_df_output,
        trake_steps_state: trake_steps_state_output,
        status_kis_qna: status_msg,
        kis_qna_df_state: kis_qna_df_state_output,
    }
    
    # Cập nhật các component TRAKE
    for i in range(MAX_STEPS):
        output_dict[trake_candidate_headers[i]] = gr.update(value=trake_workspace_updates[i*4], visible=trake_workspace_updates[i*4+1])
        output_dict[trake_candidate_tables[i]] = gr.update(value=trake_workspace_updates[i*4+2], visible=trake_workspace_updates[i*4+3])

    return output_dict

def on_kis_qna_select(kis_qna_df: pd.DataFrame, evt: gr.SelectData):
    if evt.index is None or kis_qna_df.empty:
        return None, "Vui lòng chọn một hàng để xem chi tiết."
    selected_row_index = evt.index[0]
    selected_row = kis_qna_df.iloc[selected_row_index]
    video_clip = create_mock_video_segment(selected_row['video_path'], selected_row['timestamp'])
    detailed_info_html = f"""<h4>Thông tin Chi tiết</h4><ul><li><b>Video ID:</b> {selected_row['video_id']}</li><li><b>Keyframe ID:</b> {selected_row['keyframe_id']}</li><li><b>Timestamp:</b> {selected_row['timestamp']:.2f}s</li><li><b>Final Score:</b> {selected_row['final_score']:.4f}</li></ul>"""
    return video_clip, detailed_info_html

def update_kis_qna_view(kis_qna_df: pd.DataFrame, sort_by: str, filter_video: str):
    if kis_qna_df is None or kis_qna_df.empty:
        return pd.DataFrame()
    df_processed = kis_qna_df.copy()
    if filter_video and filter_video.strip():
        df_processed = df_processed[df_processed['video_id'].str.contains(filter_video.strip(), case=False)]
    if sort_by and sort_by in df_processed.columns:
        is_ascending = not ('score' in sort_by)
        df_processed = df_processed.sort_values(by=sort_by, ascending=is_ascending)
    return df_processed

def add_to_submission_list_from_kis(submission_list: pd.DataFrame, kis_qna_df: pd.DataFrame, evt: gr.SelectData):
    if evt.index is None or kis_qna_df.empty:
        gr.Warning("Chưa có ứng viên nào được chọn!")
        return submission_list, None
    selected_row_index = evt.index[0]
    selected_row = kis_qna_df.iloc[[selected_row_index]]
    if submission_list is None: submission_list = pd.DataFrame()
    updated_list = pd.concat([submission_list, selected_row]).reset_index(drop=True)
    gr.Info(f"Đã thêm {selected_row['keyframe_id'].iloc[0]} vào danh sách nộp bài!")
    return updated_list, None

def update_current_sequence(current_sequence: pd.DataFrame, step_index: int, all_steps_data: list, evt: gr.SelectData):
    if evt.index is None or not all_steps_data or step_index >= len(all_steps_data):
        return current_sequence, "Lỗi: Dữ liệu không hợp lệ."
    selected_row_index = evt.index[0]
    df_step = all_steps_data[step_index]
    selected_row = df_step.iloc[[selected_row_index]]
    if current_sequence is None: current_sequence = pd.DataFrame()
    selected_row['step'] = step_index + 1
    updated_sequence = pd.concat([current_sequence, selected_row]).sort_values(by='step').reset_index(drop=True)
    is_valid, validation_msg = validate_sequence(updated_sequence)
    return updated_sequence, validation_msg

def validate_sequence(sequence_df: pd.DataFrame):
    if sequence_df.empty or len(sequence_df) <= 1:
        return True, "✅ Chuỗi hợp lệ (1 bước)."
    if sequence_df['video_id'].nunique() > 1:
        return False, "❌ Lỗi: Các bước phải cùng một video!"
    if not sequence_df['timestamp'].is_monotonic_increasing:
        return False, "❌ Lỗi: Timestamp phải tăng dần!"
    return True, f"✅ Chuỗi hợp lệ ({len(sequence_df)} bước)."

def clear_current_sequence():
    return pd.DataFrame(), "Đã xóa chuỗi hiện tại."

def add_sequence_to_submission(submission_list: pd.DataFrame, current_sequence: pd.DataFrame):
    is_valid, msg = validate_sequence(current_sequence)
    if not is_valid:
        gr.Warning(f"Không thể thêm chuỗi không hợp lệ! {msg}")
        return submission_list
    if current_sequence.empty:
        gr.Warning("Chuỗi đang xây dựng rỗng!")
        return submission_list
    scores = pd.to_numeric(current_sequence['final_score'], errors='coerce')
    mean_score = scores.mean()
    submission_row = { 'task_type': ['TRAKE'], 'final_score': [mean_score], 'video_id': [current_sequence['video_id'].iloc[0]] }
    for i, row in current_sequence.iterrows():
        submission_row[f'frame_moment_{i+1}'] = [row['keyframe_id']]
    submission_df_row = pd.DataFrame(submission_row)
    if submission_list is None: submission_list = pd.DataFrame()
    updated_list = pd.concat([submission_list, submission_df_row]).reset_index(drop=True)
    gr.Info(f"Đã thêm chuỗi video {submission_row['video_id'][0]} vào danh sách nộp bài!")
    return updated_list

# ==============================================================================
# === GIAO DIỆN GRADIO ===
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="AIC25 Battle Station v2") as app:
    
    # --- Khai báo States ---
    full_response_state = gr.State()
    kis_qna_df_state = gr.State()
    trake_steps_state = gr.State([])
    submission_list_state = gr.State(pd.DataFrame())
    current_trake_sequence_state = gr.State(pd.DataFrame())

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
                    with gr.Row():
                        sort_dropdown = gr.Dropdown(label="Sắp xếp theo", choices=['final_score', 'clip_score', 'object_score', 'semantic_score', 'timestamp'], value='final_score')
                        filter_textbox = gr.Textbox(label="Lọc theo Video ID")
                    kis_qna_table = gr.DataFrame(label="Top 200 Ứng viên (Click vào hàng để xem chi tiết)", headers=['video_id', 'timestamp', 'final_score', 'clip_score', 'object_score', 'semantic_score'], datatype=['str', 'number', 'number', 'number', 'number', 'number'], row_count=(10, "dynamic"), col_count=(6, "fixed"), interactive=True)

                with gr.TabItem("Bàn Lắp ráp Chuỗi TRAKE"):
                    status_trake = gr.Markdown("Chưa có dữ liệu. Hãy thực hiện một truy vấn TRAKE.")
                    with gr.Row():
                        with gr.Column(scale=3):
                             gr.Markdown("#### Chuỗi đang xây dựng")
                             current_sequence_table = gr.DataFrame(label="Click vào ứng viên bên phải để thêm vào đây", headers=['step', 'video_id', 'timestamp', 'final_score'])
                             validation_status = gr.Markdown("...")
                             with gr.Row():
                                 add_seq_to_submission_button = gr.Button("➕ Thêm chuỗi này", variant="primary")
                                 clear_seq_button = gr.Button("🗑️ Xóa chuỗi")
                        with gr.Column(scale=2):
                            gr.Markdown("#### Ứng viên (Click để thêm)")
                            trake_candidate_headers = []
                            trake_candidate_tables = []
                            MAX_STEPS = 6
                            for i in range(MAX_STEPS):
                                header = gr.Markdown(f"<h4>Bước {i+1}</h4>", visible=False)
                                table = gr.DataFrame(headers=['keyframe_id', 'timestamp', 'final_score'], row_count=(5, "dynamic"), interactive=True, visible=False)
                                trake_candidate_headers.append(header)
                                trake_candidate_tables.append(table)
        
        # --- KHU VỰC 2 & 3: XẾP HẠNG & CHI TIẾT (CỘT PHẢI) ---
        with gr.Column(scale=1):
            gr.Markdown("### 3. Bảng Xếp hạng & Xem chi tiết")
            with gr.Tabs():
                with gr.TabItem("Xem chi tiết"):
                    add_to_submission_button = gr.Button("➕ Thêm ứng viên này vào Danh sách Nộp bài")
                    video_player = gr.Video(label="Video Clip Preview")
                    detailed_info = gr.HTML("Thông tin chi tiết sẽ hiện ở đây khi bạn chọn một ứng viên.")
                
                with gr.TabItem("Danh sách Nộp bài (Top 100)"):
                    # **CÁC NÚT ĐIỀU KHIỂN MỚI**
                    with gr.Row():
                        up_button = gr.Button("▲ Di chuyển Lên")
                        down_button = gr.Button("▼ Di chuyển Xuống")
                        remove_button = gr.Button("❌ Xóa khỏi danh sách")
                    
                    submission_list_table = gr.DataFrame(
                        label="Click vào hàng để chọn. Dùng các nút trên để sắp xếp lại.",
                        interactive=True
                    )
            
            with gr.Group():
                 gr.Markdown("#### Nộp bài")
                 query_id_input = gr.Textbox(label="Query ID", placeholder="query_01")
                 submission_button = gr.Button("Tạo File Nộp bài", variant="primary")
                 submission_file_output = gr.File(label="Tải file nộp bài tại đây")

    # ==============================================================================
    # === KẾT NỐI CÁC SỰ KIỆN TƯƠNG TÁC ===
    # ==============================================================================
    
    # 1. Sự kiện Tìm kiếm
    all_search_outputs = [
        analysis_summary_output,
        full_response_state,
        kis_qna_table,
        trake_steps_state,
        status_kis_qna,
        kis_qna_df_state,
    ] + trake_candidate_headers + trake_candidate_tables
    
    search_button.click(
        fn=handle_search_and_update_workspaces,
        inputs=[query_input],
        outputs=all_search_outputs
    )
    
    # 2. Sự kiện KIS/Q&A
    kis_qna_table.select(
        fn=on_kis_qna_select,
        inputs=[kis_qna_table],
        outputs=[video_player, detailed_info]
    )
    sort_dropdown.change(
        fn=update_kis_qna_view,
        inputs=[kis_qna_df_state, sort_dropdown, filter_textbox],
        outputs=[kis_qna_table]
    )
    filter_textbox.submit(
        fn=update_kis_qna_view,
        inputs=[kis_qna_df_state, sort_dropdown, filter_textbox],
        outputs=[kis_qna_table]
    )
    add_to_submission_button.click(
        fn=add_to_submission_list_from_kis,
        inputs=[submission_list_state, kis_qna_table],
        outputs=[submission_list_table, kis_qna_table]
    )

    # 3. Sự kiện TRAKE
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
        outputs=[submission_list_table]
    ).then(
        fn=clear_current_sequence,
        outputs=[current_sequence_table, validation_status]
    )
    up_button.click(
        fn=reorder_submission_list,
        inputs=[submission_list_state, gr.State("up")], # Dùng gr.State để truyền tham số "up"
        outputs=[submission_list_table, submission_list_state] # Cập nhật cả bảng và state
    )
    
    down_button.click(
        fn=reorder_submission_list,
        inputs=[submission_list_state, gr.State("down")],
        outputs=[submission_list_table, submission_list_state]
    )
    
    remove_button.click(
        fn=remove_from_submission_list,
        inputs=[submission_list_state],
        outputs=[submission_list_table, submission_list_state]
    )
    
    # Cập nhật state khi bảng thay đổi do người dùng sắp xếp
    submission_list_table.change(
        fn=lambda x: x,
        inputs=[submission_list_table],
        outputs=[submission_list_state]
    )

    # **SỰ KIỆN MỚI: Nút Nộp bài**
    submission_button.click(
        fn=handle_submission,
        inputs=[submission_list_state, query_id_input],
        outputs=[submission_file_output]
    )

if __name__ == "__main__":
    print("\n--- ✅ Khởi tạo hoàn tất. Đang launch Gradio App Server... ---")
    app.launch(debug=True, share=True)