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

# ==============================================================================
# === BẮT ĐẦU PHẦN CODE GIAO DIỆN GRADIO ===
# ==============================================================================
print("--- Giai đoạn 2/4: Đang định nghĩa các hàm logic cho giao diện...")

def perform_search(query_text: str):
    """
    Hàm xử lý sự kiện chính: gọi backend và đổ dữ liệu vào các State.
    """
    print(f"--- UI: Bắt đầu tìm kiếm cho '{query_text}' ---")
    
    # 1. Gọi backend (phiên bản mock)
    response = mock_master_searcher.search(query_text)
    
    # 2. Lấy dữ liệu từ response
    task_type = response['task_type']
    query_analysis = response['query_analysis']
    kis_qna_candidates = response['kis_qna_candidates']
    trake_step_candidates = response['trake_step_candidates']
    
    # 3. Chuẩn bị đầu ra để cập nhật UI
    # Tạo chuỗi tóm tắt phân tích
    analysis_summary = (
        f"<b>Loại nhiệm vụ:</b> {task_type.value}<br>"
        f"<b>Bối cảnh tìm kiếm:</b> {query_analysis.get('search_context', 'N/A')}<br>"
        f"<b>Thực thể:</b> {query_analysis.get('objects_en', [])}"
    )
    
    # Cập nhật các component tương ứng với loại nhiệm vụ
    if task_type == MockTaskType.TRAKE:
        # Nếu là TRAKE, cập nhật không gian làm việc TRAKE và xóa KIS/Q&A
        return (
            analysis_summary,
            response,
            pd.DataFrame(), # Xóa bảng KIS/Q&A
            trake_step_candidates,
            f"Đã tìm thấy ứng viên cho {len(trake_step_candidates)} bước TRAKE"
        )
    else: # KIS hoặc QNA
        # Cập nhật bảng KIS/Q&A và xóa TRAKE
        return (
            analysis_summary,
            response,
            kis_qna_candidates,
            [], # Xóa dữ liệu các bước TRAKE
            f"Đã tìm thấy {len(kis_qna_candidates)} ứng viên KIS/QNA"
        )


print("--- Giai đoạn 3/4: Đang xây dựng bố cục giao diện 'Trạm Tác chiến'...")

print("--- Giai đoạn 3/4: Đang xây dựng bố cục giao diện 'Trạm Tác chiến'...")

with gr.Blocks(theme=gr.themes.Soft(), title="AIC25 Battle Station v2") as app:
    
    # --- Khai báo các State để lưu trữ dữ liệu ---
    # State chứa toàn bộ response thô từ backend
    full_response_state = gr.State()
    # State cho bảng dữ liệu KIS/Q&A (dạng DataFrame)
    kis_qna_df_state = gr.State()
    # State cho dữ liệu các bước TRAKE (dạng list của list)
    trake_steps_state = gr.State()

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
                    
                    # =======================================================
                    # === SỬA LỖI TẠI ĐÂY ===
                    # Xóa 'max_rows' và thay bằng 'row_count'
                    # =======================================================
                    kis_qna_table = gr.DataFrame(
                        label="Top 200 Ứng viên (Sắp xếp, Lọc, Chọn ở Giai đoạn 2)",
                        headers=['video_id', 'timestamp', 'final_score', 'clip_score', 'object_score', 'semantic_score'],
                        datatype=['str', 'number', 'number', 'number', 'number', 'number'],
                        row_count=(10, "dynamic"), # Hiển thị 10 dòng, cho phép cuộn/phân trang
                        col_count=(6, "fixed"),    # Số cột là cố định
                        interactive=True # Sẽ dùng ở GĐ2
                    )
                    # =======================================================
                    # === KẾT THÚC SỬA LỖI ===
                    # =======================================================

                with gr.TabItem("Bàn Lắp ráp Chuỗi TRAKE"):
                    status_trake = gr.Markdown("Chưa có dữ liệu.")
                    # Ở GĐ1, chúng ta chỉ cần một placeholder. GĐ3 sẽ xây dựng chi tiết.
                    trake_workspace_placeholder = gr.HTML("Khu vực này sẽ hiển thị các cột ứng viên cho từng bước TRAKE.")


        # --- KHU VỰC 2 & 3: XẾP HẠNG & CHI TIẾT (CỘT PHẢI) ---
        with gr.Column(scale=1):
            gr.Markdown("### 3. Bảng Xếp hạng & Xem chi tiết")
            with gr.Tabs():
                with gr.TabItem("Xem chi tiết"):
                    video_player_placeholder = gr.Video(label="Video Clip Preview")
                    detailed_info_placeholder = gr.HTML("Thông tin chi tiết sẽ hiện ở đây khi bạn chọn một ứng viên.")
                
                with gr.TabItem("Danh sách Nộp bài (Top 100)"):
                    # Cũng áp dụng sửa lỗi tương tự ở đây
                    submission_list_placeholder = gr.DataFrame(
                        label="Danh sách này sẽ được sắp xếp lại bằng tay ở GĐ4",
                        row_count=(10, "dynamic"),
                        interactive=True # Để có thể chọn hàng và sắp xếp lại
                    )
            
            with gr.Group():
                 gr.Markdown("#### Nộp bài")
                 query_id_input = gr.Textbox(label="Query ID", placeholder="query_01")
                 submission_button = gr.Button("Tạo File Nộp bài")


    # ==============================================================================
    # === ĐỊNH NGHĨA CÁC SỰ KIỆN TƯƠNG TÁC ===
    # ==============================================================================
    print("--- Giai đoạn 4/4: Đang kết nối các sự kiện tương tác...")

    search_button.click(
        fn=perform_search,
        inputs=[query_input],
        outputs=[
            analysis_summary_output,
            full_response_state,
            kis_qna_table, # Cập nhật trực tiếp bảng KIS/Q&A
            trake_steps_state, # Cập nhật state TRAKE
            status_kis_qna # Dùng chung status cho cả 2 để đơn giản
        ]
    )

if __name__ == "__main__":
    print("\n--- ✅ Khởi tạo hoàn tất. Đang launch Gradio App Server... ---")
    app.launch(debug=True, share=True)