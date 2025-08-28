print("--- 🚀 Bắt đầu khởi chạy AIC25 Video Search Engine ---")
print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")

import sys
import os

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

print("--- Giai đoạn 2/4: Đang cấu hình và khởi tạo Backend...")

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
RERANK_METADATA_PATH = '/kaggle/input/stage1/rerank_metadata.parquet'
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
            gemini_api_key=GEMINI_API_KEY
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
        track_vqa_candidates: int  # <-- Thêm tham số
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
        "track_vqa_retrieval": int(track_vqa_retrieval), # <-- Thêm vào dict
        "track_vqa_candidates": int(track_vqa_candidates)  # <-- Thêm vào dict
    }
    
    start_time = time.time()
    
    full_response = master_searcher.search(query=query_text, config=config)
    
    search_time = time.time() - start_time
    
    formatted_gallery = format_results_for_gallery(full_response)
    
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
    cleaned_response_for_state = {
        "task_type": full_response.get("task_type"),
        "results": full_response.get("results")
    }
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    status_msg_html = f"✅ Tìm kiếm hoàn tất trong {search_time:.2f}s. Chế độ: {task_type_msg}"
    
    return formatted_gallery, status_msg_html, cleaned_response_for_state, gemini_analysis_html, stats_info_html

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

def on_gallery_select(response_state: Dict[str, Any], evt: gr.SelectData):
    """
    Hàm xử lý sự kiện khi người dùng chọn một ảnh trong gallery.
    *** PHIÊN BẢN ĐƠN GIẢN VÀ TỐI ƯU ***
    """
    if not response_state or evt is None:
        return None, "", ""

    results = response_state.get("results", [])
    if not results or evt.index >= len(results):
        gr.Error("Lỗi: Không tìm thấy kết quả tương ứng. Vui lòng thử tìm kiếm lại.")
        return None, "Lỗi: Dữ liệu không đồng bộ.", ""

    selected_result = results[evt.index]
    task_type = response_state.get('task_type')

    # --- Nhánh 1: Xử lý kết quả tổng hợp TRACK_VQA ---
    if selected_result.get("is_aggregated_result"):
        final_answer = selected_result.get("final_answer", "N/A")
        evidence_paths = selected_result.get("evidence_paths", [])
        evidence_captions = selected_result.get("evidence_captions", [])
        
        # Tạo grid đơn giản cho evidence
        evidence_html = ""
        if evidence_paths:
            evidence_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 8px; margin-top: 12px;">'
            for i, (path, caption) in enumerate(zip(evidence_paths, evidence_captions)):
                image_url = f"/file={path}"
                evidence_html += f'''
                <div style="border: 1px solid #ddd; border-radius: 6px; overflow: hidden; background: white;">
                    <img src="{image_url}" style="width: 100%; height: 80px; object-fit: cover;" alt="Evidence {i+1}">
                    <div style="padding: 6px; font-size: 10px; color: #555; text-align: center;">
                        #{i+1}: {caption[:30]}{"..." if len(caption) > 30 else ""}
                    </div>
                </div>
                '''
            evidence_html += '</div>'
        else:
            evidence_html = '<div style="text-align: center; padding: 20px; color: #999; font-style: italic;">Không có hình ảnh bằng chứng</div>'
            
        detailed_info_html = f'''
        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px;">
            <h4 style="margin: 0 0 12px 0; color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 8px;">
                💡 Kết quả Tổng hợp
            </h4>
            
            <div style="background: white; border: 1px solid #e9ecef; border-radius: 4px; padding: 12px; margin-bottom: 16px;">
                <strong style="color: #28a745;">🤖 Trả lời AI:</strong>
                <p style="margin: 8px 0 0 0; line-height: 1.4; color: #495057;">{final_answer}</p>
            </div>
            
            <div>
                <strong style="color: #17a2b8;">🖼️ Bằng chứng ({len(evidence_paths)} ảnh):</strong>
                {evidence_html}
            </div>
        </div>
        '''
        
        clip_info_html = '''
        <div style="background: #e3f2fd; border: 1px solid #90caf9; border-radius: 6px; padding: 12px; text-align: center; margin-top: 8px;">
            <div style="font-weight: 600; color: #1976d2;">📊 Thông tin tổng hợp</div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">Kết quả từ phân tích đa nguồn</div>
        </div>
        '''
        
        return None, detailed_info_html, clip_info_html

    # --- Nhánh 2: Xử lý kết quả chuỗi TRAKE ---
    elif task_type == TaskType.TRAKE:
        sequence = selected_result.get('sequence', [])
        if not sequence:
             return None, "Lỗi: Chuỗi TRAKE rỗng.", ""
        
        # Lấy frame đầu tiên để tạo clip
        target_frame = sequence[0]
        video_path = target_frame.get('video_path')
        timestamp = target_frame.get('timestamp')
        
        # Tạo HTML đơn giản cho sequence
        sequence_html = ""
        for i, frame in enumerate(sequence[:5]):  # Chỉ hiển thị 5 frame đầu
            video_id = frame.get('video_id', 'N/A')
            frame_timestamp = frame.get('timestamp', 0)
            score = frame.get('final_score', 0)
            objects = frame.get('objects_detected', [])
            
            sequence_html += f'''
            <div style="border: 1px solid {'#007bff' if i == 0 else '#dee2e6'}; border-radius: 4px; padding: 10px; margin-bottom: 8px; background: {'#f8f9ff' if i == 0 else 'white'};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #495057;">#{i+1} {video_id}</strong>
                        <div style="font-size: 12px; color: #666;">⏰ {frame_timestamp:.1f}s</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 600; color: #28a745;">{score:.3f}</div>
                        <div style="background: #e9ecef; height: 3px; width: 40px; border-radius: 2px; margin-top: 2px;">
                            <div style="background: #28a745; height: 100%; width: {min(100, score*100):.0f}%; border-radius: 2px;"></div>
                        </div>
                    </div>
                </div>
                {f'<div style="margin-top: 6px; font-size: 11px; color: #666;">🏷️ {", ".join(objects[:3])}</div>' if objects else ''}
            </div>
            '''
            
        detailed_info_html = f'''
        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px;">
            <h4 style="margin: 0 0 12px 0; color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 8px;">
                🔄 Chuỗi TRAKE ({len(sequence)} frames)
            </h4>
            <div style="max-height: 300px; overflow-y: auto;">
                {sequence_html}
            </div>
            {f'<div style="text-align: center; margin-top: 8px; font-size: 12px; color: #666;">... và {len(sequence)-5} frame khác</div>' if len(sequence) > 5 else ''}
        </div>
        '''

    # --- Nhánh 3: Xử lý kết quả đơn lẻ KIS và QNA ---
    else:
        target_frame = selected_result
        video_path = target_frame.get('video_path')
        timestamp = target_frame.get('timestamp')
        
        # Thông tin cơ bản
        video_id = target_frame.get('video_id', 'N/A')
        final_score = target_frame.get('final_score', 0)
        scores = target_frame.get('scores', {})
        objects = target_frame.get('objects_detected', [])
        
        # Tạo HTML đơn giản
        objects_html = ", ".join(objects[:8]) if objects else "Không có"
        
        detailed_info_html = f'''
        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px;">
            <h4 style="margin: 0 0 12px 0; color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 8px;">
                🎬 Chi tiết Keyframe
            </h4>
            
            <div style="margin-bottom: 12px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div><strong>📹 Video:</strong><br><code style="font-size: 12px;">{video_id}</code></div>
                    <div><strong>⏰ Thời điểm:</strong><br><code>{timestamp:.2f}s</code></div>
                </div>
            </div>
        '''
        
        # Thêm thông tin VQA nếu có
        if task_type == TaskType.QNA:
            answer = target_frame.get('answer', 'N/A')
            vqa_conf = scores.get('vqa_confidence', 0)
            detailed_info_html += f'''
            <div style="background: white; border: 1px solid #e9ecef; border-radius: 4px; padding: 12px; margin-bottom: 12px;">
                <strong style="color: #6f42c1;">💬 Câu trả lời:</strong>
                <div style="margin: 6px 0; font-weight: 600;">{answer}</div>
                <div style="font-size: 12px; color: #666;">Độ tin cậy: {vqa_conf:.2f}</div>
            </div>
            '''
        
        # Điểm số
        detailed_info_html += f'''
            <div style="background: white; border: 1px solid #e9ecef; border-radius: 4px; padding: 12px; margin-bottom: 12px;">
                <strong style="color: #28a745;">🏆 Điểm số:</strong>
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Tổng:</span><strong>{final_score:.4f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                        <span>CLIP:</span><span>{scores.get('clip', 0):.3f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                        <span>Object:</span><span>{scores.get('object', 0):.3f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                        <span>Semantic:</span><span>{scores.get('semantic', 0):.3f}</span>
                    </div>
                </div>
            </div>
            
            <div>
                <strong style="color: #fd7e14;">🔍 Đối tượng:</strong>
                <div style="margin-top: 6px; font-size: 12px; color: #495057;">{objects_html}</div>
            </div>
        </div>
        '''

    # Logic chung cho Nhánh 2 và 3 (tạo video clip)
    if not selected_result.get("is_aggregated_result"):
        video_clip_path = create_video_segment(video_path, timestamp)
        
        clip_info_html = f'''
        <div style="background: #e8f5e8; border: 1px solid #c3e6c3; border-radius: 6px; padding: 12px; text-align: center; margin-top: 8px;">
            <div style="font-weight: 600; color: #155724;">🎥 Video Clip</div>
            <div style="font-size: 12px; color: #666; margin-top: 4px;">
                {max(0, timestamp - 5):.1f}s - {timestamp + 5:.1f}s
            </div>
        </div>
        '''
        
        return video_clip_path, detailed_info_html, clip_info_html
    
    return None, detailed_info_html, clip_info_html

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
    
    response_state = gr.State()
    
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
                minimum=20,
                maximum=100,
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
    with gr.Accordion("⚙️ Tùy chỉnh Nâng cao", open=False):
        gr.Markdown("Điều chỉnh các tham số của thuật toán tìm kiếm và tái xếp hạng.")
        with gr.Tabs():
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
            with gr.TabItem("Track-VQA"):
                track_vqa_retrieval_slider = gr.Slider(
                    minimum=100, maximum=500, value=300, step=25,
                    label="Số ứng viên Track-VQA ban đầu (Retrieval)",
                    info="Lấy bao nhiêu ứng viên từ FAISS để tìm tất cả các bối cảnh."
                )
                track_vqa_candidates_slider = gr.Slider(
                    minimum=1, maximum=100, value=20, step=5,
                    label="Số ứng viên Track-VQA được phân tích",
                    info="Số lượng ứng viên tốt nhất sẽ được đưa vào pipeline VQA lặp lại."
                )
    status_output = gr.HTML()
    with gr.Row():
        gemini_analysis = gr.HTML()
        stats_info = gr.HTML()
    
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

    with gr.Accordion("💾 Tạo File Nộp Bài", open=False):
        with gr.Row():
            query_id_input = gr.Textbox(label="Nhập Query ID", placeholder="Ví dụ: query_01")
            submission_button = gr.Button("Tạo File")
        submission_file_output = gr.File(label="Tải file nộp bài của bạn")

    gr.HTML(usage_guide_html)
    gr.HTML(app_footer_html)

    search_inputs = [
        query_input, 
        num_results,
        kis_retrieval_slider,
        vqa_candidates_slider,
        vqa_retrieval_slider,
        trake_candidates_per_step_slider,
        trake_max_sequences_slider,
        track_vqa_retrieval_slider, # <-- Thêm slider mới
        track_vqa_candidates_slider   # <-- Thêm slider mới
    ]
    search_outputs = [results_gallery, status_output, response_state, gemini_analysis, stats_info]
    
    search_button.click(fn=perform_search, inputs=search_inputs, outputs=search_outputs)
    query_input.submit(fn=perform_search, inputs=search_inputs, outputs=search_outputs)
    
    results_gallery.select(
        fn=on_gallery_select,
        inputs=[response_state], # Chỉ cần response_state
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

if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server ---")
    app.launch(
        share=True,
        allowed_paths=["/kaggle/input/", "/kaggle/working/"],
        debug=True, # Bật debug để xem lỗi chi tiết trên console
        show_error=True # Hiển thị lỗi trên giao diện
    )