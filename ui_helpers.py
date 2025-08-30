# /AIC25_Video_Search_Engine/ui_helpers.py

import os
import base64
from typing import Dict, Any, List
from search_core.task_analyzer import TaskType

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

def create_detailed_info_html(result: Dict[str, Any], task_type: TaskType) -> str:
    """Tạo mã HTML hiển thị thông tin chi tiết của một kết quả được chọn."""
    def create_progress_bar(score, color):
        percentage = max(0, min(100, score * 100))
        return f"""<div style='background: #e9ecef; border-radius: 5px; overflow: hidden;'><div style='background: {color}; width: {percentage}%; height: 10px; border-radius: 5px;'></div></div>"""

    video_id = result.get('video_id', 'N/A')
    keyframe_id = result.get('keyframe_id', 'N/A')
    timestamp = result.get('timestamp', 0)
    final_score = result.get('final_score', 0)
    scores = result.get('scores', {})

    # Bảng thông tin cơ bản
    info_html = f"""
    <div style='font-size: 14px; line-height: 1.6; background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>
        <p style='margin: 0;'><strong>📹 Video ID:</strong> <code>{video_id}</code></p>
        <p style='margin: 5px 0 0 0;'><strong>🖼️ Keyframe ID:</strong> <code>{keyframe_id}</code></p>
        <p style='margin: 5px 0 0 0;'><strong>⏰ Timestamp:</strong> <code>{timestamp:.2f}s</code></p>
    </div>
    """

    # Bảng điểm số chi tiết
    scores_html = f"""
    <div style='background-color: #f3f4f6; padding: 15px; border-radius: 8px; margin-top: 15px;'>
        <h4 style='margin: 0 0 15px 0; color: #111827;'>🏆 Bảng điểm</h4>
        <div style='margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                <span><strong>📊 Điểm tổng:</strong></span>
                <span style='font-weight: bold; font-size: 16px;'>{final_score:.4f}</span>
            </div>
            {create_progress_bar(final_score, '#10b981')}
        </div>
        """
    score_items = [
        ('CLIP', 'clip', '#3b82f6'), 
        ('Object', 'object', '#f97316'), 
        ('Semantic', 'semantic', '#8b5cf6')
    ]
    for name, key, color in score_items:
        if key in scores:
            scores_html += f"""
            <div style='margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                    <span>{name} Score:</span><span>{scores[key]:.3f}</span>
                </div>
                {create_progress_bar(scores[key], color)}
            </div>
            """
    scores_html += "</div>"
    
    return info_html + scores_html

def format_submission_list_for_display(submission_list: List[Dict[str, Any]]) -> str:
    """Biến danh sách submission thành một chuỗi text đẹp mắt để hiển thị."""
    if not submission_list:
        return "Chưa có kết quả nào được thêm vào."
    
    display_text = []
    for i, item in enumerate(submission_list):
        task_type = item.get('task_type')
        item_info = ""
        if task_type == TaskType.TRAKE:
            item_info = f"TRAKE Seq | Vid: {item.get('video_id')} | Score: {item.get('final_score', 0):.3f}"
        else:  # KIS, QNA
            item_info = f"Frame | {item.get('keyframe_id')} | Score: {item.get('final_score', 0):.3f}"
        
        display_text.append(f"{i+1:02d}. {item_info}")
    
    return "\n".join(display_text)

def build_selected_preview(gallery_items, selected_indices):
    """Tạo danh sách đường dẫn ảnh cho khu vực 'Ảnh đã chọn'."""
    imgs = []
    # Helper function to normalize item
    def normalize_item_to_path(item):
        return item[0] if isinstance(item, (list, tuple)) else item

    for i in sorted(selected_indices or []):
        if 0 <= i < len(gallery_items or []):
            path = normalize_item_to_path(gallery_items[i])
            if path:
                imgs.append(path)
    return imgs