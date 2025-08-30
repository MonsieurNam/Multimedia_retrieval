# /AIC25_Video_Search_Engine/backend_loader.py

import os
import glob
from search_core.basic_searcher import BasicSearcher
from search_core.master_searcher import MasterSearcher
from config import (
    VIDEO_BASE_PATH, FAISS_INDEX_PATH, RERANK_METADATA_PATH, 
    CLIP_FEATURES_PATH, ALL_ENTITIES_PATH, OPENAI_API_KEY, GEMINI_API_KEY
)

def initialize_backend():
    """
    Khởi tạo và trả về instance của MasterSearcher đã sẵn sàng hoạt động.
    Hàm này chỉ nên được gọi một lần khi ứng dụng khởi động.
    """
    print("--- 🚀 Giai đoạn 2/4: Đang cấu hình và khởi tạo Backend... ---")
    print("--- Đang khởi tạo các model (quá trình này chỉ chạy một lần)... ---")
    
    # 1. Tạo bản đồ đường dẫn video
    all_video_files = glob.glob(os.path.join(VIDEO_BASE_PATH, "**", "*.mp4"), recursive=True)
    video_path_map = {os.path.basename(f).replace('.mp4', ''): f for f in all_video_files}
    print(f"   -> Tìm thấy {len(video_path_map)} video.")

    # 2. Khởi tạo BasicSearcher
    print("   -> 1/2: Khởi tạo BasicSearcher...")
    basic_searcher = BasicSearcher(
        FAISS_INDEX_PATH, 
        RERANK_METADATA_PATH, 
        video_path_map, 
        clip_features_path=CLIP_FEATURES_PATH
    )
    
    # 3. Khởi tạo MasterSearcher
    print("   -> 2/2: Khởi tạo MasterSearcher (OpenAI & Gemini Edition)...")
    master_searcher = MasterSearcher(
        basic_searcher=basic_searcher,
        openai_api_key=OPENAI_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        entities_path=ALL_ENTITIES_PATH,
        clip_features_path=CLIP_FEATURES_PATH
    )    
    
    if not master_searcher.mmr_builder:
         print("   -> ⚠️ Cảnh báo: MMR Builder chưa được kích hoạt. Kết quả sẽ không có tính đa dạng.")

    print("--- ✅ Backend đã khởi tạo thành công! ---")
    return master_searcher