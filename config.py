# /AIC25_Video_Search_Engine/config.py

import os
from kaggle_secrets import UserSecretsClient

# --- Hằng số Giao diện & Tìm kiếm ---
ITEMS_PER_PAGE = 20
MAX_SUBMISSION_RESULTS = 100

# --- Đường dẫn tới các file dữ liệu ---
KAGGLE_INPUT_DIR = '/kaggle/input'
KAGGLE_WORKING_DIR = '/kaggle/working'

CLIP_FEATURES_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/features.npy')
FAISS_INDEX_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/faiss.index')
RERANK_METADATA_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/rerank_metadata_ultimate_v5.parquet')
ALL_ENTITIES_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/all_detection_entities.json')
VIDEO_BASE_PATH = os.path.join(KAGGLE_INPUT_DIR, 'aic2025-batch-1-video/')

def load_api_keys():
    """
    Tải API keys từ Kaggle Secrets một cách an toàn.
    Trả về một tuple chứa (openai_api_key, gemini_api_key).
    """
    openai_key, gemini_key = None, None
    try:
        user_secrets = UserSecretsClient()
        openai_key = user_secrets.get_secret("OPENAI_API_KEY")
        print("--- ✅ Cấu hình OpenAI API Key thành công! ---")
    except Exception:
        print("--- ⚠️ Không tìm thấy OpenAI API Key. ---")
        
    try:
        user_secrets = UserSecretsClient()
        gemini_key = user_secrets.get_secret("GOOGLE_API_KEY")
        print("--- ✅ Cấu hình GEMINI API Key thành công! ---")
    except Exception:
        print("--- ⚠️ Không tìm thấy GEMINI API Key. ---")
        
    return openai_key, gemini_key

# Tải keys khi module được import
OPENAI_API_KEY, GEMINI_API_KEY = load_api_keys()