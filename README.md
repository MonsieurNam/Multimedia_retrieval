/kaggle/working/Multimedia_retrieval
|-- main_app.py             # File chính để chạy Gradio App
|-- search_core/
|   |-- __init__.py
|   |-- basic_searcher.py     # Class BasicSearcher đã có
|   |-- semantic_searcher.py  # Class SemanticSearcher đã có
|   |-- task_analyzer.py      # Module 6.1: Phân loại truy vấn
|   |-- vqa_handler.py        # Module 6.2: Xử lý Q&A
|   |-- trake_solver.py       # Module 6.3: Xử lý TRAKE
|   |-- master_searcher.py    # Module 6.4: Bộ điều phối chính
|-- utils/
|   |-- __init__.py
|   |-- video_utils.py        # Các hàm liên quan đến ffmpeg
|   |-- formatting.py         # Các hàm định dạng output để nộp bài
