import time
import random
from itertools import product
from typing import List, Dict, Any

# ==============================================================================
# ĐỊNH NGHĨA 2 PHIÊN BẢN THUẬT TOÁN ĐỂ SO SÁNH
# ==============================================================================

def find_sequences_bruteforce(video_step_candidates: List[List[Dict]]) -> List[Dict]:
    """Phiên bản duyệt toàn bộ dùng itertools.product."""
    valid_sequences = []
    
    # Tạo tất cả các tổ hợp chuỗi khả thi
    # Dùng list(product(*...)) để hiện thực hóa generator, dễ debug hơn
    all_combinations = list(product(*video_step_candidates))
    
    for sequence_tuple in all_combinations:
        is_valid_order = all(
            sequence_tuple[i]['timestamp'] < sequence_tuple[i+1]['timestamp'] 
            for i in range(len(sequence_tuple) - 1)
        )
        
        if is_valid_order:
            avg_score = sum(item['final_score'] for item in sequence_tuple) / len(sequence_tuple)
            valid_sequences.append({
                "video_id": sequence_tuple[0]['video_id'],
                "sequence": list(sequence_tuple),
                "final_score": avg_score
            })
            
    return sorted(valid_sequences, key=lambda x: x['final_score'], reverse=True)

def find_sequences_beam_search(video_step_candidates: List[List[Dict]], beam_width: int) -> List[Dict]:
    """Phiên bản mới dùng Beam Search."""
    if not all(video_step_candidates):
        return []

    num_steps = len(video_step_candidates)
    
    beam = [([cand], cand['final_score']) for cand in video_step_candidates[0]]
    
    for step_idx in range(1, num_steps):
        next_beam = []
        for current_sequence, current_score in beam:
            last_frame_timestamp = current_sequence[-1]['timestamp']
            
            for next_candidate in video_step_candidates[step_idx]:
                if next_candidate['timestamp'] > last_frame_timestamp:
                    new_sequence = current_sequence + [next_candidate]
                    new_score = current_score + next_candidate['final_score']
                    next_beam.append((new_sequence, new_score))
        
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]

    final_sequences = []
    for final_sequence, total_score in beam:
        avg_score = total_score / len(final_sequence)
        final_sequences.append({
            "video_id": final_sequence[0]['video_id'],
            "sequence": final_sequence,
            "final_score": avg_score
        })
        
    return sorted(final_sequences, key=lambda x: x['final_score'], reverse=True)


# ==============================================================================
# HÀM TẠO DỮ LIỆU GIẢ LẬP VÀ THỰC THI KIỂM THỬ
# ==============================================================================
def create_mock_data(num_steps=4, candidates_per_step=20):
    print(f"\n--- Tạo dữ liệu giả: {num_steps} bước, {candidates_per_step} ứng viên/bước ---")
    
    all_step_candidates = []
    golden_sequence = []
    
    for i in range(num_steps):
        step_candidates = []
        time_offset = i * 20 # Đảm bảo timestamp các bước không chồng chéo
        
        for j in range(candidates_per_step):
            step_candidates.append({
                'video_id': 'test_video_01',
                'timestamp': time_offset + random.uniform(1, 15),
                'final_score': random.uniform(0.6, 0.85)
            })
        
        # Cấy một ứng viên "vàng" có điểm số cao vào mỗi bước
        golden_candidate = {
            'video_id': 'test_video_01',
            'timestamp': time_offset + 10,
            'final_score': 0.95 + random.uniform(-0.02, 0.02) # Điểm rất cao
        }
        step_candidates[random.randint(0, candidates_per_step-1)] = golden_candidate
        golden_sequence.append(golden_candidate)
        
        all_step_candidates.append(step_candidates)
        
    print(" -> Dữ liệu đã được tạo với một 'chuỗi vàng' được cấy vào.")
    return all_step_candidates, golden_sequence

def run_test():
    mock_data, golden_sequence = create_mock_data(num_steps=4, candidates_per_step=20)
    
    print("\n" + "="*50)
    print("--- 1. Kiểm thử Thuật toán Brute Force (Ground Truth) ---")
    start_time = time.time()
    bruteforce_results = find_sequences_bruteforce(mock_data)
    end_time = time.time()
    
    bruteforce_time = end_time - start_time
    best_bruteforce_score = bruteforce_results[0]['final_score'] if bruteforce_results else -1
    
    print(f" -> Thời gian thực thi: {bruteforce_time:.6f} giây")
    print(f" -> Tìm thấy {len(bruteforce_results)} chuỗi hợp lệ.")
    print(f" -> Điểm số chuỗi tốt nhất: {best_bruteforce_score:.4f}")
    
    # In ra chuỗi vàng để so sánh
    golden_score = sum(item['final_score'] for item in golden_sequence) / len(golden_sequence)
    print(f" -> (Điểm số 'chuỗi vàng' được cấy vào: {golden_score:.4f})")
    
    print("="*50)
    
    beam_widths_to_test = [2, 5, 10]
    for bw in beam_widths_to_test:
        print(f"\n--- 2. Kiểm thử Thuật toán Beam Search (beam_width = {bw}) ---")
        start_time = time.time()
        beam_search_results = find_sequences_beam_search(mock_data, beam_width=bw)
        end_time = time.time()
        
        beam_time = end_time - start_time
        best_beam_score = beam_search_results[0]['final_score'] if beam_search_results else -1
        
        print(f" -> Thời gian thực thi: {beam_time:.6f} giây")
        print(f" -> Tìm thấy {len(beam_search_results)} chuỗi hợp lệ.")
        print(f" -> Điểm số chuỗi tốt nhất: {best_beam_score:.4f}")
        
        # So sánh với Brute Force
        print(f" -> Tăng tốc so với Brute Force: {bruteforce_time / beam_time:.2f} lần")
        print(f" -> Độ chính xác (so với điểm tốt nhất): {best_beam_score / best_bruteforce_score * 100:.2f}%")
        print("-"*50)

# Chạy kiểm thử
if __name__ == "__main__":
    run_test()