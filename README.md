# ASL Learning Suite v3.2 (Python + PyQt5)

Ứng dụng học **bảng chữ cái ASL (A–Z)** bằng **webcam + hand tracking (MediaPipe/cvzone)** và **mô hình TensorFlow/Keras (MobileNetV2)**.

---

## Repo GitHub
Source code của dự án được lưu tại:
- https://github.com/LuuAnhDuc-20204875/Sign_Language_Learning_Application.git

Clone nhanh:
```bash
git clone https://github.com/LuuAnhDuc-20204875/Sign_Language_Learning_Application.git
cd Sign_Language_Learning_Application
1) Tính năng
Learn: học tuần tự A–Z, yêu cầu streak dự đoán đúng để hoàn thành 1 chữ.

Quiz: random chữ để luyện, chấm điểm + lưu tiến độ theo chữ.

MCQ: trắc nghiệm 4 lựa chọn (A/B/C/D); chọn bằng gesture (pinch).

Spelling: đánh vần từ (word list lấy từ config.json -> spelling_words).

Snake: game rắn điều khiển bằng tay (ngón trỏ); có vùng “hand space” phía dưới để tránh mất tracking.

Teacher Mode: hiển thị FPS + confidence/overlay hỗ trợ dạy học.

User hệ local: Guest hoặc đăng ký/đăng nhập (lưu trong users.json) + high score theo từng mode.

Progress + Telemetry: tự lưu tiến độ và log thống kê phục vụ báo cáo/luận văn.

2) Yêu cầu hệ thống
Khuyến nghị Windows (vì requirements.txt đang “pin” phiên bản để tránh lỗi TensorFlow/MediaPipe trên Windows):

Windows 10/11, webcam hoạt động.

Python 3.10 (64-bit) (khuyến nghị mạnh).

Cài Microsoft Visual C++ Redistributable 2015–2022 (x64) (nếu không dễ lỗi TensorFlow DLL).

Nếu bạn chạy Linux/macOS: có thể vẫn chạy được, nhưng đôi khi cần tự điều chỉnh phiên bản TensorFlow/MediaPipe.

3) Cài đặt & chạy (Windows – khuyến nghị)
Mở PowerShell trong thư mục PRODUCT/ (hoặc thư mục chứa main.py):

py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python main.py
Nếu gặp lỗi TensorFlow/MediaPipe/Protobuf trên Windows, xem thêm: README_SETUP_WINDOWS.md.

4) Xuất báo cáo tiến độ nhanh
Dự án có sẵn script để xuất báo cáo tiến độ từ file progress_<user>.json (tổng số lần làm, số đúng, accuracy theo chữ/từ).

Chạy trong terminal (đang ở thư mục PRODUCT/):

python export_progress_report.py progress_guest.json
# hoặc
python export_progress_report.py progress_<username>.json
Gợi ý: file progress sẽ tự được tạo sau khi bạn chơi Learn/Quiz/Spelling/MCQ/Snake ít nhất 1 lần.

5) Huấn luyện mô hình (Notebooks .ipynb)
Phần code huấn luyện không nằm trong app PyQt5, mà nằm trong các notebook Jupyter để dễ chạy trên Kaggle/Colab.

5.1 Notebook benchmark (ASL Alphabet – so sánh nhiều mô hình)
File: asl-alphabet-by-duc.ipynb

Mục tiêu: benchmark nhanh trên dataset ASL Alphabet (29 lớp: A–Z + del/nothing/space) với các mô hình như SimpleCNN / MobileNetV2 / ResNet50.

Dataset path (Kaggle) trong notebook:

train_dir = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"

test_dir = "/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test"

Output model (save):

asl_model_<name>.h5 (ví dụ: asl_model_mobilenetv2.h5, ...)

5.2 Notebook train model cuối (Sign_Language_dataset_cropped_v3_fullhand – MobileNetV2 2 phase)
File: final-sign-language-by-duc.ipynb

Mục tiêu: train MobileNetV2 theo 2 phase (freeze backbone → fine-tune) trên dataset:

Kaggle dataset folder: Sign_Language_dataset_cropped_v3_fullhand

Trong notebook:

BASE_DIR = "/kaggle/input/sign-language-dataset/Sign_Language_dataset_cropped_v3_fullhand"

train_dir = BASE_DIR + "/alphabet_train"

test_dir = BASE_DIR + "/alphabet_test"

Output model (save):

asl_mobilenetv2_phase1_best.h5

asl_mobilenetv2_finetune_best.h5

asl_model_MobileNetV2_finetuned.h5

5.3 Dùng model đã train vào ứng dụng
Sau khi train xong, tải file .h5 về và đặt vào thư mục:

PRODUCT/models/

Sau đó sửa config.json của app:

"model_path": "models/asl_model_MobileNetV2_finetuned_compat.h5"
Lưu ý: notebook được thiết kế để chạy trên Kaggle/Colab (đường dẫn /kaggle/input/...). Nếu bạn chạy local, chỉ cần đổi BASE_DIR/train_dir/test_dir sang đường dẫn dataset trên máy.

6) File dữ liệu được tạo ra (tự động)
Tất cả lưu local, không gửi mạng:

users.json
Lưu user (salt/hash) + high_scores theo mode (snake/quiz/spelling/mcq). Guest cũng có bucket riêng.

progress_<user>.json (vd: progress_guest.json, progress_Anhduc1406.json)
Lưu thống kê đúng/sai theo letters và words.

logs/events_<user>.jsonl
Log dạng JSON Lines theo thời gian (mode_enter, gesture_attempt, mcq_answer, snake_game_over, …).

logs/stats_<user>.json
Thống kê tổng hợp phục vụ report (attempts/correct/completions/time_sum/conf_sum,…).

7) Troubleshooting nhanh
7.1 Lỗi TensorFlow/MediaPipe/Protobuf (Windows)
Triệu chứng thường gặp:

AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

ImportError: DLL load failed while importing _pywrap_tensorflow_internal

Failed to load the native TensorFlow runtime

Cách xử lý:

Dùng Python 3.10 x64 + cài VC++ Redistributable 2015–2022 (x64)

Cài đúng dependencies từ requirements.txt (đang pin version)

Xem thêm file: README_SETUP_WINDOWS.md

7.2 Camera đen / không lên hình
Thử đổi camera_index (0/1/2…) trong config.json.

Đóng app khác đang dùng webcam (Zoom/Teams/OBS…).

Kiểm tra quyền camera của Windows.