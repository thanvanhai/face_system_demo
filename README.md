# YOLOv8 + Face Recognition Streamlit App

Ứng dụng Streamlit kết hợp YOLOv8 và face_recognition để nhận dạng khuôn mặt trong video.

---

## Mô tả

- **Enroll**: Thêm người vào database bằng cách upload ảnh mặt, có thể thêm nhiều ảnh để cải thiện độ chính xác.
- **Recognize**: Tải video lên, app sẽ dùng YOLOv8 phát hiện người, sau đó nhận diện mặt dựa trên database đã enroll.

---

## Yêu cầu

- Python 3.8+
- Các thư viện Python:
  - streamlit
  - ultralytics (YOLOv8)
  - opencv-python
  - numpy
  - face_recognition
  - dlib
  - (có thể thêm các thư viện khác nếu dùng)

---

## Cài đặt

1. Tạo môi trường ảo (khuyến nghị):

```bash
python3 -m venv venv
source venv/bin/activate
