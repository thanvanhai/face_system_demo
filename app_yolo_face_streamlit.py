# app_yolo_face_streamlit.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import face_recognition
import os
import time
import json

st.set_page_config(page_title="YOLOv8 + Face Recognition", layout="wide")
st.title("YOLOv8 + face_recognition — Enroll & Recognize")

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model path", value="yolov8n.pt")
conf_th = st.sidebar.slider("YOLO confidence threshold", 0.1, 1.0, 0.45, 0.05)
face_tol = st.sidebar.slider("Face match tolerance (lower = stricter)", 0.25, 0.8, 0.5, 0.01)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)
expand_box_ratio = st.sidebar.slider("Expand person bbox (%) for face detection", 0, 50, 10, 5)

# -------------------------
# Load model (lazy)
# -------------------------
@st.cache_resource
def load_yolo(path, to_cuda=False):
    model = YOLO(path)
    if to_cuda:
        try:
            model.to("cuda")
        except Exception:
            pass
    return model

# -------------------------
# Enrollment area
# -------------------------
st.header("1) Enroll: Thêm người (upload ảnh, có thể nhiều ảnh cho 1 người)")
st.markdown("Upload ảnh từng người — nhập tên. Bạn có thể upload nhiều ảnh cho cùng 1 tên để improve matching.")

if "people_db" not in st.session_state:
    st.session_state["people_db"] = []  # mỗi item: {"name":..., "encodings":[...], "sample_paths":[...]}

with st.form("enroll_form", clear_on_submit=True):
    uploaded = st.file_uploader("Chọn ảnh (jpg/png). Nhấn Add để lưu vào DB tạm", type=["jpg","jpeg","png"])
    name = st.text_input("Tên người")
    add_btn = st.form_submit_button("Add to DB")
    if add_btn:
        if not uploaded or not name.strip():
            st.error("Cần both ảnh và tên.")
        else:
            # save temp
            tmpdir = os.path.join(tempfile.gettempdir(), "yolo_face_enroll")
            os.makedirs(tmpdir, exist_ok=True)
            path = os.path.join(tmpdir, f"{int(time.time()*1000)}_{uploaded.name}")
            with open(path, "wb") as f:
                f.write(uploaded.read())
            # load & encode
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if len(encs) == 0:
                st.error("Không tìm thấy mặt trong ảnh. Vui lòng dùng ảnh rõ mặt.")
                os.remove(path)
            else:
                # find entry for name or create
                entry = next((e for e in st.session_state["people_db"] if e["name"]==name.strip()), None)
                if entry is None:
                    entry = {"name": name.strip(), "encodings": [], "sample_paths": []}
                    st.session_state["people_db"].append(entry)
                entry["encodings"].append(encs[0].tolist())
                entry["sample_paths"].append(path)
                st.success(f"Đã lưu ảnh cho {name.strip()} (encodings: {len(entry['encodings'])})")

# show DB
st.subheader("Database (tạm):")
for i, person in enumerate(st.session_state["people_db"]):
    cols = st.columns([1,4,1])
    cols[0].image(person["sample_paths"][0], width=80)
    cols[1].markdown(f"**{person['name']}**  \nEncodings: {len(person['encodings'])}")
    if cols[2].button("Xóa", key=f"del_{i}"):
        st.session_state["people_db"].pop(i)
        st.experimental_rerun()

# -------------------------
# Recognition area
# -------------------------
st.header("2) Recognize: Upload video để kiểm tra")
video_file = st.file_uploader("Upload video (mp4/avi/mov)", type=["mp4","avi","mov"], key="video_upload")

if video_file and len(st.session_state["people_db"]) == 0:
    st.info("Vui lòng enroll tối thiểu 1 người trước khi chạy nhận dạng.")

if st.button("▶ Run recognition") and video_file and len(st.session_state["people_db"])>0:
    # Save video
    tpath = os.path.join(tempfile.gettempdir(), f"input_{int(time.time()*1000)}.mp4")
    with open(tpath, "wb") as f:
        f.write(video_file.read())

    # load model
    model = load_yolo(model_path, to_cuda=use_gpu)

    # prepare enroll encodings list for fast compare
    people = []
    for p in st.session_state["people_db"]:
        encs = [np.array(e) for e in p["encodings"]]
        people.append({"name": p["name"], "encodings": encs})

    cap = cv2.VideoCapture(tpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_path = os.path.join(tempfile.gettempdir(), f"out_{int(time.time()*1000)}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    progress = st.progress(0)
    placeholder = st.empty()
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inference
        results = model.predict(frame, conf=conf_th, verbose=False)
        boxes = results[0].boxes

        annotated = frame.copy()

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # expand bbox a bit for better face crop
            w_box = x2 - x1
            h_box = y2 - y1
            pad_w = int(w_box * (expand_box_ratio / 100.0))
            pad_h = int(h_box * (expand_box_ratio / 100.0))
            x1_e = max(0, x1 - pad_w)
            y1_e = max(0, y1 - pad_h)
            x2_e = min(w-1, x2 + pad_w)
            y2_e = min(h-1, y2 + pad_h)

            roi = frame[y1_e:y2_e, x1_e:x2_e]
            if roi.size == 0:
                continue
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb)
            matched_any = False
            best_name = None
            best_dist = 999.0

            if face_locs:
                encs = face_recognition.face_encodings(rgb, face_locs)
                for enc in encs:
                    for p in people:
                        for ref in p["encodings"]:
                            dist = np.linalg.norm(enc - ref)
                            if dist < best_dist:
                                best_dist = dist
                                best_name = p["name"]
                    # after compare all, check threshold
                    if best_dist <= face_tol:
                        matched_any = True
                        break

            color = (0,255,0) if matched_any else (0,0,255)
            label_text = best_name if matched_any else "Unknown"
            # draw on original coords
            cv2.rectangle(annotated, (x1_e, y1_e), (x2_e, y2_e), color, 2)
            cv2.putText(annotated, f"{label_text} {best_dist:.2f}" if matched_any else "Unknown",
                        (x1_e, max(15,y1_e-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(annotated)
        processed += 1
        if total_frames>0:
            progress.progress(min(1.0, processed/total_frames))
        preview = cv2.resize(annotated, (min(900,w), int(min(900,w)*h/w)))
        placeholder.image(preview, channels="BGR")

    cap.release()
    out.release()
    progress.empty()
    st.success("Hoàn thành nhận dạng!")
    with open(out_path, "rb") as f:
        st.download_button("📥 Tải video kết quả", f, file_name="yolo_face_result.mp4")
