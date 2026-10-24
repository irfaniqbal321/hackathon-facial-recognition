import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import urllib.request

st.set_page_config(page_title="Facial Recognition Hackathon", layout="wide")
st.title("🧠 Facial Recognition - FINAL Version")

DB_PATH = "face_db"
ATTENDANCE_FILE = "attendance.csv"
CASCADE_FILE = "haarcascade_frontalface_default.xml"

os.makedirs(DB_PATH, exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name","Date","Time"]).to_csv(ATTENDANCE_FILE, index=False)

# --- FIX 1: Cascade file ko download karo agar na ho ---
def get_cascade():
    # pehle opencv wala path try karo
    path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
    if os.path.exists(path):
        c = cv2.CascadeClassifier(path)
        if not c.empty():
            return c
    # agar fail ho to local download karo
    if not os.path.exists(CASCADE_FILE):
        try:
            st.warning("Haarcascade download ho raha hai... 1 baar wait karo")
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{CASCADE_FILE}",
                CASCADE_FILE
            )
        except:
            pass
    if os.path.exists(CASCADE_FILE):
        return cv2.CascadeClassifier(CASCADE_FILE)
    return cv2.CascadeClassifier()

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = get_cascade()

menu = st.sidebar.selectbox("Menu", ["Enroll New Face", "Live Recognition", "Attendance Log"])

def train_model():
    faces, labels, label_map = [], [], {}
    curr_id = 0
    for person in os.listdir(DB_PATH):
        person_dir = os.path.join(DB_PATH, person)
        if not os.path.isdir(person_dir): 
            continue
        if person not in label_map:
            label_map[person] = curr_id
            curr_id += 1
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(label_map[person])
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        return label_map
    return {}

# ============ ENROLL ============
if menu == "Enroll New Face":
    st.header("📸 Enroll New Face")
    name = st.text_input("Name", value="m.irfan")
    uploaded = st.file_uploader("Apni 5-6 photos upload karo", accept_multiple_files=True, type=['jpg','jpeg','png'])
    if uploaded and name:
        person_dir = os.path.join(DB_PATH, name)
        os.makedirs(person_dir, exist_ok=True)
        saved = 0
        for file in uploaded:
            try:
                pil_img = Image.open(file).convert("L")
                pil_img = pil_img.resize((200,200))
                pil_img.save(os.path.join(person_dir, f"img_{saved}_{file.name}.jpg"))
                saved += 1
            except:
                pass
        st.success(f"✅ {saved} photos saved for {name}!")
        if saved>0:
            train_model()

# ============ LIVE RECOGNITION ============
elif menu == "Live Recognition":
    st.header("🎥 Live Recognition")
    if face_cascade.empty():
        st.error("Cascade file load nahi hua, dubara refresh karo")
    else:
        st.info(f"Cascade OK! Total persons: {len(os.listdir(DB_PATH))}")
    
    if st.button("Train & Start Camera"):
        label_map = train_model()
        if not label_map:
            st.error("Pehle Enroll karo! face_db khali hai")
        else:
            inv_map = {v:k for k,v in label_map.items()}
            st.success(f"Model Trained! Persons: {list(label_map.keys())}")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)
            
            FRAME = st.empty()
            st.write("Camera chal raha hai... Stop karne ke liye page refresh karo")
            
            while True:
                ret, frame = cap.read()
                if not ret or frame is None: 
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray is None or gray.size == 0:
                    continue
                
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x,y,w,h) in faces:
                    roi = cv2.resize(gray[y:y+h, x:x+w], (200,200))
                    try:
                        label, conf = recognizer.predict(roi)
                        if conf < 75:
                            name = inv_map[label]
                            color = (0,255,0)
                            df = pd.read_csv(ATTENDANCE_FILE)
                            now = datetime.now()
                            today = now.strftime("%Y-%m-%d")
                            if not ((df['Name']==name) & (df['Date']==today)).any():
                                pd.DataFrame([[name, today, now.strftime("%H:%M:%S")]], columns=["Name","Date","Time"]).to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
                        else:
                            name = "Unknown"
                            color = (0,0,255)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{name} {conf:.0f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    except:
                        pass
                FRAME.image(frame, channels="BGR")

# ============ ATTENDANCE LOG ============
elif menu == "Attendance Log":
    st.header("📋 Attendance Log")
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df, use_container_width=True)