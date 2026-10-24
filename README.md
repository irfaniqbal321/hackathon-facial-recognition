# 🧠 Smart Attendance System - Facial Recognition

> Hackathon Project - Real-time Face Recognition Attendance System

## 🎯 Problem
Proxy attendance in universities & offices.

## ✅ Solution
AI-powered facial recognition jo automatically attendance mark karta hai date & time ke sath.

## ✨ Features
- 📸 **Enroll New Face** - 5 images se training
- 🎥 **Live Recognition** - Name + Confidence (e.g., m.irfan 43.5)
- 📋 **Auto Attendance** - CSV me Date/Time save
- 👥 **Multi-Person Detection** - Ek sath multiple faces
- 💾 **Offline** - Internet ki zarurat nahi

## 🛠️ Tech Stack
- Python
- OpenCV LBPH Recognizer
- Streamlit UI
- Pandas

## 🚀 How to Run
```bash
pip install opencv-contrib-python streamlit pandas pillow numpy
streamlit run app.py
