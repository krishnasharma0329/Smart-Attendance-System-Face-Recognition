# Smart Attendance System
Face-recognition based attendance tracker using OpenCV + Tkinter + SQLite.

## Requirements
- Python 3.8+
- Webcam

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## How It Works

1. **Register Students**
   - Click `+ REGISTER STUDENT`
   - Enter name and roll number
   - The camera opens and captures 50 face images automatically
   - Model is trained immediately after capture

2. **Start Session**
   - Click `▶ START SESSION`
   - The camera detects and recognizes faces in real-time
   - Each student's time in frame is tracked continuously

3. **Attendance Rule**
   - Student must be detected for **≥ 50 minutes** out of 60 to be marked **Present**
   - Less than 50 minutes → **Absent**

4. **View Records**
   - Switch to the **ATTENDANCE** tab in the right panel
   - Filter by date or view all records
   - Present = green, Absent = red

## File Structure
```
attendance_system/
├── main.py            # Main application
├── requirements.txt   # Dependencies
├── attendance.db      # SQLite database (auto-created)
├── dataset/           # Face images (auto-created)
└── trainer.yml        # Trained model (auto-created after first registration)
```

## Notes
- `opencv-contrib-python` is required for the LBPH face recognizer
- The model re-trains every time a new student is registered
- Confidence threshold is set to 70 (lower = stricter matching)
