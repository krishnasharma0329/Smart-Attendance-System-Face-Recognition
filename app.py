import streamlit as st
import cv2
import os
import sqlite3
import numpy as np
from datetime import datetime, date
import time
import pandas as pd
from PIL import Image
import threading

# ─── Constants ────────────────────────────────────────────────────────────────
DB_PATH     = "attendance.db"
DATASET_DIR = "dataset"
MODEL_FILE  = "trainer.yml"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #080C14;
        color: #F1F5F9;
    }

    section[data-testid="stSidebar"] {
        background-color: #0D1321 !important;
        border-right: 1px solid #1E2A3A;
    }

    .metric-card {
        background: #111827;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .metric-card.blue  { border-color: #38BDF8; }
    .metric-card.green { border-color: #34D399; }
    .metric-card.red   { border-color: #F87171; }
    .metric-card.yellow{ border-color: #FBBF24; }

    .metric-icon  { font-size: 24px; }
    .metric-value { font-size: 32px; font-weight: 700; margin: 4px 0; }
    .metric-label { font-size: 12px; color: #64748B; }
    .metric-card.blue   .metric-value { color: #38BDF8; }
    .metric-card.green  .metric-value { color: #34D399; }
    .metric-card.red    .metric-value { color: #F87171; }
    .metric-card.yellow .metric-value { color: #FBBF24; }

    .session-bar {
        background: #1E293B;
        border-radius: 10px;
        padding: 14px 20px;
        margin: 10px 0 20px 0;
        display: flex;
        gap: 40px;
        align-items: center;
    }
    .session-item { font-size: 13px; color: #64748B; }
    .session-value { font-size: 15px; font-weight: 600; color: #38BDF8; }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 8px 20px !important;
    }

    .log-box {
        background: #0D1321;
        border: 1px solid #1E2A3A;
        border-radius: 10px;
        padding: 12px;
        height: 200px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 12px;
        color: #34D399;
    }

    div[data-testid="stDataFrame"] {
        background: #111827;
        border-radius: 10px;
    }

    .present-badge {
        background: #064E3B;
        color: #34D399;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .absent-badge {
        background: #450A0A;
        color: #F87171;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    h1, h2, h3 { color: #F1F5F9 !important; }
    .stTextInput > div > div > input {
        background: #111827 !important;
        color: #F1F5F9 !important;
        border: 1px solid #1E2A3A !important;
        border-radius: 8px !important;
    }
    .stSelectbox > div > div {
        background: #111827 !important;
        color: #F1F5F9 !important;
    }
    [data-testid="stMetricValue"] { color: #38BDF8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        roll_no TEXT UNIQUE NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date TEXT,
        first_seen TEXT,
        last_seen TEXT,
        total_minutes REAL DEFAULT 0,
        status TEXT DEFAULT 'Absent',
        FOREIGN KEY(student_id) REFERENCES students(id)
    )""")
    conn.commit()
    conn.close()

def get_all_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll_no FROM students ORDER BY roll_no")
    rows = c.fetchall()
    conn.close()
    return rows

def add_student(name, roll_no):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (name, roll_no) VALUES (?, ?)", (name, roll_no))
        conn.commit()
        return c.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_attendance_records(filter_date=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if filter_date:
        c.execute("""
            SELECT s.roll_no, s.name, a.date, a.first_seen, a.last_seen,
                   ROUND(a.total_minutes,1), a.status
            FROM attendance_log a JOIN students s ON a.student_id=s.id
            WHERE a.date=? ORDER BY s.roll_no
        """, (filter_date,))
    else:
        c.execute("""
            SELECT s.roll_no, s.name, a.date, a.first_seen, a.last_seen,
                   ROUND(a.total_minutes,1), a.status
            FROM attendance_log a JOIN students s ON a.student_id=s.id
            ORDER BY a.date DESC, s.roll_no
        """)
    rows = c.fetchall()
    conn.close()
    return rows

def get_today_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = str(date.today())
    c.execute("SELECT COUNT(*) FROM students")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM attendance_log WHERE date=? AND status='Present'", (today,))
    present = c.fetchone()[0]
    conn.close()
    return total, present

def upsert_attendance(student_id, minutes, first_seen, last_seen, session_minutes=0):
    today    = str(date.today())
    required = session_minutes * 0.9 if session_minutes > 0 else 1
    status   = "Present" if minutes >= required else "Absent"
    conn     = sqlite3.connect(DB_PATH)
    c        = conn.cursor()
    c.execute("SELECT id FROM attendance_log WHERE student_id=? AND date=?", (student_id, today))
    if c.fetchone():
        c.execute("""UPDATE attendance_log SET total_minutes=?, last_seen=?, status=?
                     WHERE student_id=? AND date=?""",
                  (minutes, last_seen, status, student_id, today))
    else:
        c.execute("""INSERT INTO attendance_log
                     (student_id, date, first_seen, last_seen, total_minutes, status)
                     VALUES (?,?,?,?,?,?)""",
                  (student_id, today, first_seen, last_seen, minutes, status))
    conn.commit()
    conn.close()
    return status

# ─── Face Functions ────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)

def train_model():
    ensure_dirs()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    for fname in os.listdir(DATASET_DIR):
        if not fname.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(DATASET_DIR, fname), cv2.IMREAD_GRAYSCALE)
        sid = int(fname.split(".")[1])
        faces.append(img)
        ids.append(sid)
    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.write(MODEL_FILE)
        return True
    return False

# ─── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "session_running":    False,
        "session_start":      None,
        "tracker":            {},
        "last_seen":          {},
        "log":                [],
        "page":               "Dashboard",
        "capture_mode":       False,
        "capture_student_id": None,
        "capture_name":       "",
        "capture_count":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px 0;'>
            <div style='font-size:40px;'>◈</div>
            <div style='font-size:20px; font-weight:700; color:#F1F5F9;'>AttendAI</div>
            <div style='font-size:11px; color:#64748B;'>Smart Attendance System</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**NAVIGATION**")

        pages = ["🏠 Dashboard", "👥 Students", "📋 Attendance"]
        for p in pages:
            label = p.split(" ", 1)[1]
            if st.button(p, key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label

        st.divider()
        st.markdown("**SESSION CONTROLS**")

        if not st.session_state.session_running:
            if st.button("▶  Start Session", use_container_width=True, type="primary"):
                st.session_state.session_running = True
                st.session_state.session_start   = time.time()
                st.session_state.tracker         = {}
                st.session_state.last_seen        = {}
                st.session_state.log.append(f"[{datetime.now().strftime('%H:%M')}] Session started")
                st.rerun()
        else:
            if st.button("■  Stop Session", use_container_width=True):
                st.session_state.session_running = False
                st.session_state.log.append(f"[{datetime.now().strftime('%H:%M')}] Session stopped")
                st.rerun()

        st.divider()

        # Session info
        if st.session_state.session_running and st.session_state.session_start:
            elapsed  = (time.time() - st.session_state.session_start) / 60.0
            required = elapsed * 0.9
            st.markdown(f"⏱ **Session:** `{elapsed:.1f} min`")
            st.markdown(f"✅ **Required (90%):** `{required:.1f} min`")
        else:
            st.markdown("⏹ No active session")

        st.divider()
        st.markdown(f"🕐 `{datetime.now().strftime('%H:%M:%S')}`")
        st.markdown(f"📅 `{date.today().strftime('%d %b %Y')}`")

# ─── Dashboard Page ────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown("## 🏠 Dashboard")

    total, present = get_today_stats()
    absent = total - present
    rate   = f"{int(present/total*100)}%" if total else "0%"

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card blue'>
            <div class='metric-icon'>👥</div>
            <div class='metric-value'>{total}</div>
            <div class='metric-label'>Total Students</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card green'>
            <div class='metric-icon'>✅</div>
            <div class='metric-value'>{present}</div>
            <div class='metric-label'>Present Today</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card red'>
            <div class='metric-icon'>❌</div>
            <div class='metric-value'>{absent}</div>
            <div class='metric-label'>Absent Today</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card yellow'>
            <div class='metric-icon'>📊</div>
            <div class='metric-value'>{rate}</div>
            <div class='metric-label'>Attendance Rate</div>
        </div>""", unsafe_allow_html=True)

    # Session bar
    if st.session_state.session_running and st.session_state.session_start:
        elapsed  = (time.time() - st.session_state.session_start) / 60.0
        required = elapsed * 0.9
        st.markdown(f"""<div class='session-bar'>
            <div class='session-item'>⏱ Session Duration <br><span class='session-value'>{elapsed:.1f} min</span></div>
            <div class='session-item'>✅ Required for Present (90%) <br><span class='session-value'>{required:.1f} min</span></div>
            <div class='session-item'>🔴 Status <br><span class='session-value'>LIVE</span></div>
        </div>""", unsafe_allow_html=True)

    # Camera + Log
    cam_col, log_col = st.columns([2, 1])

    with cam_col:
        st.markdown("### 🎥 Live Camera Feed")
        cam_placeholder = st.empty()

        if st.session_state.session_running:
            run_recognition(cam_placeholder)
        else:
            cam_placeholder.info("▶ Start a session to activate the camera feed.")

    with log_col:
        st.markdown("### 📋 Activity Log")
        log_html = "<div class='log-box'>"
        for entry in reversed(st.session_state.log[-30:]):
            log_html += f"<div>{entry}</div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

# ─── Face Recognition Loop ────────────────────────────────────────────────────
def run_recognition(placeholder):
    if not os.path.exists(MODEL_FILE):
        placeholder.warning("⚠️ No trained model found. Register students first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    students = {s[0]: (s[1], s[2]) for s in get_all_students()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        placeholder.error("❌ Cannot open camera. Make sure webcam is connected.")
        return

    session_start = st.session_state.session_start

    # Run for ~3 seconds per page refresh
    end_time = time.time() + 3
    while time.time() < end_time and st.session_state.session_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces   = cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        now     = time.time()
        now_str = datetime.now().strftime("%H:%M:%S")
        session_mins  = (now - session_start) / 60.0
        required_mins = session_mins * 0.9

        for (x, y, w, h) in faces:
            try:
                sid, conf = recognizer.predict(gray[y:y+h, x:x+w])
            except:
                continue

            if conf < 70 and sid in students:
                name, roll = students[sid]
                if sid not in st.session_state.tracker:
                    st.session_state.tracker[sid] = {
                        "first_seen": now_str, "last_seen": now_str, "minutes": 0.0
                    }
                    st.session_state.log.append(
                        f"[{datetime.now().strftime('%H:%M')}] {name} ({roll}) entered")
                else:
                    elapsed = (now - st.session_state.last_seen.get(sid, now)) / 60.0
                    st.session_state.tracker[sid]["minutes"] += elapsed
                    st.session_state.tracker[sid]["last_seen"] = now_str

                st.session_state.last_seen[sid] = now
                mins   = st.session_state.tracker[sid]["minutes"]
                pct    = (mins / session_mins * 100) if session_mins > 0 else 0
                status = upsert_attendance(sid, mins,
                                           st.session_state.tracker[sid]["first_seen"],
                                           now_str, session_mins)
                color = (52, 211, 153) if status == "Present" else (251, 191, 36)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{roll} {name}", (x, y-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"{mins:.1f}/{required_mins:.1f}min ({pct:.0f}%)",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (248, 113, 113), 2)
                cv2.putText(frame, "Unknown", (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 113, 113), 2)

        # Show frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb, channels="RGB", use_container_width=True)

    cap.release()
    # Auto-refresh while session is running
    if st.session_state.session_running:
        time.sleep(0.1)
        st.rerun()

# ─── Students Page ─────────────────────────────────────────────────────────────
def page_students():
    st.markdown("## 👥 Students")

    # Register form
    with st.expander("➕ Register New Student", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", placeholder="e.g. Krishu Sharma")
        with col2:
            roll = st.text_input("Roll Number", placeholder="e.g. CS101")

        if st.button("Register & Capture Face", type="primary"):
            if not name or not roll:
                st.error("Please enter both name and roll number.")
            else:
                sid = add_student(name.strip(), roll.strip())
                if sid is None:
                    st.error("Roll number already exists!")
                else:
                    st.session_state.capture_mode       = True
                    st.session_state.capture_student_id = sid
                    st.session_state.capture_name       = name
                    st.session_state.capture_count      = 0
                    st.success(f"Student {name} registered! Now capturing face...")
                    st.rerun()

    # Face capture mode
    if st.session_state.capture_mode:
        capture_faces_web()
        return

    # Students table
    students = get_all_students()
    if students:
        st.markdown(f"**{len(students)} students registered**")
        df = pd.DataFrame(students, columns=["ID", "Name", "Roll No"])
        df = df[["Roll No", "Name"]]
        df.index = range(1, len(df)+1)
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No students registered yet. Add your first student above!")

# ─── Face Capture (Web) ────────────────────────────────────────────────────────
def capture_faces_web():
    sid   = st.session_state.capture_student_id
    name  = st.session_state.capture_name
    count = st.session_state.capture_count

    st.markdown(f"### 📸 Capturing face for **{name}**")
    progress = st.progress(count / 50)
    st.markdown(f"**{count} / 50** images captured")

    cam_placeholder = st.empty()
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        capture = st.button("📸 Capture Frame", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("✅ Done Capturing", use_container_width=True):
            if count > 0:
                with st.spinner("Training model..."):
                    success = train_model()
                if success:
                    st.success("Model trained successfully!")
                else:
                    st.warning("Training failed — no images found.")
            st.session_state.capture_mode = False
            st.rerun()

    # Open camera and show frame
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (56, 189, 248), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, channels="RGB", use_container_width=True)

            if capture and len(faces) > 0:
                ensure_dirs()
                for (x, y, w, h) in faces:
                    count += 1
                    cv2.imwrite(
                        os.path.join(DATASET_DIR, f"User.{sid}.{count}.jpg"),
                        gray[y:y+h, x:x+w])
                st.session_state.capture_count = count
                cap.release()
                st.rerun()
            elif capture and len(faces) == 0:
                st.warning("No face detected in frame. Please look at the camera.")
        cap.release()
    else:
        st.error("Cannot open camera.")

# ─── Attendance Page ────────────────────────────────────────────────────────────
def page_attendance():
    st.markdown("## 📋 Attendance Records")

    # Filter
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        filter_date = st.date_input("Filter by date", value=date.today())
    with col2:
        show_all = st.checkbox("Show all dates")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("⟳ Refresh", use_container_width=True)

    fd = None if show_all else str(filter_date)
    records = get_attendance_records(fd)

    if records:
        df = pd.DataFrame(records,
                          columns=["Roll No", "Name", "Date",
                                   "First Seen", "Last Seen", "Minutes", "Status"])

        # Color status column
        def style_status(val):
            if val == "Present":
                return "background-color: #064E3B; color: #34D399; font-weight: bold; border-radius: 20px;"
            return "background-color: #450A0A; color: #F87171; font-weight: bold; border-radius: 20px;"

        styled = df.style.applymap(style_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, height=450)

        # Summary
        present_count = len(df[df["Status"] == "Present"])
        absent_count  = len(df[df["Status"] == "Absent"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Present", present_count)
        col3.metric("Absent", absent_count)
    else:
        st.info("No attendance records found for the selected date.")

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_db()
    init_state()
    render_sidebar()

    page = st.session_state.page

    if page == "Dashboard":
        page_dashboard()
    elif page == "Students":
        page_students()
    elif page == "Attendance":
        page_attendance()

if __name__ == "__main__":
    main()
