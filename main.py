import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import os
import sqlite3
import numpy as np
from datetime import datetime, date
import threading
import time
from PIL import Image, ImageTk

# ─── Constants ────────────────────────────────────────────────────────────────
DB_PATH     = "attendance.db"
DATASET_DIR = "dataset"
MODEL_FILE  = "trainer.yml"

C = {
    "bg":      "#080C14",
    "surface": "#0D1321",
    "card":    "#111827",
    "border":  "#1E2A3A",
    "accent":  "#38BDF8",
    "accent2": "#818CF8",
    "green":   "#34D399",
    "red":     "#F87171",
    "yellow":  "#FBBF24",
    "text":    "#F1F5F9",
    "muted":   "#64748B",
    "subtle":  "#1E293B",
}

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
    today     = str(date.today())
    required  = session_minutes * 0.9 if session_minutes > 0 else 50
    status    = "Present" if minutes >= required else "Absent"
    conn      = sqlite3.connect(DB_PATH)
    c      = conn.cursor()
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

# ─── Face Capture & Train ─────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)

def capture_faces(student_id, name, status_cb):
    ensure_dirs()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap     = cv2.VideoCapture(0)
    count   = 0
    status_cb(f"Capturing: {name} (0/50)")
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(os.path.join(DATASET_DIR, f"User.{student_id}.{count}.jpg"),
                        gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (56, 189, 248), 2)
            status_cb(f"Capturing: {name} ({count}/50)")
        cv2.imshow(f"Capturing — {name} | Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    status_cb("Training model...")
    train_model(status_cb)

def train_model(status_cb=None):
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
        if status_cb:
            status_cb("Model ready! Start a session.")
    else:
        if status_cb:
            status_cb("No training data found.")

# ─── Attendance Session ────────────────────────────────────────────────────────
class AttendanceSession:
    def __init__(self, status_cb, frame_cb, log_cb, stats_cb):
        self.running   = False
        self.status_cb = status_cb
        self.frame_cb  = frame_cb
        self.log_cb    = log_cb
        self.stats_cb  = stats_cb
        self.tracker   = {}
        self.last_seen = {}

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def _run(self):
        if not os.path.exists(MODEL_FILE):
            self.status_cb("No trained model. Register students first.")
            return
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)
        cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        students      = {s[0]: (s[1], s[2]) for s in get_all_students()}
        cap           = cv2.VideoCapture(0)
        session_start = time.time()
        self.status_cb("Session live — detecting faces...")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
            now     = time.time()
            now_str = datetime.now().strftime("%H:%M:%S")

            for (x, y, w, h) in faces:
                try:
                    sid, conf = recognizer.predict(gray[y:y+h, x:x+w])
                except:
                    continue
                session_mins = (now - session_start) / 60.0
                required_mins = session_mins * 0.9
                if conf < 70 and sid in students:
                    name, roll = students[sid]
                    if sid not in self.tracker:
                        self.tracker[sid] = {"first_seen": now_str,
                                             "last_seen": now_str, "minutes": 0.0}
                        self.log_cb(f"{name} ({roll}) entered")
                    else:
                        elapsed = (now - self.last_seen.get(sid, now)) / 60.0
                        self.tracker[sid]["minutes"] += elapsed
                        self.tracker[sid]["last_seen"] = now_str
                    self.last_seen[sid] = now
                    mins = self.tracker[sid]["minutes"]
                    pct  = (mins / session_mins * 100) if session_mins > 0 else 0
                    upsert_attendance(sid, mins, self.tracker[sid]["first_seen"],
                                      now_str, session_mins)
                    color = (52, 211, 153)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{roll} {name}", (x, y-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.putText(frame, f"{mins:.1f}/{required_mins:.1f} min  ({pct:.0f}%)",
                                (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (251, 191, 36), 1)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (248, 113, 113), 2)
                    cv2.putText(frame, "Unknown", (x, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 113, 113), 2)

            self.frame_cb(frame)
            session_elapsed = (time.time() - session_start) / 60.0
            self.status_cb(f"Live  |  Session: {session_elapsed:.1f} min  |  Need 90%")
            self.stats_cb(session_start)
            time.sleep(0.03)

        cap.release()
        self.status_cb("Session ended.")

# ─── Custom Widgets ────────────────────────────────────────────────────────────
class RoundBtn(tk.Canvas):
    def __init__(self, parent, text, cmd, bg, fg="#000", w=160, h=36, r=8):
        super().__init__(parent, width=w, height=h,
                         bg=parent["bg"], highlightthickness=0)
        self._bg, self._fg, self._r = bg, fg, r
        self._text = text
        self._cmd  = cmd
        self._hover = self._lighten(bg)
        self._draw(bg)
        self.bind("<Enter>",    lambda e: self._draw(self._hover))
        self.bind("<Leave>",    lambda e: self._draw(self._bg))
        self.bind("<Button-1>", lambda e: cmd())

    def _lighten(self, h):
        r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
        return f"#{min(255,r+25):02x}{min(255,g+25):02x}{min(255,b+25):02x}"

    def _draw(self, color):
        self.delete("all")
        w = self.winfo_reqwidth(); h = self.winfo_reqheight(); r = self._r
        self.create_arc(0,0,2*r,2*r, start=90, extent=90, fill=color, outline=color)
        self.create_arc(w-2*r,0,w,2*r, start=0, extent=90, fill=color, outline=color)
        self.create_arc(0,h-2*r,2*r,h, start=180, extent=90, fill=color, outline=color)
        self.create_arc(w-2*r,h-2*r,w,h, start=270, extent=90, fill=color, outline=color)
        self.create_rectangle(r,0,w-r,h, fill=color, outline=color)
        self.create_rectangle(0,r,w,h-r, fill=color, outline=color)
        self.create_text(w//2, h//2, text=self._text, fill=self._fg,
                         font=("Helvetica", 10, "bold"))

class StatCard(tk.Frame):
    def __init__(self, parent, label, var, icon, color):
        super().__init__(parent, bg=C["card"], padx=18, pady=14)
        # left accent bar
        bar = tk.Frame(self, bg=color, width=4)
        bar.place(x=0, y=0, relheight=1)
        tk.Label(self, text=icon, font=("Helvetica", 22),
                 bg=C["card"], fg=color).pack(anchor="w", padx=(10,0))
        tk.Label(self, textvariable=var, font=("Helvetica", 28, "bold"),
                 bg=C["card"], fg=color).pack(anchor="w", padx=(10,0))
        tk.Label(self, text=label, font=("Helvetica", 9),
                 bg=C["card"], fg=C["muted"]).pack(anchor="w", padx=(10,0))

# ─── App ──────────────────────────────────────────────────────────────────────
class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AttendAI — Smart Attendance System")
        self.geometry("1250x760")
        self.minsize(1050, 660)
        self.configure(bg=C["bg"])
        self.session = None

        self.var_total    = tk.StringVar(value="0")
        self.var_present  = tk.StringVar(value="0")
        self.var_absent   = tk.StringVar(value="0")
        self.var_rate     = tk.StringVar(value="0%")
        self.var_session  = tk.StringVar(value="No session")
        self.var_required = tk.StringVar(value="—")
        self.session_start_time = None

        init_db()
        self._setup_styles()
        self._build_sidebar()
        self._build_main()
        self._update_clock()
        self._update_stats()
        self._show_dashboard()

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("default")
        for name in ("A.Treeview", "B.Treeview"):
            s.configure(name, background=C["card"], foreground=C["text"],
                        fieldbackground=C["card"], font=("Helvetica", 10),
                        rowheight=38, borderwidth=0)
            s.configure(f"{name}.Heading", background=C["subtle"],
                        foreground=C["accent"], font=("Helvetica", 9, "bold"), relief="flat")
            s.map(name, background=[("selected", C["subtle"])],
                  foreground=[("selected", C["accent"])])

    # ── Sidebar ───────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sb = tk.Frame(self, bg=C["surface"], width=240)
        self.sb.pack(side="left", fill="y")
        self.sb.pack_propagate(False)

        # Logo block
        logo = tk.Frame(self.sb, bg=C["surface"], pady=28)
        logo.pack(fill="x")
        tk.Label(logo, text="◈", font=("Helvetica", 32), bg=C["surface"],
                 fg=C["accent"]).pack()
        tk.Label(logo, text="AttendAI", font=("Helvetica", 17, "bold"),
                 bg=C["surface"], fg=C["text"]).pack()
        tk.Label(logo, text="Smart Attendance System", font=("Helvetica", 8),
                 bg=C["surface"], fg=C["muted"]).pack(pady=(2,0))

        tk.Frame(self.sb, bg=C["border"], height=1).pack(fill="x", padx=18, pady=10)

        # Navigation
        tk.Label(self.sb, text="NAVIGATION", font=("Helvetica", 8, "bold"),
                 bg=C["surface"], fg=C["muted"]).pack(anchor="w", padx=22, pady=(6,4))

        self._active_nav = None
        self._nav_frames = {}
        navs = [
            ("🏠", "Dashboard",  self._show_dashboard),
            ("👥", "Students",   self._show_students),
            ("📋", "Attendance", self._show_attendance),
        ]
        for icon, label, cmd in navs:
            self._make_nav(icon, label, cmd)

        tk.Frame(self.sb, bg=C["border"], height=1).pack(fill="x", padx=18, pady=14)

        # Controls
        tk.Label(self.sb, text="CONTROLS", font=("Helvetica", 8, "bold"),
                 bg=C["surface"], fg=C["muted"]).pack(anchor="w", padx=22, pady=(0,8))

        RoundBtn(self.sb, "▶  Start Session", self.start_session,
                 bg=C["green"], fg="#000", w=200, h=38).pack(padx=20, pady=4)
        RoundBtn(self.sb, "■  Stop Session", self.stop_session,
                 bg=C["red"], fg="#fff", w=200, h=38).pack(padx=20, pady=4)
        RoundBtn(self.sb, "+  Register Student", self.register_student,
                 bg=C["accent"], fg="#000", w=200, h=38).pack(padx=20, pady=4)

        tk.Frame(self.sb, bg=C["border"], height=1).pack(fill="x", padx=18, pady=14)

        # Status
        tk.Label(self.sb, text="STATUS", font=("Helvetica", 8, "bold"),
                 bg=C["surface"], fg=C["muted"]).pack(anchor="w", padx=22)
        self.status_var = tk.StringVar(value="Ready to go!")
        tk.Label(self.sb, textvariable=self.status_var, font=("Helvetica", 9),
                 bg=C["surface"], fg=C["accent"], wraplength=200,
                 justify="left").pack(anchor="w", padx=22, pady=6)

        # Clock
        self.clock_var = tk.StringVar()
        tk.Label(self.sb, textvariable=self.clock_var, font=("Helvetica", 10, "bold"),
                 bg=C["surface"], fg=C["muted"]).pack(side="bottom", pady=20)

    def _make_nav(self, icon, label, cmd):
        f = tk.Frame(self.sb, bg=C["surface"], cursor="hand2")
        f.pack(fill="x", padx=12, pady=2)
        self._nav_frames[label] = f

        il = tk.Label(f, text=icon, font=("Helvetica", 13),
                      bg=C["surface"], fg=C["accent"])
        il.pack(side="left", padx=(14,8), pady=11)
        tl = tk.Label(f, text=label, font=("Helvetica", 11),
                      bg=C["surface"], fg=C["text"])
        tl.pack(side="left")

        def on_click(e=None):
            if self._active_nav:
                nf = self._nav_frames[self._active_nav]
                nf.configure(bg=C["surface"])
                for w in nf.winfo_children():
                    w.configure(bg=C["surface"])
            self._active_nav = label
            f.configure(bg=C["subtle"])
            for w in f.winfo_children():
                w.configure(bg=C["subtle"])
            cmd()

        for widget in [f, il, tl]:
            widget.bind("<Button-1>", on_click)
            widget.bind("<Enter>", lambda e, fr=f: fr.configure(bg=C["subtle"]) if self._active_nav != label else None)
            widget.bind("<Leave>", lambda e, fr=f: fr.configure(bg=C["subtle"] if self._active_nav==label else C["surface"]))

    # ── Main Area ─────────────────────────────────────────────────────────
    def _build_main(self):
        self.main = tk.Frame(self, bg=C["bg"])
        self.main.pack(side="right", fill="both", expand=True)

        # Top header bar
        self.header = tk.Frame(self.main, bg=C["surface"], pady=16)
        self.header.pack(fill="x")
        self.page_title = tk.Label(self.header, text="", font=("Helvetica", 17, "bold"),
                                   bg=C["surface"], fg=C["text"])
        self.page_title.pack(side="left", padx=26)
        self.page_sub = tk.Label(self.header, text="", font=("Helvetica", 10),
                                 bg=C["surface"], fg=C["muted"])
        self.page_sub.pack(side="right", padx=26)

        # Content
        self.content = tk.Frame(self.main, bg=C["bg"])
        self.content.pack(fill="both", expand=True, padx=22, pady=18)

    def _clear(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _set_active_nav(self, label):
        for lbl, f in self._nav_frames.items():
            bg = C["subtle"] if lbl == label else C["surface"]
            f.configure(bg=bg)
            for w in f.winfo_children():
                w.configure(bg=bg)
        self._active_nav = label

    # ── Dashboard ─────────────────────────────────────────────────────────
    def _show_dashboard(self):
        self._clear()
        self._set_active_nav("Dashboard")
        self.page_title.config(text="Dashboard")
        self.page_sub.config(text=date.today().strftime("%A, %d %B %Y"))

        # Stat cards
        row = tk.Frame(self.content, bg=C["bg"])
        row.pack(fill="x", pady=(0, 18))
        for label, var, icon, color in [
            ("Total Students",  self.var_total,   "👥", C["accent"]),
            ("Present Today",   self.var_present, "✅", C["green"]),
            ("Absent Today",    self.var_absent,  "❌", C["red"]),
            ("Attendance Rate", self.var_rate,    "📊", C["yellow"]),
        ]:
            StatCard(row, label, var, icon, color).pack(
                side="left", fill="both", expand=True, padx=(0,14))

        # Session info bar
        info_bar = tk.Frame(self.content, bg=C["subtle"], pady=10)
        info_bar.pack(fill="x", pady=(0, 14))
        tk.Label(info_bar, text="⏱  Session Duration:", font=("Helvetica", 10),
                 bg=C["subtle"], fg=C["muted"]).pack(side="left", padx=(16,6))
        tk.Label(info_bar, textvariable=self.var_session, font=("Helvetica", 10, "bold"),
                 bg=C["subtle"], fg=C["accent"]).pack(side="left", padx=(0,24))
        tk.Label(info_bar, text="✅  Required for Present (90%):", font=("Helvetica", 10),
                 bg=C["subtle"], fg=C["muted"]).pack(side="left", padx=(0,6))
        tk.Label(info_bar, textvariable=self.var_required, font=("Helvetica", 10, "bold"),
                 bg=C["subtle"], fg=C["green"]).pack(side="left")
        tk.Label(info_bar, text="Rule: Student must attend 90% of session to be marked Present",
                 font=("Helvetica", 9), bg=C["subtle"], fg=C["muted"]).pack(side="right", padx=16)

        # Camera + log
        mid = tk.Frame(self.content, bg=C["bg"])
        mid.pack(fill="both", expand=True)

        # Camera card
        cam_card = tk.Frame(mid, bg=C["card"])
        cam_card.pack(side="left", fill="both", expand=True)
        hdr = tk.Frame(cam_card, bg=C["card"])
        hdr.pack(fill="x", padx=14, pady=10)
        tk.Label(hdr, text="● LIVE FEED", font=("Helvetica", 9, "bold"),
                 bg=C["card"], fg=C["red"]).pack(side="left")
        tk.Frame(cam_card, bg=C["border"], height=1).pack(fill="x", padx=14)
        self.cam_label = tk.Label(cam_card, bg="#050810",
                                  text="No feed active\nStart a session to see camera",
                                  fg=C["muted"], font=("Helvetica", 11), justify="center")
        self.cam_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Log card
        log_card = tk.Frame(mid, bg=C["card"], width=290)
        log_card.pack(side="right", fill="y", padx=(14,0))
        log_card.pack_propagate(False)
        tk.Label(log_card, text="ACTIVITY LOG", font=("Helvetica", 9, "bold"),
                 bg=C["card"], fg=C["muted"]).pack(anchor="w", padx=14, pady=(10,0))
        tk.Frame(log_card, bg=C["border"], height=1).pack(fill="x", padx=14, pady=6)
        self.log_text = tk.Text(log_card, bg=C["card"], fg=C["text"],
                                font=("Courier", 9), relief="flat",
                                state="disabled", wrap="word")
        sb = tk.Scrollbar(log_card, command=self.log_text.yview, bg=C["card"])
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=(8,0), pady=(0,10))

    # ── Students ──────────────────────────────────────────────────────────
    def _show_students(self):
        self._clear()
        self._set_active_nav("Students")
        self.page_title.config(text="Students")
        self.page_sub.config(text=f"{len(get_all_students())} registered")

        top = tk.Frame(self.content, bg=C["bg"])
        top.pack(fill="x", pady=(0,12))
        tk.Label(top, text="All registered students", font=("Helvetica", 10),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        RoundBtn(top, "+  Add Student", self.register_student,
                 bg=C["accent"], fg="#000", w=140, h=32).pack(side="right")

        card = tk.Frame(self.content, bg=C["card"])
        card.pack(fill="both", expand=True)

        cols = ("#", "Roll No", "Name")
        self.st_tree = ttk.Treeview(card, columns=cols, show="headings", style="A.Treeview")
        self.st_tree.heading("#",       text="#")
        self.st_tree.heading("Roll No", text="ROLL NO")
        self.st_tree.heading("Name",    text="STUDENT NAME")
        self.st_tree.column("#",       width=60,  anchor="center")
        self.st_tree.column("Roll No", width=140, anchor="center")
        self.st_tree.column("Name",    width=400)
        vsb = ttk.Scrollbar(card, orient="vertical", command=self.st_tree.yview)
        self.st_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.st_tree.pack(fill="both", expand=True, padx=2, pady=2)
        self._load_students()

    def _load_students(self):
        if not hasattr(self, "st_tree") or not self.st_tree.winfo_exists():
            return
        for r in self.st_tree.get_children():
            self.st_tree.delete(r)
        for i, s in enumerate(get_all_students(), 1):
            self.st_tree.insert("", "end", values=(i, s[2], s[1]))

    # ── Attendance ────────────────────────────────────────────────────────
    def _show_attendance(self):
        self._clear()
        self._set_active_nav("Attendance")
        self.page_title.config(text="Attendance Records")
        self.page_sub.config(text="50+ min = Present")

        # Filter row
        fr = tk.Frame(self.content, bg=C["bg"])
        fr.pack(fill="x", pady=(0,12))
        tk.Label(fr, text="Filter by date:", font=("Helvetica", 10),
                 bg=C["bg"], fg=C["muted"]).pack(side="left", padx=(0,8))
        self.date_var = tk.StringVar(value=str(date.today()))
        tk.Entry(fr, textvariable=self.date_var, bg=C["card"], fg=C["text"],
                 font=("Helvetica", 10), relief="flat", width=13,
                 insertbackground=C["text"]).pack(side="left", padx=(0,8), ipady=5)
        RoundBtn(fr, "Filter", self._load_att, bg=C["accent"], fg="#000", w=80, h=32).pack(side="left", padx=4)
        RoundBtn(fr, "All", lambda: [self.date_var.set(""), self._load_att()],
                 bg=C["subtle"], fg=C["text"], w=60, h=32).pack(side="left", padx=4)
        RoundBtn(fr, "⟳ Refresh", self._load_att,
                 bg=C["subtle"], fg=C["text"], w=100, h=32).pack(side="right")

        card = tk.Frame(self.content, bg=C["card"])
        card.pack(fill="both", expand=True)

        cols = ("Roll", "Name", "Date", "First Seen", "Last Seen", "Minutes", "Status")
        self.att_tree = ttk.Treeview(card, columns=cols, show="headings", style="B.Treeview")
        widths = [90, 180, 100, 95, 95, 80, 100]
        for c, w in zip(cols, widths):
            self.att_tree.heading(c, text=c.upper())
            self.att_tree.column(c, width=w, anchor="center")
        self.att_tree.tag_configure("present", foreground=C["green"])
        self.att_tree.tag_configure("absent",  foreground=C["red"])
        vsb = ttk.Scrollbar(card, orient="vertical", command=self.att_tree.yview)
        self.att_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.att_tree.pack(fill="both", expand=True, padx=2, pady=2)
        self._load_att()

    def _load_att(self):
        if not hasattr(self, "att_tree") or not self.att_tree.winfo_exists():
            return
        for r in self.att_tree.get_children():
            self.att_tree.delete(r)
        fd = getattr(self, "date_var", None)
        fd = fd.get().strip() if fd else None
        for rec in get_attendance_records(fd or None):
            tag = "present" if rec[6] == "Present" else "absent"
            self.att_tree.insert("", "end",
                                 values=(rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6]),
                                 tags=(tag,))

    # ── Helpers ──────────────────────────────────────────────────────────
    def _update_clock(self):
        now = datetime.now()
        self.clock_var.set(now.strftime("%H:%M:%S\n%d %b %Y"))
        self.after(1000, self._update_clock)

    def _update_stats(self, session_start=None):
        if session_start:
            self.session_start_time = session_start
        total, present = get_today_stats()
        absent = total - present
        rate   = f"{int(present/total*100)}%" if total else "0%"
        self.var_total.set(str(total))
        self.var_present.set(str(present))
        self.var_absent.set(str(absent))
        self.var_rate.set(rate)
        if self.session_start_time:
            elapsed  = (time.time() - self.session_start_time) / 60.0
            required = elapsed * 0.9
            self.var_session.set(f"{elapsed:.1f} min")
            self.var_required.set(f"{required:.1f} min  (90%)")
        self.after(5000, self._update_stats)

    def log(self, msg):
        if not hasattr(self, "log_text") or not self.log_text.winfo_exists():
            return
        self.log_text.config(state="normal")
        ts = datetime.now().strftime("%H:%M")
        self.log_text.insert("end", f"[{ts}]  {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def set_status(self, msg):
        self.status_var.set(msg)

    def update_frame(self, frame):
        if not hasattr(self, "cam_label") or not self.cam_label.winfo_exists():
            return
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            w     = max(self.cam_label.winfo_width(), 400)
            h     = max(self.cam_label.winfo_height(), 300)
            img   = img.resize((w, h), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        except Exception:
            pass

    def register_student(self):
        name = simpledialog.askstring("Register Student", "Enter student full name:", parent=self)
        if not name:
            return
        roll = simpledialog.askstring("Register Student", "Enter roll number:", parent=self)
        if not roll:
            return
        sid = add_student(name.strip(), roll.strip())
        if sid is None:
            messagebox.showerror("Error", "Roll number already exists!")
            return
        self._load_students()
        self._update_stats()
        messagebox.showinfo("Capture Faces",
                            f"Camera will now capture face data for {name}.\n"
                            "Look at the camera and slightly move your head.\n"
                            "Press Q to stop early.")
        threading.Thread(target=capture_faces,
                         args=(sid, name, self.set_status), daemon=True).start()

    def start_session(self):
        if self.session and self.session.running:
            return
        self.session_start_time = None
        self.var_session.set("0.0 min")
        self.var_required.set("0.0 min  (90%)")
        self._show_dashboard()
        self.session = AttendanceSession(
            status_cb=self.set_status,
            frame_cb=self.update_frame,
            log_cb=self.log,
            stats_cb=self._update_stats,
        )
        self.session.start()
        self.log("Session started")

    def stop_session(self):
        if self.session:
            self.session.stop()
        self.session_start_time = None
        self.var_session.set("No session")
        self.var_required.set("—")
        self.log("Session stopped")


if __name__ == "__main__":
    init_db()
    app = AttendanceApp()
    app.mainloop()
