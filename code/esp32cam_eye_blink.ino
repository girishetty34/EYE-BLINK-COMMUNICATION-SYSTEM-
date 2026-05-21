import cv2
import mediapipe as mp
import time
import math
import threading
from collections import deque

# ================= MORSE TABLE =================
MORSE_TABLE = {
    '.-': 'A',   '-...': 'B', '-.-.': 'C', '-..': 'D',
    '.': 'E',    '..-.': 'F', '--.': 'G',  '....': 'H',
    '..': 'I',   '.---': 'J', '-.-': 'K',  '.-..': 'L',
    '--': 'M',   '-.': 'N',   '---': 'O',  '.--.': 'P',
    '--.-': 'Q', '.-.': 'R',  '...': 'S',  '-': 'T',
    '..-': 'U',  '...-': 'V', '.--': 'W',  '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9',
}

def decode_morse(code):
    return MORSE_TABLE.get(code, f'[?{code}]')

# ================= ON-SCREEN LOG =================
on_screen_log = deque(maxlen=5)

def log(msg):
    """Print to terminal AND push to on-screen rolling log."""
    ts = time.strftime("%H:%M:%S")
    print(f"\n  [{ts}] {msg}", flush=True)
    on_screen_log.append(msg)

# ================= FLASH BANNER =================
flash_text  = ""
flash_until = 0.0
flash_color = (0, 255, 0)

def flash(msg, duration=1.8, color=(0, 255, 0)):
    global flash_text, flash_until, flash_color
    flash_text  = msg
    flash_until = time.time() + duration
    flash_color = color
    log(msg)

# ================= MEDIAPIPE =================
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# ================= CAMERA =================
ESP32_CAM_IP   = "10.239.9.32"
ESP32_CAM_PORT = 81
ESP32_STREAM   = f"http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/stream"

print(f"\n[INFO] Connecting to: {ESP32_STREAM}")
cap = cv2.VideoCapture(ESP32_STREAM, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open stream at {ESP32_STREAM}\n"
        "  • Check IP and port.\n"
        "  • Same Wi-Fi network?\n"
        "  • Open URL in browser to confirm."
    )
print("[OK]   Stream opened.\n")

# ================= THREADED FRAME GRABBER =================
class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap     = cap
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

grabber = FrameGrabber(cap)
grabber.start()
print("[OK]   Frame grabber thread started.\n")

# ================= LANDMARK IDS =================
LEFT_EYE   = [33,  160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS = [473, 474, 475, 476]

# ================= THRESHOLDS =================
EAR_CLOSE      = 0.18
EAR_OPEN       = 0.22
DOT_DASH_LIMIT = 0.3
LETTER_GAP     = 1.2
WORD_GAP       = 2.5
RATIO_RIGHT    = 0.57
RATIO_LEFT     = 0.37
GAZE_COOLDOWN  = 0.8
DECISION_GAP   = 1.0

# ================= MODE =================
MODE = None

# ================= MODE-SELECT BLINK STATE =================
sel_eye_closed  = False
sel_blink_count = 0
sel_blink_time  = 0.0
SEL_BLINK_WINDOW = 1.5   # seconds to wait for a 2nd blink before deciding

# ================= SHARED STATE =================
eye_closed     = False
close_start    = 0.0
last_gaze_time = 0.0
avg_ear        = 0.0

# ================= MOVEMENT MODE STATE =================
WAITING = "WAITING"
INPUT   = "INPUT"
mv_state        = WAITING
blink_count     = 0
left_count      = 0
right_count     = 0
last_event_time = time.time()

# ================= MORSE MODE STATE =================
current_morse    = ""
decoded_text     = ""
last_symbol_time = time.time()

# ================= FPS TRACKING =================
fps_prev          = time.time()
fps               = 0.0
PRINT_INTERVAL    = 0.3
last_status_print = 0.0

# ================= HELPERS =================
def ear(lm, eye):
    A = math.dist((lm[eye[1]].x, lm[eye[1]].y), (lm[eye[5]].x, lm[eye[5]].y))
    B = math.dist((lm[eye[2]].x, lm[eye[2]].y), (lm[eye[4]].x, lm[eye[4]].y))
    C = math.dist((lm[eye[0]].x, lm[eye[0]].y), (lm[eye[3]].x, lm[eye[3]].y))
    return (A + B) / (2.0 * C)

# -------- MORSE HELPERS --------
def add_symbol(sym):
    global current_morse, last_symbol_time
    current_morse   += sym
    last_symbol_time = time.time()
    log(f"Symbol: {sym}  |  Buffer: [{current_morse}]")

def commit_letter():
    global current_morse, decoded_text
    if current_morse:
        letter = decode_morse(current_morse)
        decoded_text += letter
        log(f"LETTER: '{letter}'  |  Text: [{decoded_text}]")
        current_morse = ""

def commit_word():
    global decoded_text
    commit_letter()
    if decoded_text and not decoded_text.endswith(' '):
        decoded_text += ' '
        log(f"WORD GAP  |  Text: [{decoded_text.strip()}]")

# ================= DRAW HELPERS =================
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def draw_text_bg(frame, text, pos, scale, color, thickness=2, pad=6):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

def draw_log(frame, h, w):
    msgs = list(on_screen_log)
    for i, msg in enumerate(reversed(msgs)):
        alpha = 1.0 - i * 0.18
        c = int(200 * alpha)
        y = 18 + i * 16
        cv2.putText(frame, msg[:50], (w - 10 - len(msg[:50]) * 7, y),
                    FONT, 0.32, (c, c, c), 1, cv2.LINE_AA)

def draw_flash(frame, h, w, now):
    if now < flash_until:
        (tw, th), _ = cv2.getTextSize(flash_text, FONT_BOLD, 0.8, 2)
        x = (w - tw) // 2
        y = h // 2
        cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), flash_color, 2)
        cv2.putText(frame, flash_text, (x, y), FONT_BOLD, 0.8, flash_color, 2, cv2.LINE_AA)

# ================= PRINT HEADER =================
print("Controls: ESC = quit  |  M = switch mode  |  C = clear morse text")
print("-" * 60)

# ===================== MAIN LOOP =====================
while True:
    frame = grabber.get_frame()
    if frame is None:
        time.sleep(0.01)
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    # FPS
    fps      = 1.0 / max(now - fps_prev, 1e-6)
    fps_prev = now

    # MediaPipe on smaller frame for speed
    small  = cv2.resize(frame, (320, 240))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    # -------- BLINK-BASED MODE SELECTION SCREEN --------
    if MODE is None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "EYE ASSIST SYSTEM",
                    (w//2 - 80, h//2 - 65), FONT_BOLD,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 1-blink box (LEFT)
        cv2.rectangle(frame, (10, h//2 - 35), (w//2 - 8, h//2 + 35), (0, 255, 255), 1)
        cv2.putText(frame, "1 Blink",
                    (18, h//2 - 10), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Movement Mode",
                    (12, h//2 + 16), FONT, 0.38, (0, 220, 220), 1, cv2.LINE_AA)

        # 2-blink box (RIGHT)
        cv2.rectangle(frame, (w//2 + 8, h//2 - 35), (w - 10, h//2 + 35), (0, 255, 0), 1)
        cv2.putText(frame, "2 Blinks",
                    (w//2 + 16, h//2 - 10), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Morse Code Mode",
                    (w//2 + 12, h//2 + 16), FONT, 0.38, (0, 220, 0), 1, cv2.LINE_AA)

        # Blink counter indicator
        blink_indicator = f"Blinks: {sel_blink_count}"
        (tw, _), _ = cv2.getTextSize(blink_indicator, FONT_BOLD, 0.55, 1)
        cv2.putText(frame, blink_indicator, ((w - tw) // 2, h//2 + 58),
                    FONT_BOLD, 0.55, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, "ESC = quit",
                    (w//2 - 28, h//2 + 75), FONT, 0.32, (160, 160, 160), 1, cv2.LINE_AA)

        # Blink detection for mode selection
        if result.multi_face_landmarks:
            lm          = result.multi_face_landmarks[0].landmark
            sel_ear_val = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2.0

            if sel_ear_val < EAR_CLOSE and not sel_eye_closed:
                sel_eye_closed = True

            elif sel_ear_val > EAR_OPEN and sel_eye_closed:
                sel_eye_closed  = False
                sel_blink_count += 1
                sel_blink_time   = now
                print(f"\n  [SELECT] Blink #{sel_blink_count}", flush=True)

        # Decide mode after blink window expires
        if sel_blink_count > 0 and (now - sel_blink_time) > SEL_BLINK_WINDOW:
            if sel_blink_count == 1:
                MODE = "MOVEMENT"
                flash("MOVEMENT MODE", color=(0, 255, 255))
                log("Double blink -> INPUT mode")
                log("3 blinks -> HUNGRY | 5+ -> EMERGENCY")
                log("Look L x2 -> MEDICINES | R x2 -> WATER")
            elif sel_blink_count == 2:
                MODE = "MORSE"
                flash("MORSE CODE MODE", color=(0, 255, 0))
                log("Short blink=DOT  Long blink=DASH")
                log("Look L -> letter | Look R -> word")
                log("Press C to clear text")
            else:
                log(f"Unknown: {sel_blink_count} blinks — try again")
            # Reset selection state
            sel_blink_count = 0
            sel_eye_closed  = False

        # Terminal status
        if now - last_status_print > PRINT_INTERVAL:
            face_str = "Face:YES" if result.multi_face_landmarks else "Face:NO "
            print(f"\r[SELECT MODE]  {face_str}  FPS:{fps:5.1f}  Blinks:{sel_blink_count}  1=Movement  2=Morse  ", end="")
            last_status_print = now

        draw_flash(frame, h, w, now)
        cv2.imshow("Eye Assist System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        continue

    # -------- FACE LANDMARKS --------
    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        # Draw eye boxes — both eyes
        for eye_ids in (LEFT_EYE, RIGHT_EYE):
            xs = [int(lm[i].x * w) for i in eye_ids]
            ys = [int(lm[i].y * h) for i in eye_ids]
            cv2.rectangle(frame,
                          (min(xs) - 10, min(ys) - 10),
                          (max(xs) + 10, max(ys) + 10),
                          (0, 255, 0), 1)

        avg_ear = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2.0

        iris_x    = lm[RIGHT_IRIS[0]].x
        eye_left  = lm[362].x
        eye_right = lm[263].x
        ratio     = (iris_x - eye_left) / (eye_right - eye_left + 1e-6)

        # ==========================================
        #           MOVEMENT MODE
        # ==========================================
        if MODE == "MOVEMENT":

            if avg_ear < EAR_CLOSE and not eye_closed:
                eye_closed = True

            elif avg_ear > EAR_OPEN and eye_closed:
                eye_closed = False
                blink_count += 1
                last_event_time = now
                log(f"Blink #{blink_count}  |  State: {mv_state}")

            if mv_state == WAITING:
                if blink_count == 2:
                    mv_state    = INPUT
                    blink_count = 0
                    left_count  = 0
                    right_count = 0
                    last_event_time = now
                    flash("INPUT MODE", duration=1.0, color=(255, 255, 0))

            elif mv_state == INPUT:
                if not eye_closed and (now - last_gaze_time) > GAZE_COOLDOWN:
                    if ratio > RATIO_RIGHT:
                        right_count += 1
                        last_event_time = now
                        last_gaze_time  = now
                        log(f"Gaze RIGHT x{right_count}")
                    elif ratio < RATIO_LEFT:
                        left_count += 1
                        last_event_time = now
                        last_gaze_time  = now
                        log(f"Gaze LEFT x{left_count}")

                if (blink_count + left_count + right_count) > 0 and \
                   (now - last_event_time) > DECISION_GAP:

                    cmd = ""
                    if blink_count == 3:
                        cmd = "HUNGRY"
                    elif blink_count > 5:
                        cmd = "EMERGENCY"
                    elif blink_count == 2:
                        cmd = "WASHROOM"
                    elif left_count >= 2:
                        cmd = "MEDICINES"
                    elif right_count >= 2:
                        cmd = "WATER"

                    if cmd:
                        flash(cmd, duration=2.5, color=(0, 255, 255))
                    else:
                        log("No command matched")

                    mv_state    = WAITING
                    blink_count = 0
                    left_count  = 0
                    right_count = 0
                    eye_closed  = False

            # Bottom HUD bar
            cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 0), -1)
            draw_text_bg(frame, f"STATE: {mv_state}", (6, h - 35), 0.45, (255, 255, 0), 1, 3)
            draw_text_bg(frame, f"Blinks:{blink_count}  L:{left_count}  R:{right_count}",
                         (6, h - 12), 0.45, (0, 255, 0), 1, 3)

            # Terminal status
            if now - last_status_print > PRINT_INTERVAL:
                print(f"\r[MOVEMENT]  EAR:{avg_ear:.3f}  State:{mv_state:<7}  "
                      f"Blinks:{blink_count}  L:{left_count}  R:{right_count}  FPS:{fps:5.1f}  ", end="")
                last_status_print = now

        # ==========================================
        #           MORSE MODE
        # ==========================================
        elif MODE == "MORSE":

            if avg_ear < EAR_CLOSE and not eye_closed:
                eye_closed  = True
                close_start = now

            elif avg_ear > EAR_OPEN and eye_closed:
                eye_closed = False
                blink_dur  = now - close_start
                if blink_dur < DOT_DASH_LIMIT:
                    add_symbol('.')
                else:
                    add_symbol('-')

            if not eye_closed and (now - last_gaze_time) > GAZE_COOLDOWN:
                if ratio < RATIO_LEFT:
                    commit_letter()
                    last_gaze_time = now
                elif ratio > RATIO_RIGHT:
                    commit_word()
                    last_gaze_time = now

            # Auto decode on silence
            if current_morse and (now - last_symbol_time) > LETTER_GAP:
                commit_letter()
            if decoded_text and not decoded_text.endswith(' ') and \
               (now - last_symbol_time) > WORD_GAP:
                commit_word()

            # Blink hint — top centre
            if eye_closed:
                dur   = now - close_start
                hint  = "DASH ..." if dur >= DOT_DASH_LIMIT else "DOT ..."
                color = (0, 100, 255) if dur >= DOT_DASH_LIMIT else (0, 200, 255)
                (tw, _), _ = cv2.getTextSize(hint, FONT_BOLD, 0.7, 2)
                cv2.putText(frame, hint, ((w - tw) // 2, 30),
                            FONT_BOLD, 0.7, color, 2, cv2.LINE_AA)

            # Bottom HUD bar
            cv2.rectangle(frame, (0, h - 55), (w, h), (0, 0, 0), -1)
            draw_text_bg(frame, f"Buf:{current_morse}", (6, h - 35), 0.45, (0, 255, 255), 1, 3)
            draw_text_bg(frame, f"Txt:{decoded_text[-30:]}", (6, h - 12), 0.45, (0, 255, 0), 1, 3)

            # Terminal status
            if now - last_status_print > PRINT_INTERVAL:
                hint_str = ""
                if eye_closed:
                    dur = now - close_start
                    hint_str = f"[{'DASH' if dur >= DOT_DASH_LIMIT else 'DOT'} {dur:.2f}s]"
                print(f"\r[MORSE]  EAR:{avg_ear:.3f}  Buf:[{current_morse:<6}]  "
                      f"Text:[{decoded_text[-20:]:<20}]  FPS:{fps:5.1f}  {hint_str:<18}", end="")
                last_status_print = now

    else:
        # No face detected
        if now - last_status_print > PRINT_INTERVAL:
            print(f"\r[{MODE}]  NO FACE  FPS:{fps:5.1f}   ", end="")
            last_status_print = now

    # -------- TOP-LEFT: EAR + MODE --------
    ear_str = f"EAR:{avg_ear:.2f}" if result.multi_face_landmarks else "No face"
    draw_text_bg(frame, f"{ear_str}  {MODE}", (4, 16), 0.38,
                 (0, 0, 255) if eye_closed else (200, 200, 200), 1, 2)

    # -------- BOTTOM-RIGHT: switch hint --------
    cv2.putText(frame, "M=switch mode", (w - 105, h - 4),
                FONT, 0.32, (120, 120, 120), 1, cv2.LINE_AA)

    # -------- ROLLING LOG top-right --------
    draw_log(frame, h, w)

    # -------- FLASH BANNER --------
    draw_flash(frame, h, w, now)

    cv2.imshow("Eye Assist System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('m') and MODE is not None:
        print(f"\n\n>>> MODE RESET — look LEFT or RIGHT to select\n")
        MODE           = None
        eye_closed     = False
        blink_count    = 0
        left_count     = 0
        right_count    = 0
        mv_state       = WAITING
        current_morse  = ""
        decoded_text   = ""
        last_gaze_time = 0.0
        avg_ear        = 0.0
        flash("SELECT MODE", duration=1.5, color=(255, 200, 0))
        log("Mode reset")
    elif key == ord('c') and MODE == "MORSE":
        decoded_text  = ""
        current_morse = ""
        flash("TEXT CLEARED", duration=1.2, color=(200, 100, 255))

# ===================== CLEANUP =====================
grabber.stop()
cap.release()
cv2.destroyAllWindows()
print("\n\n[DONE]")
if decoded_text.strip():
    print(f"Final morse text: '{decoded_text.strip()}'")