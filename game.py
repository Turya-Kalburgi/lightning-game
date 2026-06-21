import cv2
import math
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- sound ----------
SOUND_ON = True
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2)
    def make_zap():
        sr, dur = 22050, 0.18
        n = int(sr * dur); t = np.linspace(0, dur, n, False)
        freq = np.linspace(900, 200, n)
        wave = np.sign(np.sin(2 * np.pi * freq * t))
        noise = np.random.uniform(-1, 1, n) * 0.3
        env = np.exp(-t * 25)
        sig = (wave * 0.6 + noise) * env
        audio = np.int16(sig * 32767 * 0.4)
        stereo = np.ascontiguousarray(np.column_stack([audio, audio]))
        return pygame.sndarray.make_sound(stereo)
    ZAP = make_zap()
except Exception as e:
    print("Sound off:", e); SOUND_ON = False; ZAP = None
def play_zap():
    if SOUND_ON and ZAP: ZAP.play()

# ---------- detector ----------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2,
                                       running_mode=vision.RunningMode.IMAGE)
detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

P1_COLOR = (255, 120, 0)
P2_COLOR = (0, 120, 255)

BASE_LEN = 160      # blade length when still
MAX_LEN  = 650      # max length on a fast swing
REACH_K  = 6.0      # how much speed stretches the blade

def blade_points(start, end, phase, segments=20):
    sx, sy = start; ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1: return None
    px, py = -dy / length, dx / length
    pts = []
    for i in range(segments + 1):
        t = i / segments
        bx, by = sx + dx * t, sy + dy * t
        wave = math.sin(t * 5 + phase) * length * 0.08 * t
        jit = random.uniform(-2, 2) * t * length * 0.02
        off = wave + jit
        pts.append([int(bx + px * off), int(by + py * off)])
    return np.array(pts, np.int32).reshape((-1, 1, 2))

# ---------- state ----------
p1_health = p2_health = 100
p1_cooldown = p2_cooldown = 0
game_over, winner = False, ""
phase = 0.0
prev = {"L": None, "R": None}      # previous fingertip pos per side
speed = {"L": 0.0, "R": 0.0}       # smoothed swing speed per side

while True:
    ok, frame = cap.read()
    if not ok: print("Can't read from camera"); break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    mid_x = w // 2
    phase += 0.6
    for k in ("L", "R"): speed[k] *= 0.85      # blade retracts when slow
    if p1_cooldown > 0: p1_cooldown -= 1
    if p2_cooldown > 0: p2_cooldown -= 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    blades = []
    seen = {"L": False, "R": False}
    for hand in result.hand_landmarks:
        knuckle, tip, wrist = hand[5], hand[8], hand[0]
        kx, ky = int(knuckle.x * w), int(knuckle.y * h)
        tx, ty = int(tip.x * w), int(tip.y * h)
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        side = "L" if tx < mid_x else "R"
        seen[side] = True

        inst = math.hypot(tx - prev[side][0], ty - prev[side][1]) if prev[side] else 0
        speed[side] = max(speed[side], inst)        # spike on fast swings
        prev[side] = (tx, ty)

        dirx, diry = tx - kx, ty - ky
        dlen = math.hypot(dirx, diry) or 1
        ux, uy = dirx / dlen, diry / dlen
        blade_px = min(MAX_LEN, BASE_LEN + speed[side] * REACH_K)
        ex, ey = int(tx + ux * blade_px), int(ty + uy * blade_px)

        pts = blade_points((tx, ty), (ex, ey), phase)
        color = P1_COLOR if side == "L" else P2_COLOR
        blades.append({"side": side, "color": color, "pts": pts,
                       "end": (ex, ey), "wrist": (wx, wy)})
    for k in ("L", "R"):
        if not seen[k]: prev[k] = None

    p1 = next((b for b in blades if b["side"] == "L"), None)
    p2 = next((b for b in blades if b["side"] == "R"), None)

    # ---- hits ----
    flash_color = None
    if not game_over and p1 and p2:
        if math.hypot(p1["end"][0]-p2["wrist"][0], p1["end"][1]-p2["wrist"][1]) < 45 and p2_cooldown == 0:
            p2_health = max(0, p2_health-10); p2_cooldown = 18; play_zap(); flash_color = P2_COLOR
        if math.hypot(p2["end"][0]-p1["wrist"][0], p2["end"][1]-p1["wrist"][1]) < 45 and p1_cooldown == 0:
            p1_health = max(0, p1_health-10); p1_cooldown = 18; play_zap(); flash_color = P1_COLOR
        if p1_health == 0: game_over, winner = True, "PLAYER 2"
        elif p2_health == 0: game_over, winner = True, "PLAYER 1"

    # ---- glow pass (bloom) ----
    glow = np.zeros_like(frame)
    for b in blades:
        if b["pts"] is None: continue
        cv2.polylines(glow, [b["pts"]], False, b["color"], 22, cv2.LINE_AA)
        cv2.polylines(glow, [b["pts"]], False, b["color"], 12, cv2.LINE_AA)
        cv2.circle(glow, tuple(b["pts"][-1][0]), 16, b["color"], -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 9)
    frame = cv2.add(frame, glow)

    # ---- crisp cores + targets ----
    for b in blades:
        if b["pts"] is None: continue
        cv2.polylines(frame, [b["pts"]], False, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, tuple(b["pts"][-1][0]), 6, (255, 255, 255), -1)
        cd = (p1_cooldown if b["side"] == "L" else p2_cooldown)
        wx, wy = b["wrist"]
        if cd > 0:
            cv2.circle(frame, (wx, wy), 38, (255, 255, 255), -1)
        else:
            cv2.circle(frame, (wx, wy), 38, b["color"], 3)

    if flash_color:
        ov = frame.copy(); ov[:] = flash_color
        cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)

    # ---- UI panel ----
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.4, frame, 0.6, 0, frame)

    def bar(x, health, color, label, right=False):
        bw = 230
        cv2.rectangle(frame, (x, 22), (x+bw, 50), (50, 50, 50), -1)
        fill = int(bw * health / 100)
        if right: cv2.rectangle(frame, (x+bw-fill, 22), (x+bw, 50), color, -1)
        else:     cv2.rectangle(frame, (x, 22), (x+fill, 50), color, -1)
        cv2.rectangle(frame, (x, 22), (x+bw, 50), (255, 255, 255), 2)
        cv2.putText(frame, f"{label}", (x, 78), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
    bar(20, p1_health, P1_COLOR, "PLAYER 1")
    bar(w-250, p2_health, P2_COLOR, "PLAYER 2", right=True)
    cv2.putText(frame, "SWORD DUEL", (w//2-115, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)

    if game_over:
        ov = frame.copy(); ov[:] = (0, 0, 0)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        txt = winner + " WINS!"
        wc = P1_COLOR if winner == "PLAYER 1" else P2_COLOR
        cv2.putText(frame, txt, (w//2-230, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.8, wc, 8)
        cv2.putText(frame, txt, (w//2-230, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255,255,255), 2)
        cv2.putText(frame, "R = rematch    Q = quit", (w//2-180, h//2+55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (230,230,230), 1)

    cv2.imshow("Sword Game", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if key == ord("r") and game_over:
        p1_health = p2_health = 100; game_over, winner = False, ""

cap.release()
cv2.destroyAllWindows()
if SOUND_ON: pygame.mixer.quit()