"""
ENERGY DUEL v3 — easy-trigger edition.

WHAT CHANGED:
  - Ball charges with ONE or TWO hands (no more vanishing when hands touch).
  - Detector is more sensitive -> lights up easily, even with a busy background.
  - Video tracking mode -> ball stays stable instead of flickering.

HOW TO PLAY:
  - Left side = Player 1 (blue), right side = Player 2 (red).
  - Raise your hand(s) on your side and HOLD -> energy ball charges & grows.
  - When FULL it AUTO-FIRES a lightning bolt at your opponent.
  - First to drain the other to 0 wins.   R = rematch,  Q = quit.

RUN (from sword-game folder, with hand_landmarker.task present):
  python lightning.py
"""

import cv2
import math
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================= SOUND =======================
SOUND_ON = True
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2)

    def _tone(f_start, f_end, dur, kind="sine", vol=0.4, decay=12):
        sr = 22050
        n = int(sr * dur)
        t = np.linspace(0, dur, n, False)
        freq = np.linspace(f_start, f_end, n)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        if kind == "square":
            wave = np.sign(np.sin(phase))
        elif kind == "saw":
            wave = 2 * (phase / (2 * np.pi) % 1) - 1
        else:
            wave = np.sin(phase)
        noise = np.random.uniform(-1, 1, n)
        sig = (wave * 0.7 + noise * 0.3) * np.exp(-t * decay) * vol
        audio = np.int16(np.clip(sig, -1, 1) * 32767)
        stereo = np.ascontiguousarray(np.column_stack([audio, audio]))
        return pygame.sndarray.make_sound(stereo)

    FIRE_SND = _tone(320, 60, 0.45, "square", vol=0.5, decay=7)
    CHARGE_SND = _tone(200, 720, 0.5, "saw", vol=0.22, decay=2)
except Exception as e:
    print("Sound off:", e)
    SOUND_ON = False
    FIRE_SND = CHARGE_SND = None

def play(snd):
    if SOUND_ON and snd:
        snd.play()

# ======================= HAND DETECTOR (sensitive + video tracking) =======================
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=4,                          # set to 2 if it feels laggy in solo
    running_mode=vision.RunningMode.VIDEO,  # track across frames = stable
    min_hand_detection_confidence=0.3,    # lower = spots hands more eagerly
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# ======================= TUNING KNOBS =======================
P1_COLOR    = (255, 120, 0)    # blue (BGR) - left
P2_COLOR    = (0, 120, 255)    # red  (BGR) - right
CHARGE_RATE = 3.5     # charge speed per frame (bigger = fires sooner)
MAX_CHARGE  = 100     # auto-fires when charge reaches this
BALL_MIN    = 30      # starting ball radius (bigger = more visible instantly)
BEAM_LIFE   = 12
BEAM_THICK  = 60
DAMAGE      = 25
COOLDOWN    = 12

def palm_center(hand, w, h):
    xs = (hand[0].x + hand[5].x + hand[17].x) / 3
    ys = (hand[0].y + hand[5].y + hand[17].y) / 3
    return int(xs * w), int(ys * h)

def jagged(p0, p1, segments=16, chaos=22):
    x0, y0 = p0
    x1, y1 = p1
    pts = [(x0, y0)]
    for i in range(1, segments):
        t = i / segments
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t + random.uniform(-chaos, chaos)
        pts.append((int(x), int(y)))
    pts.append((x1, y1))
    return np.array(pts, np.int32).reshape((-1, 1, 2))

# ======================= STATE =======================
health = {"L": 100, "R": 100}
charge = {"L": 0.0, "R": 0.0}
cooldown = {"L": 0, "R": 0}
beams = []
game_over, winner = False, ""
frame_ts = 0      # timestamp for video mode (must keep increasing)

def fire(side, bx, by):
    beams.append({
        "side": side,
        "color": P1_COLOR if side == "L" else P2_COLOR,
        "x": bx, "y": by,
        "dir": 1 if side == "L" else -1,
        "power": charge[side], "life": BEAM_LIFE, "hit": False,
    })
    play(FIRE_SND)
    charge[side] = 0.0
    cooldown[side] = COOLDOWN

def reset():
    global game_over, winner
    health["L"] = health["R"] = 100
    charge["L"] = charge["R"] = 0.0
    cooldown["L"] = cooldown["R"] = 0
    beams.clear()
    game_over, winner = False, ""

# ======================= MAIN LOOP =======================
while True:
    ok, frame = cap.read()
    if not ok:
        print("Can't read from camera")
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    mid_x = w // 2

    for s in ("L", "R"):
        if cooldown[s] > 0:
            cooldown[s] -= 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 33
    result = detector.detect_for_video(mp_image, frame_ts)

    sides = {"L": [], "R": []}
    for hand in result.hand_landmarks:
        cx, cy = palm_center(hand, w, h)
        sides["L" if cx < mid_x else "R"].append((cx, cy))

    glow = np.zeros_like(frame)

    # ---- charge / fire (works with 1 OR 2 hands) ----
    for side in ("L", "R"):
        hands = sides[side]
        color = P1_COLOR if side == "L" else P2_COLOR

        if len(hands) >= 1 and cooldown[side] == 0:
            if len(hands) >= 2:
                (x1, y1), (x2, y2) = hands[0], hands[1]
                bx, by = (x1 + x2) // 2, (y1 + y2) // 2   # ball between both hands
            else:
                bx, by = hands[0]                          # ball at the one hand

            if charge[side] == 0:
                play(CHARGE_SND)
            charge[side] = min(MAX_CHARGE, charge[side] + CHARGE_RATE)

            r = int(BALL_MIN + charge[side] * 0.7)
            ready = charge[side] >= MAX_CHARGE
            cv2.circle(glow, (bx, by), r, (255, 255, 255) if ready else color, -1)
            cv2.circle(glow, (bx, by), int(r * 0.55), (255, 255, 255), -1)

            if ready:
                fire(side, bx, by)
        else:
            charge[side] = max(0.0, charge[side] - 1.5)

    # ---- bolts ----
    alive = []
    for b in beams:
        b["life"] -= 1
        thick = int(BEAM_THICK * (b["power"] / MAX_CHARGE) + 12)
        x_end = w if b["dir"] > 0 else 0
        bolt = jagged((b["x"], b["y"]), (x_end, b["y"]))
        cv2.polylines(glow, [bolt], False, b["color"], thick, cv2.LINE_AA)
        cv2.polylines(glow, [bolt], False, (255, 255, 255), max(3, thick // 4), cv2.LINE_AA)

        if not b["hit"] and not game_over:
            opp = "R" if b["side"] == "L" else "L"
            for (ox, oy) in sides[opp]:
                crosses = (ox > b["x"]) if b["dir"] > 0 else (ox < b["x"])
                if crosses and abs(oy - b["y"]) < thick:
                    health[opp] = max(0, health[opp] - DAMAGE)
                    b["hit"] = True
                    break
        if b["life"] > 0:
            alive.append(b)
    beams = alive

    # ---- bloom ----
    glow = cv2.GaussianBlur(glow, (0, 0), 13)
    frame = cv2.add(frame, glow)

    # ---- win check ----
    if not game_over:
        if health["L"] == 0:
            game_over, winner = True, "PLAYER 2"
        elif health["R"] == 0:
            game_over, winner = True, "PLAYER 1"

    # ---- UI ----
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.4, frame, 0.6, 0, frame)

    def meter(x, val, vmax, color, y, right=False, label=""):
        bw = 230
        cv2.rectangle(frame, (x, y), (x + bw, y + 22), (50, 50, 50), -1)
        fill = int(bw * val / vmax)
        if right:
            cv2.rectangle(frame, (x + bw - fill, y), (x + bw, y + 22), color, -1)
        else:
            cv2.rectangle(frame, (x, y), (x + fill, y + 22), color, -1)
        cv2.rectangle(frame, (x, y), (x + bw, y + 22), (255, 255, 255), 2)
        if label:
            cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

    meter(20, health["L"], 100, P1_COLOR, 30, label="PLAYER 1")
    meter(20, charge["L"], MAX_CHARGE, (255, 255, 255), 62)
    meter(w - 250, health["R"], 100, P2_COLOR, 30, right=True, label="PLAYER 2")
    meter(w - 250, charge["R"], MAX_CHARGE, (255, 255, 255), 62, right=True)

    cv2.putText(frame, "ENERGY DUEL", (w // 2 - 120, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "raise hand(s) to charge - auto-fires when full",
                (w // 2 - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if game_over:
        ov = frame.copy(); ov[:] = (0, 0, 0)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        wc = P1_COLOR if winner == "PLAYER 1" else P2_COLOR
        cv2.putText(frame, winner + " WINS!", (w // 2 - 240, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, wc, 8)
        cv2.putText(frame, winner + " WINS!", (w // 2 - 240, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 2)
        cv2.putText(frame, "R = rematch    Q = quit", (w // 2 - 180, h // 2 + 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (230, 230, 230), 1)

    cv2.line(frame, (mid_x, 100), (mid_x, h), (80, 80, 80), 1)
    cv2.imshow("Energy Duel", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r") and game_over:
        reset()

cap.release()
cv2.destroyAllWindows()
if SOUND_ON:
    pygame.mixer.quit()
