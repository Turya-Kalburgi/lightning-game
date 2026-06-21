# Hand Gesture Games

Two webcam games you play with your hands. No controller, just your camera.

I built these to learn computer vision and have some fun with it.

## The games

**game.py** — sword fight. Your finger controls a glowing blade. Swing faster, it reaches further. Hit the other player to win.

**lightning.py** — charge an energy ball with your hands and fire a lightning bolt at your opponent, kind of like Goku. Two players, different colors.

## Run it

You need Python 3.12.

```
git clone https://github.com/Turya-Kalburgi/lightning-game.git
cd lightning-game
python3.12 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe numpy pygame
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Then:
```
python game.py
python lightning.py
```

Q to quit, R to rematch.

## Built with

Python, MediaPipe (hand tracking), OpenCV, NumPy, Pygame.

The model file and venv aren't in the repo, the steps above get you the model.
