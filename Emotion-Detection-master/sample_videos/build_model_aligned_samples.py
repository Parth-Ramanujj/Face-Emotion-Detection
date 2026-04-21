from pathlib import Path

import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402


OUTPUT_DIR = ROOT / "model_aligned"
EMOTION_INDEX = {name: index for index, name in enumerate(utils.emotion_labels)}


def score_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_equalized = utils.clahe.apply(gray)
    faces = utils._detect_faces(gray, gray_equalized)
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = utils._extract_face(gray_equalized, x, y, w, h)
    if face.size == 0:
        return None, None

    prediction = utils._ensemble_predict(face)
    return prediction, frame


def build_video(source_path, target_path, target_emotion, target_frames=48):
    capture = cv2.VideoCapture(str(source_path))
    scored_frames = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        prediction, preserved_frame = score_frame(frame)
        if prediction is not None:
            score = float(prediction[EMOTION_INDEX[target_emotion]])
            scored_frames.append((score, frame_index, preserved_frame.copy()))
        frame_index += 1

    capture.release()

    if not scored_frames:
        return 0

    best_frames = sorted(scored_frames, key=lambda item: item[0], reverse=True)[:target_frames]
    best_frames.sort(key=lambda item: item[1])

    height, width = best_frames[0][2].shape[:2]
    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (width, height),
    )

    for _, _, frame in best_frames:
        writer.write(frame)

    writer.release()
    return len(best_frames)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for source_path in sorted(ROOT.glob("*.mp4")):
        target_emotion = source_path.stem.lower()
        target_path = OUTPUT_DIR / source_path.name
        frame_count = build_video(source_path, target_path, target_emotion)
        print(f"built {target_path.name} ({frame_count} selected frames)")


if __name__ == "__main__":
    main()
