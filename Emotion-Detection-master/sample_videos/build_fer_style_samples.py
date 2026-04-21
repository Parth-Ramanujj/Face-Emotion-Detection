from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "fer_style"
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_primary_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(70, 70),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def crop_face(frame, face_box):
    x, y, w, h = face_box
    pad_x = int(w * 0.22)
    pad_y = int(h * 0.25)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    return frame[y1:y2, x1:x2]


def fer_style_frame(frame, previous_face=None):
    face_box = detect_primary_face(frame)
    if face_box is None:
        face_box = previous_face
    if face_box is None:
        return frame, previous_face

    face = crop_face(frame, face_box)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (256, 256))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    return face, face_box


def build_video(source_path, target_path):
    capture = cv2.VideoCapture(str(source_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0

    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (256, 256),
    )

    previous_face = None
    frame_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        output_frame, previous_face = fer_style_frame(frame, previous_face)
        writer.write(output_frame)
        frame_count += 1

    capture.release()
    writer.release()
    return frame_count


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for source_path in sorted(ROOT.glob("*.mp4")):
        target_path = OUTPUT_DIR / source_path.name
        frame_count = build_video(source_path, target_path)
        print(f"built {target_path.name} ({frame_count} frames)")


if __name__ == "__main__":
    main()
