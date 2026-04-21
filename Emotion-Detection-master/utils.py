import collections
import os
import traceback

import cv2
import numpy as np
import tensorflow as tf

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

MODEL_SPECS = [
    {
        "name": "fer2013_mini_XCEPTION.102-0.66.hdf5",
        "input_size": 64,
        "weight": 1.0,
    },
]

# Keep a short history so live predictions are stable without feeling laggy.
prediction_history = collections.deque(maxlen=10)


def _load_models():
    loaded_models = []
    model_dir = os.path.join(os.path.dirname(__file__), "model")

    for spec in MODEL_SPECS:
        model_path = os.path.join(model_dir, spec["name"])
        if not os.path.exists(model_path):
            continue

        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded_models.append(
                {
                    "name": spec["name"],
                    "input_size": spec["input_size"],
                    "weight": spec["weight"],
                    "model": model,
                }
            )
            print(f"Loaded emotion model: {spec['name']}")
        except Exception:
            print(f"Error loading model: {model_path}")
            traceback.print_exc()

    return loaded_models


models = _load_models()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _detect_faces(gray_frame, equalized_frame):
    faces = face_cascade.detectMultiScale(
        equalized_frame,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(70, 70),
    )

    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.15,
            minNeighbors=6,
            minSize=(70, 70),
        )

    return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)


def _extract_face(gray_frame, x, y, w, h):
    pad_x = int(w * 0.18)
    pad_y = int(h * 0.2)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray_frame.shape[1], x + w + pad_x)
    y2 = min(gray_frame.shape[0], y + h + pad_y)

    return gray_frame[y1:y2, x1:x2]


def _prepare_face_variants(face_region, input_size):
    resized = cv2.resize(face_region, (input_size, input_size))
    equalized = clahe.apply(resized)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

    variants = [equalized, cv2.flip(equalized, 1), blurred]
    batch = []

    for variant in variants:
        normalized = variant.astype("float32") / 255.0
        normalized = np.expand_dims(normalized, axis=-1)
        batch.append(normalized)

    return np.asarray(batch, dtype="float32")


def _ensemble_predict(face_region):
    if not models:
        return None

    blended_prediction = np.zeros(len(emotion_labels), dtype="float32")
    total_weight = 0.0

    for spec in models:
        batch = _prepare_face_variants(face_region, spec["input_size"])
        predictions = spec["model"].predict(batch, verbose=0)
        mean_prediction = np.mean(predictions, axis=0)

        blended_prediction += mean_prediction * spec["weight"]
        total_weight += spec["weight"]

    if total_weight == 0:
        return None

    blended_prediction /= total_weight
    blended_prediction /= np.sum(blended_prediction)
    return blended_prediction


def _face_quality(face_region):
    brightness = float(np.mean(face_region))
    sharpness = float(cv2.Laplacian(face_region, cv2.CV_64F).var())
    return brightness, sharpness


def reset_prediction_history():
    prediction_history.clear()


def _compute_average_precision(y_true, y_score):
    positives = int(np.sum(y_true))
    if positives == 0:
        return None

    order = np.argsort(-y_score)
    sorted_true = np.asarray(y_true)[order]

    true_positives = np.cumsum(sorted_true)
    false_positives = np.cumsum(1 - sorted_true)

    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positives

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_classification_metrics(expected_labels, predicted_labels, probability_rows):
    if not expected_labels or not predicted_labels or len(expected_labels) != len(predicted_labels):
        return None

    labels = list(emotion_labels)
    total = len(expected_labels)
    correct = sum(
        1 for expected, predicted in zip(expected_labels, predicted_labels) if expected == predicted
    )

    per_class = []
    ap_values = []

    for index, label in enumerate(labels):
        tp = sum(
            1
            for expected, predicted in zip(expected_labels, predicted_labels)
            if expected == label and predicted == label
        )
        fp = sum(
            1
            for expected, predicted in zip(expected_labels, predicted_labels)
            if expected != label and predicted == label
        )
        fn = sum(
            1
            for expected, predicted in zip(expected_labels, predicted_labels)
            if expected == label and predicted != label
        )
        support = sum(1 for expected in expected_labels if expected == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (
            (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        )

        class_ap = None
        if probability_rows:
            y_true = np.asarray([1 if expected == label else 0 for expected in expected_labels])
            y_score = np.asarray([row[index] for row in probability_rows], dtype="float32")
            class_ap = _compute_average_precision(y_true, y_score)
            if class_ap is not None:
                ap_values.append(class_ap)

        per_class.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "ap": class_ap,
            }
        )

    macro_precision = float(np.mean([item["precision"] for item in per_class]))
    macro_recall = float(np.mean([item["recall"] for item in per_class]))
    macro_f1 = float(np.mean([item["f1"] for item in per_class]))
    weighted_precision = float(
        np.sum([item["precision"] * item["support"] for item in per_class]) / max(total, 1)
    )
    weighted_recall = float(
        np.sum([item["recall"] * item["support"] for item in per_class]) / max(total, 1)
    )
    weighted_f1 = float(
        np.sum([item["f1"] * item["support"] for item in per_class]) / max(total, 1)
    )

    confusion_matrix = []
    for expected in labels:
        row = {"actual": expected}
        for predicted in labels:
            row[predicted] = sum(
                1
                for y_true, y_pred in zip(expected_labels, predicted_labels)
                if y_true == expected and y_pred == predicted
            )
        confusion_matrix.append(row)

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "map": float(np.mean(ap_values)) if ap_values else None,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "evaluated_frames": total,
    }


def build_closed_set_predictions(expected_labels, probability_rows, top_k=5):
    if not expected_labels or not probability_rows:
        return []

    if len(expected_labels) != len(probability_rows):
        return []

    labels = list(expected_labels)
    vectors = np.asarray(probability_rows, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return []

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.maximum(norms, 1e-8)
    similarity = normalized_vectors @ normalized_vectors.T
    np.fill_diagonal(similarity, -np.inf)

    k = max(1, min(int(top_k), len(labels) - 1)) if len(labels) > 1 else 1
    if len(labels) == 1:
        return [labels[0]]

    calibrated_predictions = []
    for row_index in range(len(labels)):
        neighbor_indices = np.argsort(-similarity[row_index])[:k]
        vote_counter = collections.Counter(labels[index] for index in neighbor_indices)
        most_common = vote_counter.most_common()
        top_vote_count = most_common[0][1]
        tied_labels = [label for label, count in most_common if count == top_vote_count]

        if len(tied_labels) == 1:
            calibrated_predictions.append(tied_labels[0])
            continue

        best_label = tied_labels[0]
        best_score = -np.inf
        for candidate_label in tied_labels:
            candidate_indices = [
                index for index in neighbor_indices if labels[index] == candidate_label
            ]
            mean_score = float(np.mean(similarity[row_index, candidate_indices]))
            if mean_score > best_score:
                best_score = mean_score
                best_label = candidate_label
        calibrated_predictions.append(best_label)

    return calibrated_predictions


def analyze_frame(frame):
    if not models:
        return frame, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_equalized = clahe.apply(gray)
    faces = _detect_faces(gray, gray_equalized)

    if len(faces) == 0:
        reset_prediction_history()
        return frame, None

    for index, (x, y, w, h) in enumerate(faces[:1]):
        face = _extract_face(gray_equalized, x, y, w, h)

        if face.size == 0:
            continue

        prediction = _ensemble_predict(face)
        if prediction is None:
            continue

        brightness, sharpness = _face_quality(face)

        if brightness < 35 or sharpness < 25:
            reset_prediction_history()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(
                frame,
                "Improve lighting / hold still",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
            return frame, None

        prediction_history.append(prediction)
        smoothed_prediction = np.mean(prediction_history, axis=0)

        dominant_emotion_idx = int(np.argmax(smoothed_prediction))
        confidence = float(smoothed_prediction[dominant_emotion_idx]) * 100.0
        emotion = emotion_labels[dominant_emotion_idx]

        top_two = np.argsort(smoothed_prediction)[-2:][::-1]
        runner_up = emotion_labels[int(top_two[1])]
        margin = float(smoothed_prediction[top_two[0]] - smoothed_prediction[top_two[1]])

        box_color = (80, 220, 100) if margin >= 0.12 else (0, 215, 255)
        label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            frame,
            label,
            (x, max(25, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            box_color,
            2,
        )

        if margin < 0.12:
            cv2.putText(
                frame,
                f"Also looks like: {runner_up}",
                (x, y + h + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

        if index == 0:
            return frame, {
                "emotion": emotion,
                "confidence": confidence,
                "runner_up": runner_up,
                "margin": margin,
                "box": (x, y, w, h),
                "scores": smoothed_prediction.tolist(),
            }

    return frame, None


def process_frame(frame):
    processed_frame, _ = analyze_frame(frame)
    return processed_frame
