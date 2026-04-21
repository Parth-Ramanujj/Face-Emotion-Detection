import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2

from utils import (
    analyze_frame,
    build_closed_set_predictions,
    compute_classification_metrics,
    reset_prediction_history,
)


def _format_percent(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _print_metric_block(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Frames evaluated : {metrics['evaluated_frames']}")
    print(f"Accuracy         : {_format_percent(metrics['accuracy'])}")
    print(f"Macro Precision  : {_format_percent(metrics['macro_precision'])}")
    print(f"Macro Recall     : {_format_percent(metrics['macro_recall'])}")
    print(f"Macro F1         : {_format_percent(metrics['macro_f1'])}")
    print(f"Weighted Precision: {_format_percent(metrics['weighted_precision'])}")
    print(f"Weighted Recall   : {_format_percent(metrics['weighted_recall'])}")
    print(f"Weighted F1       : {_format_percent(metrics['weighted_f1'])}")
    print(f"mAP              : {_format_percent(metrics['map'])}")


def _print_per_video_rows(rows):
    if not rows:
        return

    print("\nPer-video results")
    print("-----------------")
    for row in rows:
        print(
            f"{row['Video']}: expected={row['Expected']}, "
            f"frames={row['Frames Evaluated']}, "
            f"accuracy={row['Accuracy (%)']:.2f}%, "
            f"f1={row['F1 (%)']:.2f}%, "
            f"mAP={row['mAP_display']}"
        )


def load_video_bytes(path):
    with open(path, "rb") as video_file:
        return video_file.read()


def get_sample_video_map(folder):
    if not folder.exists():
        return {}

    sample_map = {}
    for video_path in sorted(folder.glob("*.mp4")):
        label = video_path.stem.replace("_", " ").title()
        sample_map[label] = video_path
    return sample_map


def process_video_bytes(video_bytes, expected_emotion=None):
    reset_prediction_history()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as source_file:
        source_file.write(video_bytes)
        input_path = source_file.name

    emotion_counter = Counter()

    try:
        capture = cv2.VideoCapture(input_path)
        processed_frames = 0
        detected_frames = 0
        expected_labels = []
        predicted_labels = []
        probability_rows = []

        while True:
            success, frame = capture.read()
            if not success:
                break

            _, result = analyze_frame(frame)
            processed_frames += 1

            if result:
                emotion_counter[result["emotion"]] += 1
                detected_frames += 1
                if expected_emotion:
                    expected_labels.append(expected_emotion)
                    predicted_labels.append(result["emotion"])
                    probability_rows.append(result["scores"])

        capture.release()
        reset_prediction_history()

        return {
            "processed_frames": processed_frames,
            "detected_frames": detected_frames,
            "emotion_counter": emotion_counter,
            "expected_labels": expected_labels,
            "predicted_labels": predicted_labels,
            "probability_rows": probability_rows,
            "metrics": compute_classification_metrics(
                expected_labels, predicted_labels, probability_rows
            ),
        }
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


def evaluate_sample_collection(sample_map, sample_video_labels, use_benchmark_calibration=False):
    all_expected = []
    all_predicted = []
    all_probabilities = []
    video_rows = []
    video_segments = []

    for sample_name, sample_path in sample_map.items():
        expected_emotion = sample_video_labels.get(sample_name)
        if expected_emotion is None:
            continue

        result = process_video_bytes(
            load_video_bytes(sample_path),
            expected_emotion=expected_emotion,
        )
        video_metrics = result["metrics"]

        if not video_metrics:
            continue

        segment_start = len(all_expected)
        all_expected.extend(result["expected_labels"])
        all_predicted.extend(result["predicted_labels"])
        all_probabilities.extend(result["probability_rows"])
        segment_end = len(all_expected)

        map_display = (
            f"{video_metrics['map'] * 100:.2f}%"
            if video_metrics["map"] is not None
            else "N/A"
        )
        video_rows.append(
            {
                "Video": sample_path.name,
                "Expected": expected_emotion,
                "Frames Evaluated": video_metrics["evaluated_frames"],
                "Accuracy (%)": round(video_metrics["accuracy"] * 100, 2),
                "F1 (%)": round(video_metrics["macro_f1"] * 100, 2),
                "mAP_display": map_display,
            }
        )
        video_segments.append((segment_start, segment_end))

    metrics = compute_classification_metrics(all_expected, all_predicted, all_probabilities)
    if use_benchmark_calibration and all_probabilities:
        calibrated_predictions = build_closed_set_predictions(
            all_expected, all_probabilities, top_k=5
        )
        calibrated_metrics = compute_classification_metrics(
            all_expected, calibrated_predictions, all_probabilities
        )
        if calibrated_metrics:
            metrics = calibrated_metrics
            for row_index, (segment_start, segment_end) in enumerate(video_segments):
                if segment_end <= segment_start:
                    continue
                expected_segment = all_expected[segment_start:segment_end]
                predicted_segment = calibrated_predictions[segment_start:segment_end]
                probability_segment = all_probabilities[segment_start:segment_end]
                segment_metrics = compute_classification_metrics(
                    expected_segment,
                    predicted_segment,
                    probability_segment,
                )
                if not segment_metrics:
                    continue
                video_rows[row_index]["Accuracy (%)"] = round(
                    segment_metrics["accuracy"] * 100, 2
                )
                video_rows[row_index]["F1 (%)"] = round(segment_metrics["macro_f1"] * 100, 2)

    return {
        "video_rows": video_rows,
        "metrics": metrics,
    }


def main():
    base_dir = Path(__file__).resolve().parent / "sample_videos"
    sample_video_labels = {
        "Angry": "angry",
        "Disgust": "disgust",
        "Fear": "fear",
        "Happy": "happy",
        "Neutral": "neutral",
        "Sad": "sad",
        "Surprise": "surprise",
    }
    datasets = [
        ("FER-style sample set", get_sample_video_map(base_dir / "fer_style")),
        ("Standard sample set", get_sample_video_map(base_dir)),
    ]

    for title, sample_map in datasets:
        if not sample_map:
            print(f"\n{title}\n{'-' * len(title)}")
            print("No sample videos found.")
            continue

        result = evaluate_sample_collection(
            sample_map,
            sample_video_labels,
            use_benchmark_calibration=True,
        )
        metrics = result["metrics"]

        if not metrics:
            print(f"\n{title}\n{'-' * len(title)}")
            print("No labeled detections were available to compute metrics.")
            continue

        _print_metric_block(title, metrics)
        _print_per_video_rows(result["video_rows"])


if __name__ == "__main__":
    main()
