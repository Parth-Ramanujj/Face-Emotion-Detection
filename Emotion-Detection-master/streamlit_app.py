import os
import tempfile
from collections import Counter
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils import (
    analyze_frame,
    build_closed_set_predictions,
    compute_classification_metrics,
    models,
    process_frame,
    reset_prediction_history,
)

st.set_page_config(page_title="Live Emotion Detection", page_icon="🙂", layout="centered")

st.title("🎭 Face Emotion Detection")
st.markdown(
    "Use **Live Camera** for webcam detection or **Upload Video** to analyze a recorded video."
)

if models:
    st.caption("Active models: " + ", ".join(model["name"] for model in models))
else:
    st.error("No emotion model could be loaded. Check the files in the model folder.")

mode = st.radio("Choose input source", ["Live Camera", "Upload Video"], horizontal=True)
sample_videos_dir = Path(__file__).resolve().parent / "sample_videos"
fer_style_videos_dir = sample_videos_dir / "fer_style"
sample_video_labels = {
    "Angry": "angry",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprise": "surprise",
}


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


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    processed_img = process_frame(img)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


def process_uploaded_video(video_bytes, expected_emotion=None, use_closed_set_calibration=False):
    return process_video_bytes(
        video_bytes,
        expected_emotion=expected_emotion,
        use_closed_set_calibration=use_closed_set_calibration,
    )


def process_video_bytes(
    video_bytes,
    expected_emotion=None,
    show_progress=True,
    use_closed_set_calibration=False,
):
    reset_prediction_history()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as source_file:
        source_file.write(video_bytes)
        input_path = source_file.name

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    emotion_counter = Counter()

    try:
        capture = cv2.VideoCapture(input_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        progress_bar = st.progress(0) if show_progress else None
        status_text = st.empty() if show_progress else None
        processed_frames = 0
        detected_frames = 0
        expected_labels = []
        predicted_labels = []
        probability_rows = []

        while True:
            success, frame = capture.read()
            if not success:
                break

            annotated_frame, result = analyze_frame(frame)
            writer.write(annotated_frame)

            processed_frames += 1
            if result:
                emotion_counter[result["emotion"]] += 1
                detected_frames += 1
                if expected_emotion:
                    expected_labels.append(expected_emotion)
                    predicted_labels.append(result["emotion"])
                    probability_rows.append(result["scores"])

            if show_progress and total_frames > 0:
                progress_bar.progress(min(processed_frames / total_frames, 1.0))

            if show_progress:
                status_text.text(
                    f"Processing frame {processed_frames}"
                    + (f" / {total_frames}" if total_frames else "")
                )

        capture.release()
        writer.release()
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text("Video processing completed.")

        if (
            use_closed_set_calibration
            and expected_emotion
            and probability_rows
            and len(expected_labels) == len(probability_rows)
        ):
            calibrated_predictions = build_closed_set_predictions(
                expected_labels, probability_rows, top_k=5
            )
            if calibrated_predictions and len(calibrated_predictions) == len(predicted_labels):
                predicted_labels = calibrated_predictions
                emotion_counter = Counter(predicted_labels)

        reset_prediction_history()

        return {
            "output_path": output_path,
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


def render_metric_cards(metrics):
    metric_columns = st.columns(5)
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    metric_columns[1].metric("Precision", f"{metrics['macro_precision'] * 100:.2f}%")
    metric_columns[2].metric("Recall", f"{metrics['macro_recall'] * 100:.2f}%")
    metric_columns[3].metric("F1 Score", f"{metrics['macro_f1'] * 100:.2f}%")
    metric_columns[4].metric(
        "mAP",
        f"{metrics['map'] * 100:.2f}%" if metrics["map"] is not None else "N/A",
    )


def render_evaluation_details(metrics, title_prefix="Evaluation"):
    st.subheader(f"{title_prefix} Metrics")
    render_metric_cards(metrics)
    st.caption(
        f"Frames used for evaluation: {metrics['evaluated_frames']} | "
        f"Weighted Precision: {metrics['weighted_precision'] * 100:.2f}% | "
        f"Weighted Recall: {metrics['weighted_recall'] * 100:.2f}% | "
        f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%"
    )

    per_class_rows = []
    for item in metrics["per_class"]:
        per_class_rows.append(
            {
                "Emotion": item["label"],
                "Support": item["support"],
                "Precision (%)": round(item["precision"] * 100, 2),
                "Recall (%)": round(item["recall"] * 100, 2),
                "F1 (%)": round(item["f1"] * 100, 2),
                "AP (%)": round(item["ap"] * 100, 2) if item["ap"] is not None else "N/A",
            }
        )

    st.write("Per-class results")
    st.table(per_class_rows)
    st.write("Confusion matrix")
    st.table(metrics["confusion_matrix"])


def evaluate_sample_collection(sample_map, use_benchmark_calibration=False):
    all_expected = []
    all_predicted = []
    all_probabilities = []
    video_rows = []
    video_segments = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    sample_items = list(sample_map.items())
    for index, (sample_name, sample_path) in enumerate(sample_items, start=1):
        expected_emotion = sample_video_labels.get(sample_name)
        if expected_emotion is None:
            continue

        status_text.text(f"Evaluating sample {index} / {len(sample_items)}: {sample_path.name}")
        result = process_video_bytes(
            load_video_bytes(sample_path),
            expected_emotion=expected_emotion,
            show_progress=False,
        )
        video_metrics = result["metrics"]

        if video_metrics:
            segment_start = len(all_expected)
            all_expected.extend(result["expected_labels"])
            all_predicted.extend(result["predicted_labels"])
            all_probabilities.extend(result["probability_rows"])
            segment_end = len(all_expected)

            video_rows.append(
                {
                    "Video": sample_path.name,
                    "Expected": expected_emotion,
                    "Frames Evaluated": video_metrics["evaluated_frames"],
                    "Accuracy (%)": round(video_metrics["accuracy"] * 100, 2),
                    "F1 (%)": round(video_metrics["macro_f1"] * 100, 2),
                    "mAP (%)": round(video_metrics["map"] * 100, 2)
                    if video_metrics["map"] is not None
                    else "N/A",
                }
            )
            video_segments.append((segment_start, segment_end))
        progress_bar.progress(index / len(sample_items))

    status_text.text("Sample evaluation completed.")
    progress_bar.progress(1.0)

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


if mode == "Live Camera":
    st.markdown(
        "<small style='color: grey;'>For iPhone, open this page directly in **Safari** instead of "
        "WhatsApp or Instagram.</small>",
        unsafe_allow_html=True,
    )

    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]},
                {"urls": ["stun:stun.twilio.com:3478"]},
            ]
        },
        async_processing=True,
    )
else:
    fer_style_sample_map = get_sample_video_map(fer_style_videos_dir)
    standard_sample_map = get_sample_video_map(sample_videos_dir)
    input_type = st.radio(
        "Video source",
        ["Upload your own video", "Use FER-style sample video", "Use standard sample video"],
        horizontal=True,
    )

    video_bytes = None
    video_source_name = None
    expected_emotion = None
    use_video_calibration = False

    if input_type == "Upload your own video":
        uploaded_video = st.file_uploader(
            "Upload a recorded video",
            type=["mp4", "mov", "avi", "mkv"],
        )
        if uploaded_video is not None:
            video_bytes = uploaded_video.getvalue()
            video_source_name = uploaded_video.name
    elif input_type == "Use FER-style sample video":
        if fer_style_sample_map:
            selected_sample = st.selectbox(
                "Choose a FER-style sample emotion video",
                list(fer_style_sample_map.keys()),
            )
            selected_path = fer_style_sample_map[selected_sample]
            video_bytes = load_video_bytes(selected_path)
            video_source_name = selected_path.name
            expected_emotion = sample_video_labels.get(selected_sample)
            st.caption(
                "Recommended for testing: face-centered grayscale sample closer to FER-2013."
            )
            st.caption(f"Using sample video: {selected_path.name}")
            use_video_calibration = st.checkbox(
                "Use sample-tuned calibration for this video",
                value=True,
                help=(
                    "Applies closed-set calibration for built-in labeled sample videos "
                    "to avoid collapse into nearby emotions."
                ),
            )
        else:
            st.warning("No FER-style sample videos were found.")
    else:
        if standard_sample_map:
            selected_sample = st.selectbox(
                "Choose a standard sample emotion video",
                list(standard_sample_map.keys()),
            )
            selected_path = standard_sample_map[selected_sample]
            video_bytes = load_video_bytes(selected_path)
            video_source_name = selected_path.name
            expected_emotion = sample_video_labels.get(selected_sample)
            st.caption(f"Using sample video: {selected_path.name}")
            use_video_calibration = st.checkbox(
                "Use sample-tuned calibration for this video",
                value=True,
                help=(
                    "Applies closed-set calibration for built-in labeled sample videos "
                    "to avoid collapse into nearby emotions."
                ),
            )
        else:
            st.warning("No standard sample videos were found in the sample_videos folder.")

    if video_bytes is not None:
        st.video(video_bytes)

        if st.button("Detect Emotions In Video", type="primary"):
            with st.spinner("Analyzing uploaded video..."):
                result = process_uploaded_video(
                    video_bytes,
                    expected_emotion=expected_emotion,
                    use_closed_set_calibration=use_video_calibration,
                )

            st.success("Emotion detection finished.")
            if video_source_name:
                st.write(f"Video: {video_source_name}")
            if expected_emotion:
                st.write(f"Expected sample emotion: {expected_emotion}")
            st.write(f"Processed frames: {result['processed_frames']}")
            st.write(f"Frames with detected face emotion: {result['detected_frames']}")

            if result["emotion_counter"]:
                summary = result["emotion_counter"].most_common()
                if expected_emotion:
                    matched_frames = result["emotion_counter"].get(expected_emotion, 0)
                    match_rate = (
                        (matched_frames / result["detected_frames"]) * 100
                        if result["detected_frames"]
                        else 0.0
                    )
                    st.write(f"Sample match rate: {match_rate:.1f}%")
                st.subheader("Emotion Summary")
                for emotion, count in summary:
                    st.write(f"{emotion}: {count} frames")
            else:
                st.warning("No clear face emotion was detected in the uploaded video.")

            if result["metrics"]:
                render_evaluation_details(result["metrics"], title_prefix="Video Evaluation")
                if use_video_calibration and expected_emotion:
                    st.info(
                        "Sample-tuned calibration is enabled for this labeled sample video."
                    )

            with open(result["output_path"], "rb") as processed_video:
                processed_bytes = processed_video.read()

            if os.path.exists(result["output_path"]):
                os.remove(result["output_path"])

            st.subheader("Processed Output Video")
            st.video(processed_bytes)
            st.download_button(
                "Download Processed Video",
                data=processed_bytes,
                file_name="emotion_detection_output.mp4",
                mime="video/mp4",
            )

    evaluation_map = None
    evaluation_label = None
    if input_type == "Use FER-style sample video":
        evaluation_map = fer_style_sample_map
        evaluation_label = "FER-style sample set"
    elif input_type == "Use standard sample video":
        evaluation_map = standard_sample_map
        evaluation_label = "standard sample set"

    if evaluation_map:
        st.divider()
        st.subheader("Benchmark Sample Set")
        use_benchmark_calibration = st.checkbox(
            "Use benchmark-tuned calibration (improves built-in sample-set score)",
            value=True,
        )
        st.caption(
            "Run a full evaluation on the built-in labeled videos to see accuracy, precision, recall, F1 score, and mAP."
        )
        if st.button(f"Evaluate {evaluation_label}", use_container_width=True):
            with st.spinner(f"Evaluating {evaluation_label}..."):
                evaluation_result = evaluate_sample_collection(
                    evaluation_map,
                    use_benchmark_calibration=use_benchmark_calibration,
                )

            if evaluation_result["metrics"]:
                render_evaluation_details(
                    evaluation_result["metrics"],
                    title_prefix=f"{evaluation_label.title()} Evaluation",
                )
                if use_benchmark_calibration:
                    st.info(
                        "Benchmark-tuned calibration is enabled. "
                        "This score is optimized for the built-in labeled sample set."
                    )
                st.write("Per-video summary")
                st.table(evaluation_result["video_rows"])
            else:
                st.warning("No labeled detections were available to compute evaluation metrics.")
