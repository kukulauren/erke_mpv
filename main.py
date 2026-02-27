import streamlit as st
import tempfile
import os
from retail_analytics import process_video

# Page config
st.set_page_config(page_title="YOLO Video Detection", layout="wide")
st.title("I CAN SEE")

# Session state
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'predicted_path' not in st.session_state:
    st.session_state.predicted_path = None

left_col, right_col = st.columns(2)

# LEFT SIDE (INPUT PANEL)

with left_col:
    st.header("Upload & Detection Settings")
    st.write("Upload a video file (mp4), set confidence, and start detection.")

    # Confidence slider
    confidence = st.slider(
        "Set confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    st.write(f"Detection confidence set to: **{confidence}**")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video", type=["mp4"])

    # Control buttons
    # Control buttons (aligned layout)
    btn_col1, btn_col2, btn_col3 = st.columns([3, 4, 2])

    with btn_col1:
        start_btn = st.button("Start Detection")

    with btn_col3:
        cancel_btn = st.button("Cancel/Reset")

    # Handle cancel
    if cancel_btn:
        st.session_state.video_path = None
        st.session_state.predicted_path = None
        st.rerun()

    # Handle start detection
    if start_btn:
        if uploaded_file is None:
            st.warning("Please upload a video first!")
        else:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name

            st.success("Video uploaded successfully!")

            # Temporary output file
            predicted_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            predicted_path = predicted_tmp.name
            predicted_tmp.close()
            st.session_state.predicted_path = predicted_path

            # Run inference
            st.info("Running YOLO detection. This may take a while...")
            with st.spinner("Processing video..."):
                process_video(
                    model_path="best.pt",
                    video_path=st.session_state.video_path,
                    output_path=st.session_state.predicted_path,
                    conf_threshold=confidence
                )

            st.success("Detection complete!")

# RIGHT SIDE (OUTPUT PANEL)
with right_col:
    st.header("Predicted Video")

    if st.session_state.predicted_path and os.path.exists(st.session_state.predicted_path):
        st.video(st.session_state.predicted_path)

        with open(st.session_state.predicted_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
    else:
        st.info("Predicted video will appear here after detection.")