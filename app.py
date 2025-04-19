import streamlit as st
import os
import time

# Set wide layout for better spacing
st.set_page_config(layout="wide")

st.title("🎬 Video Inference App")

# Upload section
st.markdown("## 📂 Upload Files")
npy_file = st.file_uploader("Upload your .npy file for inference", type=["npy"])
mp4_file = st.file_uploader("Upload your .mp4 video file", type=["mp4"])

if npy_file is not None and mp4_file is not None:
    # Save uploaded files
    npy_filename = npy_file.name
    mp4_filename = mp4_file.name

    with open(npy_filename, "wb") as f:
        f.write(npy_file.read())
    with open(mp4_filename, "wb") as f:
        f.write(mp4_file.read())

    st.success("✅ Files uploaded successfully! Starting inference...")

    # Run inference
    result = os.popen(f"python ./singleVideoEvalPlot.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce {npy_filename} {mp4_filename}").read()

    # Two column layout
    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.subheader("▶ Uploaded Video")
        st.video(mp4_filename)

    with col2:
        st.subheader("🧠 Inference Output")
        st.code(result, language="text")

        st.subheader("🖼 Timestamp Visualization")
        with st.spinner("Generating visualization... (10-15s)"):
            time.sleep(15)

        if os.path.exists("./tal_viz_output_one_jpg_v2/video_test_full_labeled.jpg"):
            st.image("./tal_viz_output_one_jpg_v2/video_test_full_labeled.jpg", caption="Model Timestamp Output", use_column_width=True)
        else:
            st.warning("⚠ Visualization image not found. Please check if the model generated it.")

else:
    st.info("📥 Please upload both a .npy file and a .mp4 video.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © All rights reserved by <strong>B I T W I S E M I N D S</strong> | 
        <a href='https://github.com/B-I-T-W-I-S-E-M-I-N-D-S' target='_blank' style='color: gray;'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)