import streamlit as st
import os

st.title("🎬 Video Inference App")

# Upload .npy file
npy_file = st.file_uploader("📂 Upload your .npy file for inference", type=["npy"])

# Upload .mp4 file
mp4_file = st.file_uploader("🎥 Upload your .mp4 video file", type=["mp4"])

if npy_file is not None and mp4_file is not None:
    # Save files
    with open("testVideo.npy", "wb") as f:
        f.write(npy_file.read())

    with open("input_video.mp4", "wb") as f:
        f.write(mp4_file.read())

    st.success("Both files uploaded! Running inference...")

    # Run inference
    result = os.popen("python ./singleVideoEval2.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce testVideo.npy").read()
    # Display video
    st.subheader("▶ Uploaded Video")
    st.video("input_video.mp4")
    
    st.subheader("🧠 Inference Output")
    st.text(result)



else:
    st.info("Please upload both a .npy file and a .mp4 video.")