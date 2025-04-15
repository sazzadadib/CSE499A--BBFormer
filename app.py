import streamlit as st
import os

st.title("Video Inference App")

video_file = st.file_uploader("Upload your .npy video file", type=["npy"])

if video_file is not None:
    with open("testVideo.npy", "wb") as f:
        f.write(video_file.read())
    
    st.success("File uploaded. Running inference...")

    result = os.popen("python ./singleVideoEval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce testVideo.npy").read()

    st.text("Inference Output:")
    st.text(result)
