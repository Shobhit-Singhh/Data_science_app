import streamlit as st
import os

def main():
    

    st.title("Streamlit Tutorial: 7 Videos")

    video_folder = "videos"
    video_files = [
        "1.mp4",
        "2.mp4",
        "3-1.mp4",
        "3-2.mp4",
        "3-3.mp4",
        "3-4.mp4",
        "4.mp4",
    ]

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_video = st.sidebar.radio("Select Video", [f"Video {i}" for i in range(1, 7)])

    # Display selected video
    video_index = int(selected_video.split()[-1]) - 1
    video_path = os.path.join(video_folder, video_files[video_index])
    st.video(video_path)

    # Custom footer
    

if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Tutorial: 7 Videos",
        page_icon="📹",
        layout="wide"
    )
    st.sidebar.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f1f1f1;
                padding: 2 px;
                text-align: center;
                font-size: 14px;
                color: #555;
            }
        </style>
        <div class="footer">
            Data Science App Tutorial by Shobhit Singh, 
            <a class="linkedin" href="https://www.linkedin.com/in/shobhit-singhh/" target="_blank">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    main()
