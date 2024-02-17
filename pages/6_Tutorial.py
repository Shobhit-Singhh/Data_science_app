import streamlit as st
import os

def main():
    

    st.title("Analysis Tutorial in 7 Videos")

    video_folder = "videos"
    video_files = [
        "1.mp4",
        "2.mp4",
        "3.mp4",
        "4.mp4",
        "5.mp4",
        "6.mp4",
        "7.mp4",
    ]

    video_dis = [
        "Data Exploration",
        "Data Cleaning and Visualization",
        "Drop Missing Values",
        "Impute Missing Values and Outlier Detection",
        "Feature Encoding"
        "Feature Scaling and Transformation",
        "Model Training and Grid Search",
    ]
    st.sidebar.title("Navigation")
    selected_video = st.sidebar.selectbox("Select Video", [f"{i} - {video_dis[i-1]}" for i in range(1,8)])

    video_index = int(selected_video.split()[0])-1
    video_path = os.path.join(video_folder, video_files[video_index])
    st.video(video_path)

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

