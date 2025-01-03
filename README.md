# PoseMatch 🧘‍♂️🎮

Welcome to **PoseMatch**, an exciting game that challenges your body alignment skills! Your task is to match your body pose with the reference pose displayed on the screen within a given time limit. The more accurately you match the pose, the higher your score!

## Game Features 🎮
- **Real-Time Pose Matching**: Your webcam captures your movements, and the game evaluates how closely they match a reference pose. 📸
- **Timer**: You have a limited time (e.g., 30 seconds) to match the pose! ⏱️
- **Pose Evaluation**: Your alignment is evaluated in real-time based on pose similarity, and a score is generated. 🏅
- **Engaging UI**: Interactive layout with the reference pose on one side and your live webcam feed on the other. 🖥️

## Gameplay Instructions 📜
1. **Start the Game**: Click on the "Start Game" button to begin your challenge. 🔘
2. **Match the Pose**: Align your body with the reference pose displayed on the left. 🧘‍♀️
3. **Timer**: The timer starts once you click the start button. You have a fixed time to match the pose. ⏳
4. **Score**: At the end of the time limit, your similarity score will be displayed. Higher accuracy means a higher score! 🎯

## How It Works 🔧
1. The game uses **MediaPipe** for pose estimation, which detects key body landmarks in real-time from your webcam feed. 💻
2. The reference pose is pre-loaded from an image, and the system calculates the similarity between your pose and the reference pose based on normalized landmarks. 🤖
3. The game evaluates pose similarity using **Euclidean Distance** and displays a similarity score. 📊

## Technical Details 🛠️
- **Pose Estimation**: Using **MediaPipe Pose** for real-time human pose detection.
- **Score Calculation**: Pose similarity is calculated by comparing the coordinates of body landmarks using Euclidean distance. A higher score indicates a better match! 🔍
- **Real-Time Display**: Streamlit is used for a dynamic UI, with updates every second to show the timer and similarity score. 🚀

## Future Implementations 🚀
- **Multi-User Mode**: Compete with friends to see who matches the pose better! 👯‍♀️
- **Pose Variety**: Add multiple reference poses for a more diverse challenge. 🏃‍♂️🧘‍♀️
- **Difficulty Levels**: Increase or decrease the time limit or reference pose complexity based on user preferences. ⚡
- **Feedback System**: Provide real-time feedback on which areas need improvement to match the pose better. 🗣️

## Technical Stack 🔧
- **Backend**: Python, OpenCV, MediaPipe, Streamlit
- **Pose Estimation**: MediaPipe Pose
- **UI Framework**: Streamlit for dynamic web app interface

## GitHub Repository 🔗
The **Pose Matching Score Calculation** logic is inspired by the [GitHub Repo](https://github.com/mattavallone/pose-matching/blob/master/CS6643%20Final%20Project.ipynb) and used for scoring. 🎯

## Future Plans 🔮
- Integration of **Deep Learning Models** to predict poses based on gesture recognition. 🤖
- **AR Integration**: Use Augmented Reality to superimpose the reference pose on your screen and give you real-time feedback on your movements. 🕶️
- **Mobile App Version**: Make the game accessible on mobile devices, bringing the challenge to a broader audience. 📱

## My 2nd Initiative 🌟
This project is part of my ongoing journey in **Computer Vision**. I am exploring how to improve pose estimation models and incorporating them into real-world applications. Stay tuned for more updates! 🧑‍💻

## Contact 📩
For more information or collaboration inquiries, feel free to reach out! 🚀

---

Thanks for playing **PoseMatch**! Have fun while learning about body alignment and improving your posture! 💪🎉
