# Project Title
Posture and Emotion Recoginition 
## Description
This project is a Python-based library for Posture recognition and emotion detection using machine learning models. 
The system uses SVM techniques to analyze facial expressions from images and predict emotions in real-time.
#System Setup:

	Package                 Version
	----------------------- -----------
	mediapipe               0.10.21
	joblib                  1.4.2
	numpy                   1.26.4
	opencv-contrib-python   4.11.0.86
	opencv-python           4.11.0.86
	psutil                  7.0.0
	tensorflow              2.19.0
	tf_keras                2.19.0
	torch                   2.2.2
	torchvision             0.17.2
	scikit-learn            1.6.1
	scipy                   1.15.2

	** Download all the python files and model files in same directory
		emotion.py , posture.py , main_custom.py , Posture_model.pkl, Emotion_model.pkl
	** To run:
		python3 main_custom.py

# Model Choice & Pipeline
This project uses **MediaPipe** to extract pose points and face mesh points, which are essential for detecting human posture and facial expressions. **Support Vector Machines (SVM)** are employed to train and predict classes for both posture (human pose) and emotions based on the extracted points. 

This approach is specifically designed to run on CPU systems while optimizing for **higher FPS** (frames per second) to ensure real-time performance and responsiveness.

- **MediaPipe** is a cross-platform library that provides efficient solutions for various computer vision tasks, including pose estimation and face mesh detection.
- **SVM** (Support Vector Machine) is used as a classifier to categorize and predict posture and emotions from the extracted features.

Features
- **Pose Detection**: Extract human body pose landmarks (key points) using MediaPipe.
- **Face Mesh Detection**: Detect facial landmarks using MediaPipe’s face mesh model.
- **Emotion Classification**: Train an SVM model to classify emotions based on the extracted features from the face mesh and pose.
- **Posture Classification**: Use SVM to predict posture classes based on pose points.
- **Optimized for CPU**: This solution is optimized to run efficiently on CPU systems, maintaining high FPS for real-time performance.

# Performance Metrics
CPU Usage - 15 to 20%
Memory - 400mb to 500 mb
System info:
	Processor - Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz   2.11 GHz
	RAM - 8.00 GB (7.79 GB usable
	OS - Windows 
	
#Future Improvements
- **Use of DeepFace for Emotion Prediction**: One of the future improvements is to integrate **DeepFace**, an open-source library for facial recognition and emotion analysis. By combining the results from DeepFace and SVM, we can enhance the emotion classification accuracy.
- **Multi-frame Emotion Prediction**: Currently, the system predicts emotions from a single frame. A potential improvement would be to extend the system to analyze emotions over multiple frames (e.g., more than 5 frames) to provide more stable and reliable predictions for dynamic facial expressions.
- **CNN-based and Transformer-based Approaches**: In addition to SVM, we plan to experiment with **Convolutional Neural Networks (CNNs)** and **Transformer-based models** for emotion and posture detection. CNNs are well-suited for extracting spatial features from images, while Transformer models could be particularly useful for sequential data and capturing long-range dependencies in temporal video frames.

