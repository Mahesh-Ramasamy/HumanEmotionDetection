import os
import time
import numpy as np
import cv2
import mediapipe as mp
#from sklearn.svm import OneClassSVM
import joblib  # For saving and loading the model


class PostureDetector:
    def __init__(self, model_path):
        # Initialize MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Load the posture SVM model
        self.svm_model = joblib.load(model_path)

        # Initialize the webcam
        #self.cap = cv2.VideoCapture(webcam_index)

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points (x1, y1) and (x2, y2)"""
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def calculate_distance_matrix(self, landmarks):
        """
        Calculate the distance matrix for all pose landmarks.

        Parameters:
        - landmarks: List of landmarks from MediaPipe Pose model.

        Returns:
        - distance_matrix: A matrix of Euclidean distances between all pose landmarks.
        """
        num_landmarks = len(mp.solutions.pose.PoseLandmark)
        distance_matrix = np.zeros((num_landmarks, num_landmarks))

        coords = [(landmark.x, landmark.y) for landmark in landmarks]

        for i in range(num_landmarks):
            for j in range(num_landmarks):
                if i != j:
                    dist = self.calculate_distance(coords[i], coords[j])
                    distance_matrix[i][j] = dist

        return distance_matrix

    def classify_posture_with_svm(self, landmarks):
        """Classify the posture using the trained SVM"""
        # Calculate the distance matrix for the current pose
        distance_matrix = self.calculate_distance_matrix(landmarks)

        # Flatten the distance matrix to a 1D array for SVM prediction
        flattened_matrix = distance_matrix.flatten()

        # Predict if the pose is normal or abnormal
        prediction = self.svm_model.predict([flattened_matrix])
        probabilities = self.svm_model.predict_proba([flattened_matrix])
        confidence_score = max(probabilities[0])

        # Return classification result based on prediction
        if confidence_score > 0.6:
            if prediction == 1:
                return "Upright"
            elif prediction == 2:
                return "Hunched"
        else:
            return "Unknown"

    def process_frame(self, frame):
        posture = ''
        # Flip the frame for a more natural view
        frame = cv2.flip(frame, 1)

        # Process the frame for pose detection
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Classify posture using the SVM model
            posture = self.classify_posture_with_svm(landmarks)

            # Display posture classification on the frame
            #cv2.putText(frame, f'Posture: {posture}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return posture

#posture_detector = PostureDetector(model_path="Posture_model.pkl")
