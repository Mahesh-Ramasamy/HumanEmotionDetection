import os
import time
import numpy as np
import cv2
import mediapipe as mp
import joblib  # For saving and loading the model


class PostureEmotionDetector:
    def __init__(self, model_path):
        # Initialize MediaPipe FaceMesh model
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Load the emotion SVM model
        self.svm_model = joblib.load(model_path)

        # Initialize webcam
        #self.cap = cv2.VideoCapture(webcam_index)

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points (x1, y1) and (x2, y2)"""
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def get_nose_points(self, landmarks):
        """Extract nose tip and bridge points from landmarks"""
        nose_points = [
            (landmarks[1].x, landmarks[1].y),  # Nose tip (index 1)
            (landmarks[168].x, landmarks[168].y),  # Nose bridge point 1 (index 168)
            (landmarks[197].x, landmarks[197].y)  # Nose bridge point 2 (index 197)
        ]
        return nose_points

    def calculate_distances_from_nose(self, nose_points, landmarks):
        """
        Calculate the Euclidean distances from nose points to all other facial landmarks.
        """
        distances_from_nose = []
        all_landmarks = [(landmark.x, landmark.y) for landmark in landmarks]

        # For each nose point, calculate distance to all facial landmarks
        for nose_point in nose_points:
            distances = [self.calculate_distance(nose_point, landmark_coords) for landmark_coords in all_landmarks]
            distances_from_nose.append(distances)

        return distances_from_nose

    def classify_emotion_with_svm(self, flattened_matrix):
        """
        Classify emotion using the pre-trained SVM model based on the distance matrix.
        """
        prediction = self.svm_model.predict([flattened_matrix])
        probabilities = self.svm_model.predict_proba([flattened_matrix])
        confidence_score = max(probabilities[0])

        if confidence_score > 0.6:
            if prediction == 1:
                return "Happy"
            elif prediction == 2:
                return "Sad"
            elif prediction == 3:
                return "Angry"
        return "Unknown"

    def process_frame(self, frame):
        """
        Process each frame, detect face landmarks, calculate distances and classify emotion.
        """
        # Flip the frame for a more natural view
        frame = cv2.flip(frame, 1)
        emotion = ''
        # Process the frame for face mesh detection
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get nose points from the landmarks
                nose_points = self.get_nose_points(face_landmarks.landmark)
                distances_from_nose = np.array(self.calculate_distances_from_nose(nose_points, face_landmarks.landmark))

                # Flatten the distance array and classify emotion using SVM
                flattened_matrix = distances_from_nose.flatten()
                emotion = self.classify_emotion_with_svm(flattened_matrix)

                # Display emotion classification on the frame
                #cv2.putText(frame, f'Emotion: {emotion}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return emotion


#emotion_model_path = "Emotion_model.pkl"  # Adjust the model path as needed
#detector = PostureEmotionDetector(emotion_model_path)
