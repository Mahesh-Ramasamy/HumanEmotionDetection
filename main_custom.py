import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import time
import psutil

from emotion import PostureEmotionDetector
from posture import PostureDetector

import threading


class MainClass:
    def __init__(self, webcam_index=1):
        # Initialize the webcam
        self.cap = cv2.VideoCapture(webcam_index)
        self.frame_count = 0
        self.start_time = time.time()
        self.avg_fps = 0
        # Get the current process id
        self.pid = os.getpid()

        # Create a Process object for the current process
        self.process = psutil.Process(self.pid)
        self.cpu_usage = 0
        self.cpu_thread = threading.Thread(target=self.monitor_cpu_usage, args=())
        self.cpu_thread.daemon = True  # Make the thread daemon so it exits when the main program exits
        self.cpu_thread.start()

        # Create the window with OpenCV (named window)
        cv2.namedWindow("Real-Time Posture Classification", cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
        cv2.resizeWindow("Real-Time Posture Classification", 1280, 720)  # Resize window to a specific size (640x480)

    def update_fps(self):
        """Update and calculate the average FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:  # Update every second
            self.avg_fps = self.frame_count / elapsed_time
            self.start_time = time.time()  # Reset time for the next calculation
            self.frame_count = 0  # Reset frame count

    def monitor_cpu_usage(self):
        #process = psutil.Process(self.pid)
        while True:
            # Get CPU usage of the specific process
            self.cpu_usage = self.process.cpu_percent(interval=1.0)  # Measure over 1 second
            #print(f"CPU usage of process : {cpu_usage}%")
            time.sleep(1)  # Sleep to avoid spamming output, can adjust interval

    def run(self):
        self.frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            # Start time for FPS calculation

            retValuePosture = posture_detector_.process_frame(frame.copy())
            retValueEmotion = emotion_detector_.process_frame(frame.copy())
            frame[0:30, :] = 255
            cv2.putText(frame, f'Posture: {retValuePosture} , {retValueEmotion}', (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.putText(frame, f'Emotion: {retValueEmotion}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Calculate FPS
            self.update_fps()
            #cpu_usage = self.get_cpu_usage()

            # Display average FPS on the frame
            cv2.putText(frame, f'FPS: {self.avg_fps:.2f}', (frame.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 1)
            cv2.putText(frame, f'CPU: {self.cpu_usage}%', (frame.shape[1] - 150, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 1)
            cv2.imshow("Real-Time Posture Classification", frame)
            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        # Release the webcam and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    posture_detector_ = PostureDetector(model_path="Posture_model.pkl")
    emotion_detector_ = PostureEmotionDetector(model_path="Emotion_model.pkl")
    mainProcess = MainClass()
    mainProcess.run()
    print('Process exited smoothly')