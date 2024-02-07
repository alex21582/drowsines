import cv2
import time
from utils import VideoFrameHandler, RunningAverage, PersonChecker
from playsound import playsound
from pathlib import Path
import threading

# File and directory paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ALARM_FILE_PATH = str(ROOT.joinpath("audio").joinpath("zvuk-gorna.mp3"))
LOG_FILE = ROOT.joinpath("log.csv")
# Class initialization to calculate average FPS value
fps_average = RunningAverage()

# Threshold definition
eye_thresholds = {
    "THRESH": 0.15,
    "WAIT_TIME": 2
}
mouth_thresholds = {
    "THRESH": 0.8,
    "WAIT_TIME": 1.5
}

person_thresholds = {
    "WAIT_TIME": 3
}

# Configurations
eye_config = {'idxs': {"left": [362, 385, 387, 263, 373, 380],
                       "right": [33, 160, 158, 133, 153, 144]},
              "txt_pos": (10, 30),
              "alarm_str": "WAKE UP! WAKE UP",
              "close": True,
              "counter": True,
              "name": "eyes"}
mouth_config = {"idxs": {"mouth": [76, 37, 267, 306, 314, 84]},
                "txt_pos": (350, 30),
                "alarm_str": "Drowsiness detected!",
                "close": False,
                "counter": True,
                "name": "mouth"}

person_config = {"idxs": {"face": list(range(468))},
                "txt_pos": (200, 200),
                "alarm_str": "Person not detected!",
                "close": False,
                "counter": True,
                "name": "mouth"}

# Class initialization for fatigue control
mouth_handler = VideoFrameHandler(mouth_config)
eye_handler = VideoFrameHandler(eye_config)
person_handler = PersonChecker(person_config)
# Alarm status
shared_state = {"play_alarm": False}


# Alarm playback function
def play_siren():
    playsound(ALARM_FILE_PATH)


# Frame frequencies
def get_time(prev_frame_time):
    new_frame_time = time.time()
    fps_average.add_value(int(1 / (new_frame_time - prev_frame_time)))
    return new_frame_time


# Main video processing function
def process_video():
    frame_time = 0
    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_time = get_time(frame_time)
        # Video processing
        # frame, eye_alarm = person_handler.process(frame, person_thresholds)
        frame, mouth_alarm = mouth_handler.process(frame, mouth_thresholds)
        frame, eye_alarm = eye_handler.process(frame, eye_thresholds)
        if eye_alarm:  # The alarm will only be triggered by closing your eyes
            threading.Thread(target=play_siren).start()  # Playback of alarm in parallel flow

        cv2.imshow('Video', frame)
        # Press "q" to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            fps = fps_average.return_avg()
            # Save log
            eye_handler.write_file(fps, LOG_FILE)
            break

    # Camera window handler
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    threading.Thread(target=process_video).start()

