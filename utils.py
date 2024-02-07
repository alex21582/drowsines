import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
import csv
import os


def get_files_name(path):
    file_list = []
    if os.getcwd() != path:
        os.chdir(path)
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if ".avi" in name:
                file_list.append(os.path.join(root, name))
    return file_list


def get_mediapipe_app(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_coords(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        metric = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    except:
        metric = 0.0
        coords_points = None

    return metric, coords_points


def calculate_avg(landmarks, image_w, image_h, idxs_dict: dict):
    lm_coordinates = list()
    value_lst = list()
    for key in idxs_dict.keys():
        value, lm_coordinate = get_coords(landmarks, idxs_dict[key], image_w, image_h)
        lm_coordinates.append(lm_coordinate)
        value_lst.append(value)
    avg_value = np.mean(value_lst)
    return avg_value, tuple(lm_coordinates)


def plot_landmarks(frame, lm_coordinates, color, rot=False):
    for lm_coordinate in lm_coordinates:
        if lm_coordinate:
            for coord in lm_coordinate:
                cv2.circle(frame, coord, 2, color, -1)
    if rot:
        return cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class RunningAverage:
    def __init__(self):
        self.total = 0
        self.count = 0

    def add_value(self, new_value):
        self.total += new_value
        self.count += 1

    def return_avg(self):
        return self.total / self.count


class VideoFrameHandler:
    RED = (0, 0, 255)
    LIME = (0, 255, 0)
    OlIVE = (0, 128, 128)

    def __init__(self, config: dict):
        self.idxs = config['idxs']
        self.txt_pos = config['txt_pos']
        self.alarm_str = config['alarm_str']
        self.counter = config['counter']
        self.name = config['name']
        self.close = config['close']
        self.facemesh_model = get_mediapipe_app()
        self.metric = list()
        self.state_tracker = {"start_time": time.perf_counter(),
                              "CHECK_TIME": 0.0,
                              "COLOR": self.LIME,
                              "COUNTER": 0,
                              "play_alarm": False}

    def write_file(self, fps_avg, file_name='log.csv'):
        header = ['frame', 'metric']
        with open(file_name, 'w+', encoding='UTF8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            [writer.writerow([round(i / fps_avg, ndigits=4), self.metric[i]]) for i in range(len(self.metric))] # ???

    def process(self, frame: np.array, thresholds: dict):
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        time_txt_pos = (self.txt_pos[0], int(frame_h // 2 * 1.60))
        counter_txt_pos = (self.txt_pos[0], int(frame_h // 2 * 1.75))
        alarm_txt_pos = (self.txt_pos[0], int(frame_h // 2 * 1.90))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            value, coordinates = calculate_avg(landmarks, frame_w,  frame_h, self.idxs)
            self.metric.append(value)
            frame = plot_landmarks(frame, coordinates, self.state_tracker["COLOR"])

            if (value < thresholds["THRESH"]) == self.close:
                # Increase CHECK_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()

                self.state_tracker["CHECK_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["CHECK_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, self.alarm_str, alarm_txt_pos, self.state_tracker["COLOR"])
            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["CHECK_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.LIME
                if self.state_tracker["play_alarm"]:
                    self.state_tracker["COUNTER"] += 1
                self.state_tracker["play_alarm"] = False

            value_txt = f"{self.name}: {round(value, 2)}"
            time_txt = f"time: {round(self.state_tracker['CHECK_TIME'], 3)} sec"
            counter_txt = f'Counter {self.state_tracker["COUNTER"]}'
            plot_text(frame, value_txt, self.txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, time_txt, time_txt_pos, self.state_tracker["COLOR"])
            if self.counter:
                plot_text(frame, counter_txt, counter_txt_pos, self.OlIVE)

        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["CHECK_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.LIME
            if self.state_tracker["play_alarm"]:
                self.state_tracker["COUNTER"] += 1
            self.state_tracker["play_alarm"] = False

        return frame, self.state_tracker["play_alarm"]


class PersonChecker(VideoFrameHandler):
    def __init__(self, config: dict):
        super().__init__(config)

    def process(self, frame: np.array, thresholds: dict):
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        counter_txt_pos = (self.txt_pos[0], int(frame_h // 2 * 1.75))
        alarm_txt_pos = (self.txt_pos[0], int(frame_h // 2 * 1.90))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            _, coordinates = calculate_avg(landmarks, frame_w, frame_h, self.idxs)
            self.metric.append(True)
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["CHECK_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.LIME
            self.state_tracker["play_alarm"] = False
            frame = plot_landmarks(frame, coordinates, self.state_tracker["COLOR"])
        else:
            self.state_tracker["COLOR"] = self.RED
            self.metric.append(False)
            end_time = time.perf_counter()

            self.state_tracker["CHECK_TIME"] += end_time - self.state_tracker["start_time"]
            self.state_tracker["start_time"] = end_time

            if self.state_tracker["CHECK_TIME"] >= thresholds["WAIT_TIME"]:
                if not self.state_tracker["play_alarm"]:
                    self.state_tracker["COUNTER"] += 1
                self.state_tracker["play_alarm"] = True
                plot_text(frame, self.alarm_str, alarm_txt_pos, self.state_tracker["COLOR"])

        counter_txt = f'Counter {self.state_tracker["COUNTER"]}'
        if self.counter:
            plot_text(frame, counter_txt, counter_txt_pos, self.OlIVE)

        return frame, self.state_tracker["play_alarm"]