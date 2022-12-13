import cv2
import numpy as np
import mmcv
from sort import sort
import dataclasses
from typing import List

@dataclasses.dataclass()
class Standing:
    locations: List[List[int]]
    first_frame_number: int
    last_frame_number: int
    avg_velocity: float  # [pixels/frames]
    max_height: int
    matched_pocket_id: int

@dataclasses.dataclass()
class Pocket:
    locations: List[List[int]]
    first_frame_number: int
    last_frame_number: int
    avg_velocity: float  # [pixels/frames]
    max_height: int
    matched_standing_id: int

class Tracks:
    def __init__(self):
        # key is track_id, value is Standing/Pocket object
        self.standing_history = {}
        self.pocket_history = {}
        self.frame_number = 0
        self.number_surfers_history = []

    def reset_tracks(self):
        self.standing_history = {}
        self.pocket_history = {}
        self.frame_number = 0
        self.number_surfers_history = []


standing_tracker = sort.Sort(max_age=15, min_hits=15, iou_threshold=0.1, factor=8)
pocket_tracker = sort.Sort(max_age=15, min_hits=15, iou_threshold=0.1, factor=4)

tracks = Tracks()

def draw_tracks(frame, standing_tracks, pocket_tracks):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for standing in standing_tracks:
        cv2.rectangle(frame, (int(standing[0]), int(standing[1])), (int(standing[2]), int(standing[3])),
                      color=(0, 255, 0), thickness=2)

        cv2.putText(frame, str(int(standing[4])), (int(standing[0]), int(standing[1])-8), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

       # draw track trace
        if int(standing[4]) in tracks.standing_history:
            for point in tracks.standing_history[int(standing[4])].locations:
                cv2.rectangle(frame, (point[0], point[1]), (point[0]+2, point[1]+2),
                              color=(0, 255, 0), thickness=2)

    for pocket in pocket_tracks:
        cv2.rectangle(frame, (int(pocket[0]), int(pocket[1])), (int(pocket[2]), int(pocket[3])),
                      color=(255, 255, 0), thickness=2)
        cv2.putText(frame, str(int(pocket[4])), (int(pocket[0]), int(pocket[1])-8), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

        # draw track trace
        """
        if int(pocket[4]) in tracks.pocket_history:
            for point in tracks.pocket_history[int(pocket[4])].locations:
                cv2.rectangle(frame, (point[0], point[1]), (point[0]+2, point[1]+2),
                              color=(255, 255, 0), thickness=2)
        """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def div_avoid_zero(n, d):
    return n / d if d else 0
def max_avoid_empty(l):
    if len(l) == 0:
        return 0
    return max(l)

def calc_data_for_panel():
    """
    surfed_waves, unsurfed_waves, avg_number_of_surfers, avg_height_wave_surfer_ratio,
    max_height_wave_surfer_ratio, avg_surfers_velocity, avg_wave_velocity
    """
    LEGIT_TRACK_NUMBER_OF_FRAMES = 30
    panel_data = {"surfed_waves": 0, "unsurfed_waves": 0}

    panel_data['avg_number_of_surfers'] = max_avoid_empty(tracks.number_surfers_history[-90:])  # take max in the last 90 frames
    wave_surfer_ratio = []
    avg_wave_velocity = []
    for pocket in tracks.pocket_history.values():
        if len(pocket.locations) >= LEGIT_TRACK_NUMBER_OF_FRAMES:
            if pocket.matched_standing_id != -1:
                panel_data['surfed_waves'] += 1
                wave_surfer_ratio.append(pocket.max_height / tracks.standing_history[pocket.matched_standing_id].max_height)
            else:
                panel_data['unsurfed_waves'] += 1

            avg_wave_velocity.append(pocket.avg_velocity)

    panel_data['avg_height_wave_surfer_ratio'] = div_avoid_zero(sum(wave_surfer_ratio), len(wave_surfer_ratio))
    panel_data['max_height_wave_surfer_ratio'] = max_avoid_empty(wave_surfer_ratio)
    panel_data['avg_wave_velocity[pixels/frames]'] = div_avoid_zero(sum(avg_wave_velocity), len(avg_wave_velocity))

    avg_surfers_velocity = []
    for standing in tracks.standing_history.values():
        if len(standing.locations) >= LEGIT_TRACK_NUMBER_OF_FRAMES:
            avg_surfers_velocity.append(standing.avg_velocity)

    panel_data['avg_surfers_velocity[pixels/frames]'] = div_avoid_zero(sum(avg_surfers_velocity), len(avg_surfers_velocity))

    return panel_data

def draw_panel(frame, panel_data):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_for_text = 15
    for key,val in panel_data.items():
        cv2.putText(frame, f"{key}: {val}", (5, y_for_text), font, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)
        y_for_text += 17

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def update_trackers(frame, detections, tracker_video_writer):
    tracks.frame_number += 1
    sitting_detections = np.array(detections[0])
    standing_detections = np.array(detections[1])
    pocket_detections = np.array(detections[2])
    standing_tracks = standing_tracker.update(standing_detections)
    pocket_tracks = pocket_tracker.update(pocket_detections)

    tracks.number_surfers_history.append(len(sitting_detections)+len(standing_detections))

    frame = draw_tracks(frame, standing_tracks, pocket_tracks)  # draw tracks on frame

    # update standing tracking history
    for standing in standing_tracks:
        trk_id = int(standing[4])
        BB_center_x = int((int(standing[0]) + int(standing[2])) / 2)
        BB_center_y = int((int(standing[1]) + int(standing[3])) / 2)
        BB_height = int(standing[3]) - int(standing[1])
        if trk_id in tracks.standing_history:
            tracks.standing_history[trk_id].locations.append([BB_center_x, BB_center_y])
            tracks.standing_history[trk_id].last_frame_number = tracks.frame_number
            #calc avg_velocity
            distance = (((BB_center_x - tracks.standing_history[trk_id].locations[0][0])**2 + (BB_center_y - tracks.standing_history[trk_id].locations[0][1])**2)**0.5)
            number_of_frames = (tracks.standing_history[trk_id].last_frame_number - tracks.standing_history[trk_id].first_frame_number)
            tracks.standing_history[trk_id].avg_velocity = distance/number_of_frames
            if BB_height > tracks.standing_history[trk_id].max_height:
                tracks.standing_history[trk_id].max_height = BB_height

        else:
            tracks.standing_history[trk_id] = Standing([[BB_center_x, BB_center_y]], tracks.frame_number, tracks.frame_number, 0, BB_height, -1)

    # update pocket tracking history
    for pocket in pocket_tracks:
        trk_id = int(pocket[4])
        BB_center_x = int((int(pocket[0]) + int(pocket[2])) / 2)
        BB_center_y = int((int(pocket[1]) + int(pocket[3])) / 2)
        BB_height = int(pocket[3]) - int(pocket[1])
        if trk_id in tracks.pocket_history:
            tracks.pocket_history[trk_id].locations.append([BB_center_x, BB_center_y])
            tracks.pocket_history[trk_id].last_frame_number = tracks.frame_number
            # calc avg_velocity
            distance = (((BB_center_x - tracks.pocket_history[trk_id].locations[0][0]) ** 2 + (
                        BB_center_y - tracks.pocket_history[trk_id].locations[0][1]) ** 2) ** 0.5)
            number_of_frames = (tracks.pocket_history[trk_id].last_frame_number - tracks.pocket_history[
                trk_id].first_frame_number)
            tracks.pocket_history[trk_id].avg_velocity = div_avoid_zero(distance, number_of_frames)
            if BB_height > tracks.pocket_history[trk_id].max_height:
                tracks.pocket_history[trk_id].max_height = BB_height

        else:
            tracks.pocket_history[trk_id] = Pocket([[BB_center_x, BB_center_y]], tracks.frame_number,
                                                       tracks.frame_number, 0, BB_height, -1)

    # match between standing to a pocket
    for standing_id, standing in tracks.standing_history.items():
        if standing.matched_pocket_id == -1:  # no match so far
            for pocket_id, pocket in tracks.pocket_history.items():
                # if this pocket active in the same times as this standing, match between the two
                if (pocket.first_frame_number <= standing.last_frame_number and standing.last_frame_number <= pocket.last_frame_number) or \
                        (pocket.first_frame_number <= standing.first_frame_number and standing.first_frame_number <= pocket.last_frame_number) or \
                        (standing.first_frame_number <= pocket.first_frame_number and pocket.last_frame_number <= standing.last_frame_number) or \
                        (pocket.first_frame_number <= standing.first_frame_number and standing.last_frame_number <= pocket.last_frame_number):
                    standing.matched_pocket_id = pocket_id
                    pocket.matched_standing_id = standing_id

    panel_data = calc_data_for_panel()

    frame = draw_panel(frame, panel_data)  # draw panel on frame

    tracker_video_writer.write(frame)

    return panel_data
