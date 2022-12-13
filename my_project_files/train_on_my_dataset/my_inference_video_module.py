import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector
from my_tracker_module import *


def parse_video(video, architecture_config, checkpoints, result_path='', show=False, wait_time=1.0, score_thr=0.3,
                device='cuda:0', number_of_frames=None, start_frame=0):

    assert result_path or show, ('Please specify at least one operation (save/show the video)')

    model = init_detector(architecture_config, checkpoints, device=device)
    tracks.reset_tracks()

    video_reader = mmcv.VideoReader(video)
    video_writer = None
    if result_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            result_path, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

        tracker_result_path = result_path[0:-7]+"tracker_res.mp4"
        tracker_video_writer = cv2.VideoWriter(
            tracker_result_path, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    panel_data = None
    for i, frame in enumerate(mmcv.track_iter_progress(video_reader)):
        if i < start_frame:
            continue
        if number_of_frames:
            if number_of_frames == i:
                break
        result = inference_detector(model, frame)
        panel_data = update_trackers(frame, result, tracker_video_writer)
        PALETTE = [
            (255, 0, 0),  # red
            (0, 255, 0),  # green
            (255, 255, 0)]  # yellow
        frame = model.show_result(frame, result, score_thr=score_thr, bbox_color=PALETTE, text_color=PALETTE)
        if show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', int(wait_time))
        if result_path:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
        tracker_video_writer.release()
    cv2.destroyAllWindows()

    summerize_session = "field,data\n"
    summerize_surfers = "surfer track_id,data\n"
    summerize_waves = "wave track_id,data\n"

    for key, val in panel_data.items():
        summerize_session += f"{key},{val}\n"
    f = open("my_project_files/data_for_inference/output/summerize_session.csv", "w")
    f.write(summerize_session)
    f.close()

    for key, val in tracks.standing_history.items():
        val = str(val).replace(",", " ").split("]]  ")[1].split(")")[0]
        summerize_surfers += f"{key},{val}\n"
    f = open("my_project_files/data_for_inference/output/summerize_surfers.csv", "w")
    f.write(summerize_surfers)
    f.close()

    for key, val in tracks.pocket_history.items():
        val = str(val).replace(",", " ").split("]]  ")[1].split(")")[0]
        summerize_waves += f"{key},{val}\n"
    f = open("my_project_files/data_for_inference/output/summerize_waves.csv", "w")
    f.write(summerize_waves)
    f.close()




