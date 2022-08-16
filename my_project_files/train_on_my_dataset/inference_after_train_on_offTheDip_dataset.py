import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

from modify_config_to_offTheDip_dataset import cfg as config
from my_inference_video_module import parse_video

checkpoint = 'train_dump/latest.pth'  # Setup a checkpoint file to load
device='cuda:0'  # Set the device to be used for evaluation
config.model.pretrained = None  # Set pretrained to be None since we do not need pretrained model here
model = build_detector(config.model)  # Initialize the detector
checkpoint = load_checkpoint(model, checkpoint, map_location=device)  # Load checkpoint
model.CLASSES = checkpoint['meta']['CLASSES']  # Set the classes of models for inference
model.cfg = config  # We need to set the model's cfg for inference
model.to(device)  # Convert the model to GPU
model.eval()  # Convert the model into evaluation mode

# Use the detector to do inference
img = 'my_project_files/data_for_inference/1211.jpg'
result = inference_detector(model, img)

# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3)
model.show_result(img, result, out_file='my_project_files/data_for_inference/output/1211_res.jpg')

# Let's try inference a video
parse_video(
    video='my_project_files/data_for_inference/out_4_resized_video_from_after_labeling.mp4',
    #video='my_project_files/data_for_inference/out_4_resized_video.mp4',
    #video='my_project_files/data_for_inference/3.mp4',
    #video='my_project_files/data_for_inference/out_not_in_dataset_vid_resized_video.mp4',
    architecture_config='train_dump/my_config.py',
    checkpoints='train_dump/latest.pth',
    result_path='my_project_files/data_for_inference/output/video_4_after_labeling_res.mp4',
    show=True,  # uncomment to show video inference live
)