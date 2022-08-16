from train_on_my_dataset.my_inference_video_module import parse_video

parse_video(
    video='my_project_files/data_for_inference/3.mp4',
    architecture_config='configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py',
    checkpoints='my_project_files/checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth',
    result_path='my_project_files/data_for_inference/output/video_3_res.mp4',
    #show=True,  # uncomment to show video inference live
)