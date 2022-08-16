from mmdet.apis import set_random_seed
from mmcv import Config

cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# Modify dataset type and path
cfg.dataset_type = 'OffTheDipDataset'
cfg.data_root = 'my_project_files/data_training/'

cfg.data.test.type = 'OffTheDipDataset'
cfg.data.test.data_root = 'my_project_files/data_training/'
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = 'labeled_frames/video3'

cfg.data.train.type = 'OffTheDipDataset'
cfg.data.train.data_root = 'my_project_files/data_training/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'labeled_frames/video3'

cfg.data.val.type = 'OffTheDipDataset'
cfg.data.val.data_root = 'my_project_files/data_training/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'labeled_frames/video3'

# set model device
cfg.device = 'cuda'
#device='cuda:0'
#cfg.model.to(device)  # Convert the model to GPU
#cfg.model.eval()  # Convert the model into evaluation mode

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 2
# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './train_dump'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# try another config:
cfg.optimizer.type = 'SGD'
cfg.optimizer.lr = 0.02
cfg.optimizer.momentum = 0.9
cfg.optimizer.weight_decay = 0.0001
cfg.optimizer_config.grad_clip = None
cfg.lr_config.policy = 'step'
cfg.lr_config.warmup = 'linear'
cfg.lr_config.warmup_iters = 500
cfg.lr_config.warmup_ratio = 0.001
cfg.lr_config.step = [7]
#cfg.runner.max_epochs = 8
cfg.log_config.interval = 100

# try to randomly crop the photo because our objects are small
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(40, 24)),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# save the config as a .py file so we could use it later for inference
dump_file = 'train_dump/my_config.py'
cfg.dump(dump_file)