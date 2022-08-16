from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp
import mmcv

from modify_config_to_offTheDip_dataset import cfg
import offTheDipDataSet_load_LabelStudio_to_middle_format

# Build dataset
datasets = [build_dataset(cfg.data.train)]

print(datasets)
#print(cfg.device)

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience__len__ = {int} 1
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)