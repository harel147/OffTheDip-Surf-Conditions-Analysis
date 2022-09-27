import copy
import json
import os

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

@DATASETS.register_module()
class OffTheDipDataset(CustomDataset):

    CLASSES = ('sitting', 'standing', 'pocket')
    #CLASSES = ('person',)

    studio_label_ann = ['my_project_files/data_training/labels/3_with_pocket.json',
                        'my_project_files/data_training/labels/4_1470_last_frame_with_pocket.json',
                        ]
    studio_label_prefix = ['',
                           '40000',
                           ]

    def load_annotations(self, ann_file):
        data_infos = []
        frames = []

        for pre, SL_ann_file in enumerate(self.studio_label_ann):
            with open(SL_ann_file) as f:
              d = json.load(f)

            results = d["result"]

            for res in results:
                seq = res["value"]["sequence"]
                if "labels" in res["value"]:
                    obj_label = res["value"]["labels"][0]
                else:  # in case there's no label from some reason
                    obj_label = 'sitting'
                for f in seq:
                    # label studio format are percentages of overall image dimension, need to translate to real dimentions.
                    pixel_x = f["x"] / 100.0 * IMAGE_WIDTH
                    pixel_y = f["y"] / 100.0 * IMAGE_HEIGHT
                    pixel_width = f["width"] / 100.0 * IMAGE_WIDTH
                    pixel_height = f["height"] / 100.0 * IMAGE_HEIGHT

                    if f'{self.studio_label_prefix[pre]}{f["frame"]}' not in frames:
                        frames.append(f'{self.studio_label_prefix[pre]}{f["frame"]}')
                        data_info = dict(filename=f'{self.studio_label_prefix[pre]}{f["frame"]}.jpg', width=1280, height=720, ann=dict(bboxes=[], labels=[], bboxes_ignore=[], labels_ignore=[]))
                        bboxes = [pixel_x, pixel_y, (pixel_x+pixel_width), (pixel_y+pixel_height)]
                        data_info["ann"]["bboxes"].append(bboxes)
                        if obj_label == 'sitting':
                            data_info["ann"]["labels"].append(0)
                        elif obj_label == 'standing':
                            data_info["ann"]["labels"].append(1)
                        else:  # pocket
                            data_info["ann"]["labels"].append(2)
                        data_infos.append(data_info)

                    else:
                        for d in data_infos:
                            if d["filename"] == f'{self.studio_label_prefix[pre]}{f["frame"]}.jpg':
                                bboxes = [pixel_x, pixel_y, (pixel_x+pixel_width), (pixel_y+pixel_height)]
                                d["ann"]["bboxes"].append(bboxes)
                                if obj_label == 'sitting':
                                    d["ann"]["labels"].append(0)
                                elif obj_label == 'standing':
                                    d["ann"]["labels"].append(1)
                                else:  # pocket
                                    d["ann"]["labels"].append(2)

        for d in data_infos:
            d["ann"]["bboxes"] = np.array(d["ann"]["bboxes"], dtype=np.float32).reshape(-1, 4)
            d["ann"]["labels"] = np.array(d["ann"]["labels"], dtype=np.long)
            d["ann"]["bboxes_ignore"] = np.array(d["ann"]["bboxes_ignore"], dtype=np.float32).reshape(-1, 4)
            d["ann"]["labels_ignore"] = np.array(d["ann"]["labels_ignore"], dtype=np.long)

        return data_infos
