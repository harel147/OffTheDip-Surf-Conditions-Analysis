import os
import numpy as np
import json

"""
....
"""

with open(os.path.join('1.json')) as f:
    d = json.load(f)

results = d["result"]

print(results[0]["value"]["sequence"][0]["frame"])

data_infos = []
frames = []

for res in results:
    seq = res["value"]["sequence"]
    for f in seq:
        if f["frame"] not in frames:
            frames.append(f["frame"])
            data_info = dict(filename=f'{f["frame"]}.jpg', width=1280, height=720, ann=dict(bboxes=[], labels=[], bboxes_ignore=[], labels_ignore=[]))
            bboxes = [f["x"], f["y"], (f["x"]+f["width"]), (f["y"]+f["height"])]
            data_info["ann"]["bboxes"].append(bboxes)
            data_info["ann"]["labels"].append(1)
            data_infos.append(data_info)

        else:
            for d in data_infos:
                if d["filename"] == f'{f["frame"]}.jpg':
                    bboxes = [f["x"], f["y"], (f["x"] + f["width"]), (f["y"] + f["height"])]
                    d["ann"]["bboxes"].append(bboxes)
                    d["ann"]["labels"].append(1)

for d in data_infos:
    d["ann"]["bboxes"] = np.array(d["ann"]["bboxes"])
    d["ann"]["labels"] = np.array(d["ann"]["labels"])
    d["ann"]["bboxes_ignore"] = np.array(d["ann"]["bboxes_ignore"])
    d["ann"]["labels_ignore"] = np.array(d["ann"]["labels_ignore"])


print(data_infos)
print(frames)
print(len(frames))







