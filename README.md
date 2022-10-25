# OffTheDip
## Surf Conditions Analysis Based on Computer Vision Algorithms
<pre><b>
            Surfers & Waves Detection                                    Surfers & Waves Tracker
</pre></b>
![detector1](https://user-images.githubusercontent.com/63463677/197405261-7e45d23e-e620-4593-b02f-2f28a2bded00.gif)  ![tracker1](https://user-images.githubusercontent.com/63463677/197405270-dbc57a2b-90c2-443d-94dc-554182450263.gif)


### Background
Today waves forecast services aren’t reliable. The lack of accurate wave forecast force surfers to use beach webcams in order to determine waves quality in different surfing spots.
Manually watching different streams of different webcams takes too much time and it’s not practical for surfers.

This project goal is characterizing surfing conditions in real time, based on existing beach webcams.


### Stage 1 – Surfers/Waves Detection
- Detection Framework - MMDetection
- Object Detection Architecture – Faster RCNN
- Backbone – Resnet
- Transfer learning – model pre-trained on COCO dataset
  - We've created a dataset based on beach webcams recordings
  - 3 classes – [ Standing, Sitting, Pocket ]
  - Dataset size: 600 frames

https://user-images.githubusercontent.com/63463677/197402898-5d777fc3-87bd-4d27-a7fa-5f4cad329066.mp4

### Stage 2 – Surfers/Waves Tracking
- SORT tracking algorithm
  - Based on Kalman Filter
  - Simple and fast, suitable for our basic needs
- Smoothing the tracker (avoid starting a new track when miss detecting few frames)
  - SORT hyperparameters tuning
  - Changing the internal implementation of SORT and adding our own more “aggressive” smoothing technique

https://user-images.githubusercontent.com/63463677/197403019-bfee6368-fc70-4056-b9af-d7af105d1011.mp4

### Stage 3 – Analysis
- Tracker object data history (per track)
- Dynamic stitching surfer track to wave track
- Display live analysis
### ![Live_Panel](https://user-images.githubusercontent.com/63463677/197404905-42fec22d-8f4f-4942-bd8b-298853215c3d.png)

- Export session analysis summarize
### ![Session_summarize](https://user-images.githubusercontent.com/63463677/197405007-14a821bd-04a2-44a8-aeb4-be84dc7104c3.png)


