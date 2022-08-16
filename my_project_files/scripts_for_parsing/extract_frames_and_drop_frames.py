import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time


"""
this code counts the number of frames in a given video. later i calculate manually how many frames to drop, in this case i drop 1 out of 5 frames. and than i save each frame as 
an image.
i need to do this because from some reason i can't export frames from label studio, and from some reason the fps of label studio is lower the the fps cv2 extracts
"""

labeled_frames = [1, 5, 14, 18, 21, 25, 33, 37, 42, 47, 53, 58, 67, 79, 9, 23, 80, 90, 113, 134, 160, 179, 212, 223, 235, 244, 280, 287, 40, 41, 60, 74, 92, 105, 116, 117, 261, 84, 85, 87, 102, 196, 168, 172, 195, 228, 232, 240, 243, 17, 30, 38, 295, 309, 322, 335, 348, 361, 377, 384, 392, 402, 303, 310, 316, 410, 414, 426, 456, 476, 506, 518, 529, 538, 553, 597, 323, 330, 341, 358, 374, 422, 432, 442, 448, 455, 462, 485, 494, 502, 512, 564, 574, 582, 589, 608, 615, 546, 600, 621, 627, 653, 659, 668, 676, 682, 694, 710, 722, 728, 737, 747, 761, 769, 781, 633, 643, 649, 678, 697, 706, 634, 636, 629, 638, 795, 811, 824, 834, 838, 844, 857, 804, 914, 1025, 866, 977, 840, 845, 846, 853, 861, 868, 879, 892, 908, 929, 939, 948, 960, 1074, 1088, 1103, 1132, 1151, 901, 956, 997, 1017, 1035, 1046, 1064, 1052, 1056, 1058, 1068, 1170, 1174, 1180, 1210, 1226, 1243, 1281, 1183, 1184, 1193, 1202, 1217, 1232, 1233, 1265, 1275, 1288, 1295, 1271, 1313, 1324, 1343, 1369, 1370, 1383, 1392, 1334, 1355, 1397, 1413, 1427, 1431, 1446, 1454, 1462, 1474, 1482, 1506, 1523, 1530, 1495, 1537, 1551, 1567, 1587, 1618, 1670, 1692, 1706, 1717, 1733, 1743, 1765, 1773, 1783, 1791, 1800, 1813, 1822, 1833, 1843, 1879, 1469, 1488, 1636, 1728, 1729, 1738, 1742, 1756, 1437, 1445, 1458, 1464, 1517, 1536, 1547, 1558, 1577, 1598, 1655, 1681, 1748, 1854, 1921, 1867, 1887, 1871, 1899, 1926]

# Opens the Video file
cap = cv2.VideoCapture(os.path.join('3.mp4'))
frames = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    frames.append(frame)

print(len(frames))

i = 1
index = 1
for frame in frames:
    if i%5 != 0:
        cv2.imwrite("frames_from_video\\" + str(index) + '.jpg', frame)
        if index in labeled_frames:
            cv2.imwrite("only_labeld_frames\\" + str(index) + '.jpg', frame)
        index += 1
    i += 1

cap.release()
cv2.destroyAllWindows()




