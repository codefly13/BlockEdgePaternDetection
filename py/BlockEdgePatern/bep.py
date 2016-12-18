import numpy as np
import cv2 as cv
import math
from math import log2

def calcShannonEnt(dataSet):
    length,dataDict=float(len(dataSet)),{}
    for data in dataSet:
        try: dataDict[data]+=1
        except:dataDict[data]=1
    return sum([-d/length*log2(d/length) for d in list(dataDict.values())])

def edge_measure(img, gamma):
    roi00 = img[0:4, 0:4]
    roi01 = img[0:4, 4:8]
    roi10 = img[4:8, 0:4]
    roi11 = img[4:8, 4:8]
    roi_list = [roi00, roi01, roi10, roi11]
    s = [np.mean(roi) for roi in roi_list]
    theta0 = abs((s[0] + s[1]) - (s[2] + s[3])) / 2
    theta90 = abs((s[0] + s[2]) - (s[1] + s[3])) / 2
    theta45 = max(abs(s[0] - (s[1] + s[2] + s[3])/3), abs(s[3] - (s[0] + s[1] + s[2])/3))
    theta135 = max(abs(s[1] - (s[0] + s[2] + s[3])/3), abs(s[2] - (s[0] + s[1] + s[3])/3))
    # thetaNE = gamma*abs((s[0] + s[3]) - (s[1]+s[2]))/2
    thetaNE = gamma
    # print("mean:", s)
    # print("theta:", [thetaNE, theta0, theta45, theta90, theta135])
    return [thetaNE, theta0, theta45, theta90, theta135]

def edge_measure_dct(img, gamma):
    # print(img, "\n\n")
    img = img.astype(np.float32)
    roi_dct = cv.dct(img)
    # print(roi_dct, "\n")
    theta0 = abs(roi_dct[1, 0]) / 4
    theta90 = abs(roi_dct[0, 1]) / 4
    theta45 = 1/6*max(abs(roi_dct[1, 0]+roi_dct[0, 1]+roi_dct[1, 1]), abs(roi_dct[1, 1]-roi_dct[0, 1]-roi_dct[1, 0]))
    theta135 = 1/6*max(abs(roi_dct[1, 0]-roi_dct[0, 1]-roi_dct[1, 1]), abs(roi_dct[0, 1]-roi_dct[1, 0]-roi_dct[1, 1]))
    # thetaNE = gamma*abs(roi_dct[1, 1])
    thetaNE = gamma

    # print([thetaNE, theta0, theta45, theta90, theta135])
    return [thetaNE, theta0, theta45, theta90, theta135]

def my_edge_measure(img, gamma):
    roi00 = img[0:4, 0:4]
    roi01 = img[0:4, 4:8]
    roi10 = img[4:8, 0:4]
    roi11 = img[4:8, 4:8]
    roi_list = [roi00, roi01, roi10, roi11]
    s = [np.mean(roi) for roi in roi_list]
    theta0 = abs((s[0] + s[1]) - (s[2] + s[3])) / 2
    theta90 = abs((s[0] + s[2]) - (s[1] + s[3])) / 2
    theta45 = max(abs(s[0] - (s[1] + s[2] + s[3])/3), abs(s[3] - (s[0] + s[1] + s[2])/3))
    theta135 = max(abs(s[1] - (s[0] + s[2] + s[3])/3), abs(s[2] - (s[0] + s[1] + s[3])/3))
    # thetaNE = gamma*abs((s[0] + s[3]) - (s[1]+s[2]))/2
    # thetaNE = gamma
    roi0 = img[2:6, 2:6]
    thetaNE = (1-abs(np.mean(roi0) - (s[0] + s[1] + s[2] + s[3]) / 4)/max(np.mean(roi0), (s[0] + s[1] + s[2] + s[3]) / 4))*gamma
    # print("mean:", s)
    # print("theta:", [thetaNE, theta0, theta45, theta90, theta135])
    return [thetaNE, theta0, theta45, theta90, theta135]

def my_edge_measure2(img, gamma):
    roi00 = img[0:4, 0:4]
    roi01 = img[0:4, 4:8]
    roi10 = img[4:8, 0:4]
    roi11 = img[4:8, 4:8]
    roi_list = [roi00, roi01, roi10, roi11]
    s = [np.mean(roi) for roi in roi_list]
    theta0 = abs((s[0] + s[1]) - (s[2] + s[3])) / 2
    theta90 = abs((s[0] + s[2]) - (s[1] + s[3])) / 2
    theta45 = max(abs(s[0] - (s[1] + s[2] + s[3])/3), abs(s[3] - (s[0] + s[1] + s[2])/3))
    theta135 = max(abs(s[1] - (s[0] + s[2] + s[3])/3), abs(s[2] - (s[0] + s[1] + s[3])/3))
    # thetaNE = gamma*abs((s[0] + s[3]) - (s[1]+s[2]))/2
    # thetaNE = gamma
    thetaNE = (1.5-1.0/(1+math.e**-(calcShannonEnt(img.reshape(-1)))))*gamma
    # print("mean:", s)
    # print("theta:", [thetaNE, theta0, theta45, theta90, theta135])
    return [thetaNE, theta0, theta45, theta90, theta135]

def my_edge_measure2_dct(img, gamma):
    # print(img, "\n\n")
    img = img.astype(np.float32)
    roi_dct = cv.dct(img)
    # print(roi_dct, "\n")
    theta0 = abs(roi_dct[1, 0]) / 4
    theta90 = abs(roi_dct[0, 1]) / 4
    theta45 = 1/6*max(abs(roi_dct[1, 0]+roi_dct[0, 1]+roi_dct[1, 1]), abs(roi_dct[1, 1]-roi_dct[0, 1]-roi_dct[1, 0]))
    theta135 = 1/6*max(abs(roi_dct[1, 0]-roi_dct[0, 1]-roi_dct[1, 1]), abs(roi_dct[0, 1]-roi_dct[1, 0]-roi_dct[1, 1]))

    thetaNE = (1.5-1.0/(1+math.e**-(calcShannonEnt(img.reshape(-1)))))*gamma

    # print([thetaNE, theta0, theta45, theta90, theta135])
    return [thetaNE, theta0, theta45, theta90, theta135]


edge_gray = 180
gamma = 25

NE = np.zeros((8, 8), dtype=np.uint8)
NE = 255 - NE

E0 = NE.copy()
for i in range(8):
    E0[3, i] = edge_gray
    E0[4, i] = edge_gray

E45 = NE.copy()
for i in range(8):
    if i == 0:
        E45[i, 8-i-2:8-i] = edge_gray
    elif i == 7:
        E45[i, 0:2] = edge_gray
    else:
        E45[i, 8-i-2:8-i+1] = edge_gray

E90 = NE.copy()
for i in range(8):
    E90[i, 3] = edge_gray
    E90[i, 4] = edge_gray

E135 = NE.copy()
for i in range(8):
    if i == 0:
        E135[i, i:i+2] = edge_gray
    elif i == 7:
        E135[i, 8-2:8] = edge_gray
    else:
        E135[i, i-1:i+2] = edge_gray

edge_patern = [NE, E0, E45, E90, E135]

cv.namedWindow("NE", flags=cv.WINDOW_NORMAL)
cv.namedWindow("E0", flags=cv.WINDOW_NORMAL)
cv.namedWindow("E45", flags=cv.WINDOW_NORMAL)
cv.namedWindow("E90", flags=cv.WINDOW_NORMAL)
cv.namedWindow("E135", flags=cv.WINDOW_NORMAL)
cv.imshow("NE", NE)
cv.imshow("E0", E0)
cv.imshow("E45", E45)
cv.imshow("E90", E90)
cv.imshow("E135", E135)


# im = cv.imread('Lena_512512.bmp')
im = cv.imread('baocai.jpg')
im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
cv.imshow("src", im)
im_bep = np.zeros(im.shape, dtype=np.uint8)
im_bep2 = np.zeros(im.shape, dtype=np.uint8)
im_bep3 = np.zeros(im.shape, dtype=np.uint8)

for i in range(im.shape[0]//8):
    for j in range(im.shape[1]//8):
        roi = im[i * 8:i * 8 + 8, j * 8:j * 8 + 8]
        # print(roi, '\n')
        edge_value = edge_measure(roi, gamma=20)
        # edge_value2 = edge_measure_dct(roi, gamma=18)
        edge_value3 = my_edge_measure2_dct(roi, gamma=45)
        # print(edge_value, '\n')
        # print(edge_value2, '\n')
        # print(edge_value3, '\n')
        # cv.imshow("eim", edge_patern[np.argmax(edge_value)])
        im_bep[i * 8:i * 8 + 8, j * 8:j * 8 + 8] = edge_patern[np.argmax(edge_value)]
        # im_bep2[i * 8:i * 8 + 8, j * 8:j * 8 + 8] = edge_patern[np.argmax(edge_value2)]
        im_bep3[i * 8:i * 8 + 8, j * 8:j * 8 + 8] = edge_patern[np.argmax(edge_value3)]
        # cv.imshow("bep", im_bep)
        # cv.imshow("bep2", im_bep2)
        # cv.waitKey(0)
cv.imshow("bep", im_bep)
# cv.imshow("bep2", im_bep2)
cv.imshow("bep3", im_bep3)
cv.waitKey(0)
