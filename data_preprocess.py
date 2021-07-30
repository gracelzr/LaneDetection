'''
Before you run:
Download and unzip the Tusimple dataset into the 'data' folder

Expected results:
2 folders, "data/image" and "data/classes"
"image" folder contains all raw images, in PNG type
"class" folder contains all ground truth images, uses 255 to represent lanes, else 0
'''

# import required packages
import json
import numpy as np
import cv2
import os
import glob
import pickle

def generateGroundTruths(labelList, dataFolder, imgFolder, gtFolder):

    '''
        There are 3 parameters in each ground truth
            raw_file: string type. the file path in the clip
            lanes: it is a list of list of lanes. Each list corresponds to a lane and each element of the inner list is x-coordinate of ground truth lane.
            h_samples: it is a list of height values corresponding to the lanes. Each element in this list is y-coordinate of ground truth lane
    '''


    for jsonFile in labelList:

        # read in lanes from json file
        json_data = [json.loads(line) for line in open(dataFolder + jsonFile)]

        for i in range(len(json_data)):  # you can change this to a smaller range e.g. range(0,10) for testing
            js_data = json_data[i]
            label_x = js_data['lanes']
            label_y = js_data['h_samples']
            raw_img = dataFolder + js_data['raw_file']  # 'raw_file': 'clips/0531/1492626287507231547/20.jpg' type <class 'str'>

            # parse out the directories
            img_name = raw_img.split('/')[-2] + '.png'  # get image folderName
            img_save_path = imgFolder + img_name  # /0531_1/data/images/1492626287507231547.jpg
            gt_save_path = gtFolder + img_name

            # read and write the original image to the save path
            img = cv2.imread(raw_img)
            cv2.imwrite(img_save_path, img)

            # parse out ground truths
            gt_lanes_vis = [[(x, y) for (x, y) in zip(lx, label_y) if x >= 0] for lx in label_x]
            mask = np.zeros_like(img)
            # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]]

            # draw the lines
            for j in range(len(gt_lanes_vis)):
                cv2.polylines(mask, np.int32([gt_lanes_vis[j]]), isClosed=False, color=[0, 255, 0], thickness=5)

            # read and write both masks to their respective paths
            cv2.imwrite(gt_save_path, mask[:,:,1])


def pickleDataset(imgFolder, gtFolder, pickledImgs, pickledGt):

    all_imgs = []
    all_gt = []

    RESIZE_IMAGE_HEIGHT = 80
    RESIZE_IMAGE_WIDTH = 160

    imgList = glob.glob(imgFolder + '*.png')
    gtList = glob.glob(gtFolder + '*.png')

    for raw_img in imgList:
        img = cv2.imread(raw_img, cv2.IMREAD_UNCHANGED)
        if img.shape != (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 3):
            img = cv2.resize(img,
                             dsize=(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
                             interpolation=cv2.INTER_LINEAR
                             )
            img = np.reshape(img, (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 3))
            img = np.array(img / 255.0, dtype=np.uint8)
        all_imgs.append(img)

    for raw_gt in gtList:
        gt = cv2.imread(raw_gt, cv2.IMREAD_UNCHANGED)
        if gt.shape != (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 1):
            gt = cv2.resize(gt,
                             dsize=(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
                             interpolation=cv2.INTER_NEAREST
                             )
            gt = np.reshape(gt, (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 1))
            gt = np.array(gt / 255.0, dtype=np.uint8)
        all_gt.append(gt)

    pickle.dump(np.array(all_imgs), open(pickledImgs, "wb"))
    pickle.dump(np.array(all_gt), open(pickledGt, "wb"))


if __name__ == '__main__':

    dataFolder = './data/'
    imgFolder = dataFolder + '/image/'
    gtFolder = dataFolder + '/classes/'

    os.makedirs(imgFolder, exist_ok=True)  # create folder
    os.makedirs(gtFolder, exist_ok=True)  # create folder

    # Add in the json files here that you want the model to train on
    jsonList = ['label_data_0531.json']
    generateGroundTruths(jsonList, dataFolder, imgFolder, gtFolder)

    pickledImgs = dataFolder + '/full_train.p'
    pickledGt = dataFolder + '/full_labels.p'
    pickleDataset(imgFolder, gtFolder, pickledImgs, pickledGt)
