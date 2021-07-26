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
import os.path as ops
import random
import tensorflow as tf
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


#
# def generate_index_files(dataFolder, imgFolder, gtFolder):
#     index = []
#
#     for img in glob.glob(imgFolder + "/*.png"):
#         img_name = ops.split(img)[1]
#         binary_img =  gtFolder + img_name
#
#         index.append('{:s} {:s}\n'.format(
#             imgFolder + img_name,
#             binary_img)
#         )
#
#     random.shuffle(index)
#
#     length = len(index)
#     train_index = index[:int(length * 0.7)]
#     val_index = index[int(length*0.7):int(length*0.85)]
#     test_index = index[int(length*0.85):]
#
#     random.shuffle(train_index)
#     random.shuffle(val_index)
#     random.shuffle(test_index)
#
#     with open(dataFolder + 'train.txt', 'w') as file:
#         file.write(''.join(train_index))
#
#     with open(dataFolder + 'test.txt', 'w') as file:
#         file.write(''.join(test_index))
#
#     with open(dataFolder + 'val.txt', 'w') as file:
#         file.write(''.join(val_index))
#
#
#
#
# # read in the index files
# def read_index(path):
#     img_info = []
#     gt_info = []
#
#     with open(path, 'r') as file:
#         for line in file:
#             parsed_line = line.rstrip('\r').rstrip('\n').split(' ')
#             img_info.append(parsed_line[0])
#             gt_info.append(parsed_line[1])
#     return {'img_info': img_info, 'gt_info': gt_info}
#
#
# def generate_tfrecords(gt_images_paths, gt_binary_images_paths, tfrecords_path):
#
#     RESIZE_IMAGE_HEIGHT = 80
#     RESIZE_IMAGE_WIDTH = 160
#
#     _tfrecords_dir = ops.split(tfrecords_path)[0]
#     os.makedirs(_tfrecords_dir, exist_ok=True)
#
#     with tf.compat.v1.python_io.TFRecordWriter(tfrecords_path) as _writer:
#         for _index, _gt_image_path in enumerate(gt_images_paths):
#
#             # prepare gt image
#             _gt_image = cv2.imread(_gt_image_path, cv2.IMREAD_UNCHANGED)
#             if _gt_image.shape != (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 3):
#                 _gt_image = cv2.resize(
#                     _gt_image,
#                     dsize=(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
#                     interpolation=cv2.INTER_LINEAR
#                 )
#             _gt_image_raw = _gt_image.tostring()
#
#             # prepare gt binary image
#             _gt_binary_image = cv2.imread(gt_binary_images_paths[_index], cv2.IMREAD_UNCHANGED)
#             if _gt_binary_image.shape != (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 1):
#                 _gt_binary_image = cv2.resize(
#                     _gt_binary_image,
#                     dsize=(RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT),
#                     interpolation=cv2.INTER_NEAREST
#                 )
#                 _gt_binary_image = np.reshape(_gt_binary_image, (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT, 1))
#                 _gt_binary_image = np.array(_gt_binary_image / 255.0, dtype=np.uint8)
#             _gt_binary_image_raw = _gt_binary_image.tostring()
#
#             _example = tf.train.Example(
#                 features=tf.train.Features(
#                     feature={
#                         'gt_image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[_gt_image_raw])),
#                         'gt_binary_image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[_gt_binary_image_raw]))
#                     }))
#             _writer.write(_example.SerializeToString())
#
#     return

def pickleDataset(imgFolder, gtFolder):

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

    pickle.dump(np.array(all_imgs), open("./data/full_CNN_train_2.p", "wb" ))
    pickle.dump(np.array(all_gt), open("./data/full_CNN_labels_2.p", "wb"))



if __name__ == '__main__':

    dataFolder = './data/'
    imgFolder = dataFolder + '/image/'
    gtFolder = dataFolder + '/classes/'

    os.makedirs(imgFolder, exist_ok=True)  # create folder
    os.makedirs(gtFolder, exist_ok=True)  # create folder

    # TODO - add in the remaining jsons once we can train the model
    jsonList = ['label_data_0531.json']
    generateGroundTruths(jsonList, dataFolder, imgFolder, gtFolder)

    pickleDataset(imgFolder, gtFolder)



    '''
    This code below here generates the tfrecord files for train/test/val
    Since I switched over to use pickle for serialization, it's not necessary...
    '''
    # generate_index_files(dataFolder, imgFolder, gtFolder)
    #
    # tfRecordFolder = dataFolder + '/tfrecords/'
    # os.makedirs(tfRecordFolder, exist_ok=True)
    #
    # train_info = read_index(dataFolder + 'train.txt')
    # val_info = read_index(dataFolder + 'val.txt')
    # test_info = read_index(dataFolder + 'test.txt')
    #
    # generate_tfrecords(train_info['img_info'],
    #                    train_info['gt_info'],
    #                    tfRecordFolder + 'tusimple_train.tfrecords')
    #
    # generate_tfrecords(val_info['img_info'],
    #                    val_info['gt_info'],
    #                    tfRecordFolder + 'tusimple_val.tfrecords')
    #
    # generate_tfrecords(test_info['img_info'],
    #                    test_info['gt_info'],
    #                    tfRecordFolder + 'tusimple_test.tfrecords')
