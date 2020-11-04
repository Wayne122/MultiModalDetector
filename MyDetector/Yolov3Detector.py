from __future__ import division

from MyDetector.Yolov3models import *
from MyDetector.utils_cus.utils import *
from MyDetector.utils_cus.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class Yolov3Detector(object):

    def __init__(self, args):

        parser = argparse.ArgumentParser()
        parser.add_argument('--config_path', type=str, default=args.config_path,
                            help='path to model config file')
        parser.add_argument('--weights_path', type=str, default=os.path.join(args.modelbasefolder, args.modelfilename), help='path to weights file')
        parser.add_argument('--class_path', type=str, default='data/kitti.names', help='path to class label file')
        parser.add_argument('--conf_thres', type=float, default=args.threshold, help='object confidence threshold')
        parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
        parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
        parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
        parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
        parser.add_argument('--showfig', type=bool, default=args.showfig, help='whether to show and save the result')
        parser.add_argument('--FULL_LABEL_CLASSES', type=list, default=args.FULL_LABEL_CLASSES, help='all classes list')
        self.opt = parser.parse_args()

    def detect(self, image):
        cuda = torch.cuda.is_available() and self.opt.use_cuda

        if self.opt.showfig:
            os.makedirs('output', exist_ok=True)

        # Set up model
        model = Darknet(self.opt.config_path, img_size=self.opt.img_size)
        model.load_weights(self.opt.weights_path)
        print('model path: ' +self.opt.weights_path)
        if cuda:
            model.cuda()
            print("using cuda model")

        model.eval() # Set in evaluation mode

        image = np.moveaxis(image, -1, 0)
        image = image[np.newaxis, ...]

        input_imgs = torch.from_numpy(image)

        classes = self.opt.FULL_LABEL_CLASSES # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        print ('\nPerforming object detection:')

        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            #print(detections)
            detections = non_max_suppression(detections, 80, self.opt.conf_thres, self.opt.nms_thres)
            #print(detections)

        # Bounding-box colors
        cmap = plt.get_cmap('tab20b')
        #cmap = plt.get_cmap('Vega20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        if self.opt.showfig:
            print ('\nSaving images:')

            # Create plot
            img = image
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            #kitti_img_size = 11*32
            kitti_img_size = 416
            # The amount of padding that was added
            #pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
            #pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
            pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))
            # Image height and width after padding is removed
            unpad_h = kitti_img_size - pad_y
            unpad_w = kitti_img_size - pad_x

            # Draw bounding boxes and labels of detections
            if detections is not None:
                print(type(detections))
                print(detections.size())
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                    # Rescale coordinates to original dimensions
                    box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                    box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]) )
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                            edgecolor=color,
                                            facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1-30, s=classes[int(cls_pred)]+' '+ str('%.4f'%cls_conf.item()), color='white', verticalalignment='top',
                            bbox={'color': color, 'pad': 0})

            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig('output.png', bbox_inches='tight', pad_inches=0.0)
            plt.close()
