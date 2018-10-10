import os, sys
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform, img_as_ubyte
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

curr_video = 0


class ALOVDataset(Dataset):
    """ALOV Tracking Dataset
    Arguments:
        data_dir : Path to the data directory
        bounding_box_dir : Path to the directory where ground truths are stored
        transform : The kind of transformations to be applied to the data (example:- Normalize etc)
    """

    def __init__(self, data_dir, bounding_box_dir, transform=None):
        self.data_dir = data_dir
        self.bounding_box_dir = bounding_box_dir
        self.y = []
        self.x = []
        self.transform = transform
        envs = os.listdir(bounding_box_dir)

        for env in envs:
            env_videos = os.listdir(os.path.join(data_dir,env))

            for vid in env_videos:
                vid_src = os.path.join(self.data_dir ,env , vid)
                vid_annotations = os.path.join(self.bounding_box_dir , env , vid) + ".ann"
                frames = os.listdir(vid_src)
                frames.sort()

                frames = [vid_src + "/" + frame for frame in frames]
                f = open(vid_annotations, "r")
                annotations = f.readlines()
                f.close()
                frame_idxs = [int(ann.split(' ')[0]) - 1 for ann in annotations]

                frames = np.array(frames)

                for i in range(len(frame_idxs) - 1):
                    idx = frame_idxs[i]
                    next_idx = frame_idxs[i + 1]
                    self.x.append([frames[idx], frames[next_idx]])
                    self.y.append([annotations[i], annotations[i + 1]])

        self.len = len(self.y)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    # return size of dataset
    def __len__(self):
        return self.len

    # return transformed sample
    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        if (self.transform):
            sample = self.transform(sample)
        return sample

    # return sample without transformation for visualization purpose
    def get_sample(self, idx):
        prev = io.imread(self.x[idx][0])
        curr = io.imread(self.x[idx][1])
        prevbb = self.get_bb(self.y[idx][0])
        currbb = self.get_bb(self.y[idx][1])

        # Cropping the prev image with twice the size of  prev bounding box and scale the cropped image to (227,227,3)
        crop_curr = transforms.Compose([CropCurr(128)])
        scale = Rescale((227, 227))
        transform_prev = transforms.Compose([CropPrev(128), scale])
        prev_img = transform_prev({'image': prev, 'bb': prevbb})['image']

        # Cropping the current image with twice the size of  prev bounding box and scale the cropped image to (227,227,3)
        curr_obj = crop_curr({'image': curr, 'prevbb': prevbb, 'currbb': currbb})
        curr_obj = scale(curr_obj)
        curr_img = curr_obj['image']
        currbb = curr_obj['bb']
        currbb = np.array(currbb)
        sample = {'previmg': prev_img,
                  'currimg': curr_img,
                  'currbb': currbb
                  }
        return sample

    # returns original image and bounding box
    def get_orig_sample(self, idx, i=1):
        curr = io.imread(self.x[idx][i])
        currbb = self.get_bb(self.y[idx][i])
        sample = {'image': curr, 'bb': currbb}
        return sample

    # given annotation, returns bounding box in the format: (left, upper, width, height)
    def get_bb(self, ann):
        ann = ann.strip().split(' ')
        left = min(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        top = min(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        right = max(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        bottom = max(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        return [left, top, right, bottom]


# # helper function to display image at a particular index with ground truth bounding box
# # arguments: (idx, i)
# #            idx - index
# #             i - 0 for previous frame and 1 for current frame
# def show(self, idx, i):
# 	sample = self.get_orig_sample(idx, i)
# 	im = sample['image']
# 	bb = sample['bb']
# 	fig,ax = plt.subplots(1)
# 	ax.imshow(im)
# 	rect = patches.Rectangle((bb[0], bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
# 	ax.add_patch(rect)
# 	plt.show()

# # helper function to display sample, which is passed to neural net
# # display previous frame and current frame with bounding box
# def show_sample(self, idx):
# 	x = self.get_sample(idx)
# 	f, (ax1, ax2) = plt.subplots(1, 2)
# 	ax1.imshow(x['previmg'])
# 	ax2.imshow(x['currimg'])
# 	bb = x['currbb']
# 	rect = patches.Rectangle((bb[0], bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=1,edgecolor='r',facecolor='none')
# 	ax2.add_patch(rect)
# 	plt.show()


class Rescale(object):
    """Rescale image and bounding box.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        # print(image.shape, bb)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        img = img_as_ubyte(img)
        bb = [bb[0] * new_w / w, bb[1] * new_h / h, bb[2] * new_w / w, bb[3] * new_h / h]
        return {'image': img, 'bb': bb}


class CropPrev(object):
    """Crop the previous frame image using the bounding box specifications.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        global prev_bb
        image, bb = sample['image'], sample['bb']
        image = img_as_ubyte(image)
        if (len(image.shape) == 2):
            image = np.repeat(image[..., None], 3, axis=2)
        im = Image.fromarray(image)
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        left = bb[0] - w / 2
        top = bb[1] - h / 2
        right = left + 2 * w
        bottom = top + 2 * h
        box = (left, top, right, bottom)
        box = tuple([int(math.floor(x)) for x in box])
        res = np.array(im.crop(box))
        bb = [bb[0] - left, bb[1] - top, bb[2] - left, bb[3] - top]

        if (len(res.shape) <= 0):
            bb = prev_bb
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            left = bb[0] - w / 2
            top = bb[1] - h / 2
            right = left + 2 * w
            bottom = top + 2 * h
            box = (left, top, right, bottom)
            box = tuple([int(math.floor(x)) for x in box])
            res = np.array(im.crop(box))
            bb = [bb[0] - left, bb[1] - top, bb[2] - left, bb[3] - top]

        # print("Error:  bounding box degenerate")
        # sys.exit()
        else:
            prev_bb = bb
        return {'image': res, 'bb': bb}


class CropCurr(object):
    """Crop the current frame image using the bounding box specifications.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, prevbb, currbb = sample['image'], sample['prevbb'], sample['currbb']
        image = img_as_ubyte(image)
        if (len(image.shape) == 2):
            image = np.repeat(image[..., None], 3, axis=2)
        im = Image.fromarray(image)
        w = prevbb[2] - prevbb[0]
        h = prevbb[3] - prevbb[1]
        left = prevbb[0] - w / 2
        top = prevbb[1] - h / 2
        right = left + 2 * w
        bottom = top + 2 * h
        box = (left, top, right, bottom)
        box = tuple([int(math.floor(x)) for x in box])
        res = np.array(im.crop(box))
        bb = [currbb[0] - left, currbb[1] - top, currbb[2] - left, currbb[3] - top]
        return {'image': res, 'bb': bb}

# Imagenet video data
# https://www.kaggle.com/c/imagenet-object-detection-from-video-challenge/data

# alov = ALOVDataset('../alov/imagedata++/', '../alov/alov300++_rectangleAnnotation_full/')
# i = 400
# alov.show(i,0)
# alov.show(i,1)
# alov.show_sample(i)