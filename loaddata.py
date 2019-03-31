import numpy as np 
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms



class FacialKeyPointsDataset:
    """Face Landmarks dataset."""

    def __init__(self,images_dr,csv_file ,transform =None):
        """
        Args:
            images_dr : path to images folder 
            csv_file : path to csv file

        .......

        Retuen:
            tuple contain (image , keypoints)
        """
        self.images_dr = images_dr
        self.transform = transform
        self.csv_file = pd.read_csv(os.path.join(csv_file))
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self,idx):
        img_name = self.csv_file.iloc[idx][0]
        img = cv2.imread(os.path.join(self.images_dr,img_name))
        # convert image from BGR to RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        keypoints = self.csv_file.iloc[idx][1::].values.astype('float32').reshape(-1,2)
        sample = (img,keypoints)
        if(self.transform):
            sample = self.transform(sample)
        return sample       


#"""

class Rescale(object):
    """
    Rescale given sample (img , keypoints) to a given size.

    Args:
        output size : int or tuble,
                    if int : rescale smaller edge of image to given size keeping aspect ratio the same.
                    if tuple : returned image matched given size. 
    """

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        img , keypts = sample
        h , w = img.shape[:2]
        if(isinstance(self.output_size,int)):
            if h> w :
                new_h , new_w = h *self.output_size /w , self.output_size
            
            else:
                new_h , new_w =  self.output_size , w * self.output_size / h
            
        else:
            new_h , new_w = self.output_size 
        
        new_h , new_w = int(new_h) , int(new_w)

        img = cv2.resize(img,(new_w,new_h))

        # rescale keypoints too to work with new shape of image
        keypts = keypts * [new_w / w, new_h / h]
        
        return (img,keypts)


class Normalize:
    """ 
    convert given image to gray and normalize the color range to [0:1]
    and normalize it's kepoints too to be in range of [-1,1]
    """
    def __call__(self,sample):
        img , keypts = sample

        new_img = np.copy(img)
        new_keypts = np.copy(keypts)

        # convert image to gray 
        new_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        new_img = new_img/255.0

        #normalize keypoints too
        new_keypts = (keypts-100) / 50.0

        return (new_img,new_keypts)

class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))

        if(isinstance(output_size,int)):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self,sample):
        img , keypts = sample
        h , w = img.shape[0:2]
        new_h , new_w = self.output_size

        top  = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)

        img = img[top:top+new_h,left:left+new_w]
        keypts = keypts - [left,top]
        return (img,keypts)


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, keypts = sample
         
        # if image has no grayscale color channel, add one
        if(len(img.shape) == 2):
            # add that third color dim
            img = img.reshape(img.shape[0], img.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        
        return img , keypts