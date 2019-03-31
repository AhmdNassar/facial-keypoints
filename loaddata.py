import numpy as np 
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
import matplotlib.pyplot as plt


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
        keypoints = self.csv_file.iloc[idx][1::].as_matrix().astype('float32').reshape(-1,2)
        # pytorch expect img as (channels, w , h) so we do that 
        #img = np.transpose(img,(2,0,1))
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
                new_h , new_w = h * (self.output_size/w) , self.output_size
            
            else:
                new_h , new_w =  self.output_size , w * (self.output_size/h)
            
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
        new_img = cv2.cvtColor(img,cv.COLOR_RGB2GRAY)
        


"""test = FacialKeyPointsDataset('./data/training','data/training_frames_keypoints.csv')
im , k = test[0]
print(im.shape)
test2 = Rescale((255,255))
img , k = test2((im,k))
print(len(test))

print(img.shape)"""