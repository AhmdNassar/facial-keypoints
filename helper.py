import matplotlib.pyplot as plt
import torch
import numpy as np

def show_keypoints(img,keypts,gt=None,gray=False):
    # gt : right keypts and it's option
    if(gray):
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.scatter(keypts[:,0],keypts[:,1], s=30, marker='.', c='r')
    if gt is not None:
        plt.scatter(gt[:,0],gt[:,1], s=30, marker='.', c='g')


def visualize_output(images,keypts,batch_size,gt=None,gray=False):
    keypts = keypts.view(keypts.size(0),-1,2) # reshape outs to be [batch_sz , 68 , 2]
    fig = plt.figure(figsize=(10,20))
    for i in range(batch_size):
        img , key = images[i] , keypts[i]
        img = img.cpu().numpy()
        key = key.data.cpu().numpy()
        key = key *50 +100
        img = np.transpose(img,(1,2,0))
        ax = fig.add_subplot(batch_size/2,2,i+1)
        if gt is not None:
            img_gt = (gt[i]*50) + 100
            show_keypoints(np.squeeze(img),key,gt=img_gt,gray=True)
        else:
            show_keypoints(np.squeeze(img),key,gray=True)
        plt.axis('off')