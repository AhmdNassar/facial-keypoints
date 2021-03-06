import matplotlib.pyplot as plt
import torch
import numpy as np
import os

class StopEarly:
    """ Stop training early and save model if no good improvment """

    def __init__(self,model_dir = 'saved_models/',patience=10):
        """ 
            Args:
                patience : How long to wait after last time (training/test/val) loss improved .
                            Default: 7
                model_dir : dir to save model when we git best loss
                            Default: saved_models/
                
                Hint: we can use this class to stop model depend on test loss or validation loss,
                      by just pass val/test loss instead of train loss
        """
        self.best_loss = None
        self.patience = patience
        self.counter = 0
        self.model_dir = model_dir
        if(not os.path.exists(self.model_dir)):
            os.makedirs(model_dir)
    def __call__(self,model,loss):
        # model : your traning model
        # loss : current loss
        if self.best_loss is None :
            self.best_loss = loss
            return False
        
        else:
            
            if self.counter >= self.patience:
                return True
            if loss >= self.best_loss :
                self.counter += 1
                print(f"no improvement for {self.counter} / {self.patience}./n------------")
            else:
                self.counter = 0
                self.best_loss = loss
                # save model 
                print(f'We get new best loss: {self.best_loss} , saving model...')
                torch.save(model.state_dict(),self.model_dir+"model.pt")
                print('model saved./n------------')
            return False

        
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