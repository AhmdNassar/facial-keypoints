import matplotlib.pyplot as plt

def show_keypoints(img,keypts):
    plt.imshow(img)
    plt.scatter(keypts[:,0],keypts[:,1], s=30, marker='.', c='r')