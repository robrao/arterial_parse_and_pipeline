
import matplotlib

matplotlib.use('GTK3Agg')

import matplotlib.pyplot as plt

def display_img_with_mask(img, mask, filename):
    plt.figure(filename)
    plt.subplot(131)
    plt.imshow(mask, cmap='jet')
    plt.subplot(132)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.subplot(133)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.imshow(mask, cmap='jet', alpha=0.3)
    plt.show()

def display_img(img):
    plt.imshow(img, cmap=plt.cm.bone)
    plt.show()