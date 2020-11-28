######## to plot images ###########
# for i in range(6):
#     plt.subplot(2,3, i+1)
#     plt.imshow(x_train[i], cmap='gray')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

####### display stack ##############

def display_img_stack(images, stack_length = 8, stack = 'horizontal'):
    """
    images : numpy array of images : possible dims : (total_images, height, width)
    stack_length : No of images in a single horizontal/vertical stack
    stack : 'horizontal' or 'vertical'
    """
    remainder = images.shape[0] % stack_length
    images = images[: images.shape[0] - remainder + 1]
    print(images.shape)

    if images.ndim == 3:
        images_batches = images.reshape((images.shape[0]//stack_length, stack_length, images.shape[1], images.shape[2]))
        print(images_batches.shape)
        if stack == 'horizontal':
            images_stack = np.vstack((np.hstack(images_batch) for images_batch in images_batches))
            print(images_stack.shape)
        else:
            images_stack = np.hstack((np.vstack(images_batch) for images_batch in images_batches))
        plt.imshow(images_stack)
        plt.axis("off")
        plt.show()



