
"""A variety of utility functions, especially for visualising the results."""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def import_image(image_path):
    """Import an image from the supplied path.
    Returns the image name and the image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_image(img, title=''):
    """Plot the supplied image and the corresponding title."""
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.title(title, fontsize=20)


def compare_images(img_org, img_undist, titles=None):
    """Display an image comparison in a subplot."""
    if titles is None:
        titles = ('Image Before', 'Image After')

    plt.subplots(1, 2, figsize=(15, 7), dpi=80)
    plt.subplot(1, 2, 1)
    plot_image(img_org, titles[0])
    plt.subplot(1, 2, 2)
    plot_image(img_undist, titles[1])


def overlay_image(img, overlay_img, opacity=1.0):
    """Reliably combine two images based on the opacity. Black pixels are
    treated as transparent. The method also takes in a grayscale base image.
    """
    if len(img.shape) < 3:
        img = np.stack((img, img, img), axis=-1)
    if len(overlay_img.shape) < 3:
        overlay_img = np.stack((overlay_img, overlay_img, overlay_img), axis=-1)

    img_out = np.zeros(img.shape)
    overlay_img = (overlay_img * opacity).astype(int)
    faded_img = (img * (1 - opacity)).astype(int)

    for i in range(0, img.shape[2]):
        channel_out = np.where(overlay_img[:, :, i] == 0,
                               img[:, :, i],
                               faded_img[:, :, i] + overlay_img[:, :, i])

        img_out[:, :, i] = np.where(channel_out > 255, 255, channel_out)

    return img_out.astype('uint8')

def quick_rectangle(img, corners, color='green', opacity=0.9,
                    thickness=4, filled=False):
    """Draws a rectangle on the input image."""
    colors = {'green': (30, 255, 120),
              'blue': (20, 104, 229),
              'red': (224, 52, 0),
              'orange': (252, 163, 9),
              'yellow': (252, 228, 10)}

    if color.lower() not in colors:
        color = 'green'
        print('Warning unknown color, using green instead.  Please choose from:\
              green, blue, red, orange and yellow.')
    else:
        color = color.lower()

    thickness = int(thickness)
    outline = np.zeros(img.shape)
    corners = tuple(map(tuple, corners))
    outline = cv2.rectangle(outline, corners[0], corners[1], colors[color],
                            thickness=thickness)

    if filled:
        fill = np.zeros(img.shape)
        fill = cv2.rectangle(fill, corners[0], corners[1], colors[color],
                             thickness=-1)
        img = overlay_image(img, fill, opacity=opacity*0.3)

    return overlay_image(img, outline, opacity=opacity)

def save_image(image, output_folder='./output_images/',
               name='Current_Image', colorspace='RGB'):
    """Save image to a file."""
    save_path = os.path.join(output_folder, name) + '.jpg'

    if colorspace != 'BGR':
        converter = getattr(cv2, "COLOR_" + colorspace + "2BGR")
        image = cv2.cvtColor(image, converter)

    cv2.imwrite(save_path, image)

def plot_text(img, string, placement):
    
    if placement == 'left':
        position = (30,img.shape[0] - 10)
    elif placement == 'right':
        position = (int(img.shape[1]/2 + 30),img.shape[0] - 10)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, string, position, font, 1.0,(255,255,255),2,cv2.LINE_AA)
    
    return img

def darken_bg(img, top, bottom, left, right):
    dark_img = np.copy(img)
    dark_img[top:bottom,left:right,:] = np.where(img[top:bottom,left:right,:] < 100, 0,
                                            img[top:bottom,left:right,:] - 100)
    return dark_img