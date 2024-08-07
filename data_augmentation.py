import os
import sys
import random
import cv2
import imutils
import filetype
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance


def help() -> None:
    """
    Displays usage of this program
    """
    print("help:\n\tAugmentation.py [path_to_img]")


def plot_images(img, f, r, c, b, s, p):
    """
    Displays every transformed image along with the original in a plot
    Arguments:
        img (np.ndarray): array representing the original image
        f, r, c, b, s, p (np.ndarray): arrays representing the augmented images
    """
    plt.figure(figsize=(8, 6))

    # Plot original image
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')

    # Plot flipped image
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    plt.title("Flip")

    # Plot rotated image
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    plt.title("Rotation")

    # Plot rotated image
    plt.subplot(3, 3, 4)
    plt.imshow(c)
    plt.title("Contrast")

    # Plot rotated image
    plt.subplot(3, 3, 5)
    plt.imshow(b)
    plt.title("Brightness")

    # Plot rotated image
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
    plt.title("Shear")
    # Plot rotated image
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
    plt.title("Projection")

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)

    # Display the plot
    plt.show()


def flip(img_path, img):
    """
    Flips the image along vertical axis
    Arguments:
        img_path (str): path to the image
        img (np.ndarray): array representing the image
    Returns:
        An array representing the flipped image
    """
    flipped_img = cv2.flip(img, 1)
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Flip.JPG", flipped_img)

    return flipped_img


def rotate(img_path, img):
    """
    Rotates the image randomly
    Arguments:
        img_path (str): path to the image
        img (np.ndarray): array representing the image
    Returns:
        An array representing the rotated image
    """
    rotated_img = imutils.rotate_bound(img, random.choice([-30, -20, -10,
                                                           10, 20, 30]))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Rotate.JPG", rotated_img)
    return rotated_img


def contrast(img_path, img):
    """
    Adjusts the image's contrast
    Arguments:
        img_path (str): path to the image
        img (np.ndarray): array representing the image
    Returns:
        An array representing the image with adjusted contrast
    """
    # Open the image
    image = Image.open(img_path)

    # Create an enhancer object for brightness
    enhancer = ImageEnhance.Contrast(image)

    contrast_factor = 1.5
    # Adjust the brightness
    enhanced_image = enhancer.enhance(contrast_factor)

    # Save the modified image
    enhanced_image.save(img_path[0:len(img_path) - 4]+"_Contrast.JPG")

    return enhanced_image


def brightness(img_path, img):
    """
    Adjusts the image's brightness
    Arguments:
        img_path (str): path to the image
        img (np.ndarray): array representing the image
    Returns:
        An array representing the image with adjusted brightness
    """
    # Open the image
    image = Image.open(img_path)

    # Create an enhancer object for brightness
    enhancer = ImageEnhance.Brightness(image)

    brightness_factor = 1.3
    # Adjust the brightness
    enhanced_image = enhancer.enhance(brightness_factor)

    # Save the modified image
    enhanced_image.save(img_path[0:len(img_path) - 4]+"_Contrast.JPG")

    return enhanced_image


def shear(img_path, img):
    """
    Applies a random shear mapping to the image
    Arguments:
        img_path (str): path to the image
        img (np.ndarray): array representing the image
    Returns:
        An array representing the transformed image
    """
    num_rows, num_cols = img.shape[:2]
    src_points = np.float32([[0, 0], [num_cols-1, 0], [0, num_rows-1]])
    dst_points = np.float32([[0, 0],
                             [int(random.choice([0.7, 0.6, 0.5])
                                  * (num_cols-1)), 0],
                             [int(random.choice([0.6, 0.5])
                                  * (num_cols-1)), num_rows-1]])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    img_shear = cv2.warpAffine(img, matrix, (num_cols, num_rows))
    cv2.imwrite(img_path[0:len(img_path) - 4] + "_Shear.JPG", img_shear)

    return img_shear


def projection(img_path, img):
    """
    Projects the image randomly
    Arguments:
        img_path (str): path to the image
        img (np.ndarray): array representing the image
    Returns:
        An array representing the projected image
    """
    num_rows, num_cols = img.shape[:2]
    src_points = np.float32([[0, 0], [num_cols-1, 0], [0, num_rows-1],
                             [num_cols-1, num_rows-1]])
    dst_points = np.float32([[int(random.choice([0, 0.1, 0.2, 0.3])
                                  * num_cols), 0],
                             [int(random.choice([1.0, 0.9, 0.8, 0.7])
                                  * num_cols)-1, 0],
                             [int(random.choice([0, 0.1, 0.2, 0.3])
                                  * num_cols), num_rows-1],
                             [int(random.choice([1.0, 0.9, 0.8, 0.7])
                                  * num_cols), num_rows-1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_projection = cv2.warpPerspective(img, projective_matrix,
                                         (num_cols, num_rows))
    cv2.imwrite(img_path[0:len(img_path) - 4] + "_Projection.JPG",
                img_projection)

    return img_projection


def augment(img_path: str, plot=True) -> None:
    """
    Applies each previous augmentation to an image
    Arguments:
        img_path (str): path to the image
        plot (boolean, default: True): plotting of the resulting images
    """
    img = cv2.imread(img_path)

    f = flip(img_path, img)
    r = rotate(img_path, img)
    c = contrast(img_path, img)
    b = brightness(img_path, img)
    s = shear(img_path, img)
    p = projection(img_path, img)

    if plot:
        plot_images(img, f, r, c, b, s, p)


def main() -> None:
    # set random seed
    random.seed(datetime.now().timestamp())
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isfile(sys.argv[1]) is False:
        return print("Argument {} does not exist".format(sys.argv[1]))
    if (filetype.guess(sys.argv[1]) is None
       or filetype.guess(sys.argv[1]).extension != 'jpg'):
        return print("Argument {} is not a jpeg img".format(sys.argv[1]))
    augment(sys.argv[1])


if __name__ == "__main__":
    main()
