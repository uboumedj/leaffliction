import os
import sys
import random
import cv2
import imutils
import filetype
import numpy as np
import click
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageEnhance
from data_distribution import get_class_count


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

@click.command()
@click.option('--src', default=None, help='Path to the dataset in need of augmentation')
@click.argument('file', required=False)
def main(src, file) -> None:
    random.seed(datetime.now().timestamp())
    # Single file augmentation
    if file is not None:
        if os.path.isfile(file) is False:
            return print(f"{file} does not exist or is not a file")
        if (filetype.guess(file) is None
           or filetype.guess(file).extension != 'jpg'):
            return print(f"{file} is not a jpeg image")
        augment(file)
    # Directory augmentation
    elif (src is not None):
        class_count = get_class_count(src)
        highest_class = max(class_count, key=lambda key: class_count[key])
        max_count = class_count[highest_class]
        class_count.pop(highest_class)
        for key in class_count:
            file_count = 0
            augment_amount = int((max_count - class_count[key]) / 7)
            image_list = os.listdir(os.path.join(src, key))
            for i in range(augment_amount):
                print(f"\rAugmenting {i+1}/{augment_amount} images from {key}...", end='', flush=True)
                current_image = random.choice(image_list)
                augment(os.path.join(src, key, current_image), plot=False)
                image_list.remove(current_image)
                file_count += 1
            if file_count != 0:
                print(" Done !")
        print('All subdirectories augmented')
    else:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        print("You must either provide a [src] directory, or a "
              + "source file as arguments for the augmentation")
        ctx.exit()


if __name__ == "__main__":
    main()
