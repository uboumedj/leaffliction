import os
import filetype
import fnmatch
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import click


def find(pattern, path):
    """
    Finds the file whose name matches the requested pattern
    Arguments:
        pattern (string): requested pattern
        path (string): path where the file must be searched
    Returns:
        Complete path to the file, or None if file wasn't found
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None


def mask_image(img):
    """
    Generates the black and white mask necessary for many transformations
    Arguments:
        img (np.ndarray): Array representing the image
    Returns:
        A np.ndarray representing the mask of the image
    """
    gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    threshold = pcv.threshold.binary(gray_img=gray_img,
                                     threshold=60,
                                     object_type='dark')
    mask = pcv.invert(threshold)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    return mask


def transform_gaussian_blur(img):
    """
    Generates a gaussian blur transformation of the image
    Arguments:
        img (np.ndarray): Array representing the image
    Returns:
        A np.ndarray representing the transformed image
    """
    return pcv.gaussian_blur(img, ksize=(11, 11))


def transform_masked(img):
    """
    Generates a masked transformation of the image
    Arguments:
        img (np.ndarray): Array representing the image
    Returns:
        A np.ndarray representing the transformed image
    """
    mask = mask_image(img)
    return pcv.apply_mask(img=img, mask=mask, mask_color='white')


def transform_roi(img, dst='.'):
    """
    Finds ROI objects on the img and generates a representation of them
    Arguments:
        img (np.ndarray): Array representing the image
        dst (str): Destination directory for the temporary files
    Returns:
        A np.ndarray representing the transformed image
    """
    mask = mask_image(img)
    colorized_mask = pcv.visualize.colorize_masks(masks=[mask], colors=['green'])
    merged_image = pcv.visualize.overlay_two_imgs(colorized_mask, img, alpha=0.3)
    return merged_image


def transform_analysis(img):
    """
    Generates a representation of the analysis of the image
    Arguments:
        img (np.ndarray): Array representing the image
    Returns:
        A np.ndarray representing the transformed image
    """
    mask = mask_image(img)
    shape_image = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1)
    return shape_image


def transform_pseudolandmarks(img, dst='.'):
    """
    Finds pseudolandmarks on the img and generates a representation of them
    Arguments:
        img (np.ndarray): Array representing the image
        dst (str): Destination directory for the temporary files
    Returns:
        A np.ndarray representing the transformed image
    """
    pcv.params.debug_outdir = dst
    mask = mask_image(img)
    pcv.params.debug = 'print'
    pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)
    pcv.params.debug = 'None'
    pseudolandmarks_file = find("*_pseudolandmarks.png", dst)
    transformed_img, path, filename = pcv.readimage(pseudolandmarks_file)
    os.remove(pseudolandmarks_file)
    return transformed_img


def transform_image(img_path: str, dst: str, type: str) -> None:
    """
    Performs six image transformations on an image using PlantCV's functions
    Arguments:
        img_path (string): path to original image
        dst (string): directory where resulting images will be stored
        type (string): type of transformation requested
    """
    # Initialisation
    img, path, filename = pcv.readimage(img_path)
    pcv.params.debug_outdir = dst
    new_image_prefix = dst + '/'
    if img_path[0:2] == "./":
        img_path = img_path[2:]
    new_image_prefix += img_path[img_path.find('/') + 1:-4]
    new_image_directory = os.path.split(new_image_prefix)[0]
    if not os.path.exists(new_image_directory):
        os.makedirs(new_image_directory)

    mask = mask_image(img)

    # Gaussian blur of image
    if type in ['blur', 'all']:
        gaussian_blur = transform_gaussian_blur(img)
        pcv.print_image(gaussian_blur, new_image_prefix + "_BLURRED.JPG")

    # Gaussian blur of mask
    if type in ['maskblur']:
        mask_blur = transform_gaussian_blur(mask)
        pcv.print_image(mask_blur, new_image_prefix + "_MASKBLUR.JPG")

    # Masked image
    if type in ['mask', 'all']:
        masked = transform_masked(img)
        pcv.print_image(masked, new_image_prefix + "_MASKED.JPG")

    # ROI objects
    if type in ['roi', 'all']:
        roi_img = transform_roi(img, dst=dst)
        pcv.print_image(roi_img, new_image_prefix + "_ROI_OBJECTS.JPG")

    # Analyse objects
    if type in ['analysis', 'all']:
        analyse_img = transform_analysis(img)
        pcv.print_image(analyse_img, new_image_prefix + "_ANALYZED.JPG")

    # Pseudolandmarks
    if type in ['pseudolandmarks', 'all']:
        pseudo_img = transform_pseudolandmarks(img, dst=dst)
        pcv.print_image(pseudo_img, new_image_prefix + "_PSEUDOLANDMARKS.JPG")


def transform_directory(src: str, dst: str, type: str) -> None:
    """
    Performs image transformations on every image of a directory,
    including images in sub-directories
    Arguments:
        src (string): path of directory where transformations will be applied
        dst (string): directory where resulting images will be stored
        type (string): type of transformation requested
    """
    file_count = 0
    dir_content = os.listdir(src)
    for filename in dir_content:
        file_path = os.path.join(src, filename)
        if os.path.isfile(file_path) and (filetype.guess(file_path) is not None
           and filetype.guess(file_path).extension == 'jpg'):
            transform_image(img_path=file_path, dst=dst, type=type)
            file_count += 1
            print(f"\rApplying {type} to {file_count}/{len(dir_content)}...",
                  end='',
                  flush=True)
        elif os.path.isdir(file_path):
            transform_directory(src=file_path, dst=dst, type=type)
    if file_count > 0:
        print(f"\nApplied {type} to {file_count} JPG files in {src}")


def plot_images(img_path: str, dst: str) -> None:
    """
    In the case where only one file is requested, assignment requires to
    display the set of image transformations. This does it
    Arguments:
        img_path (str): Path to original image
        dst (str): Destination where the transformations were stored
    """
    # Get the paths to each transformed image
    img_prefix = dst + '/'
    if img_path[0:2] == "./":
        img_path = img_path[2:]
    img_prefix += img_path[img_path.find('/') + 1:-4]
    maskblur_path = img_prefix + "_MASKBLUR.JPG"
    mask_path = img_prefix + "_MASKED.JPG"
    roi_path = img_prefix + "_ROI_OBJECTS.JPG"
    analysis_path = img_prefix + "_ANALYZED.JPG"
    landmark_path = img_prefix + "_PSEUDOLANDMARKS.JPG"
    blur_path = img_prefix + "_BLURRED.JPG"

    # Check existence of transformed files
    image_names = ['Original Image']
    img_blur, img_mask, img_roi, img_analyze, img_marks = (None,)*5
    img_colors, img_maskblur = (None,)*2
    if os.path.isfile(maskblur_path):
        img_maskblur, path, filename = pcv.readimage(maskblur_path)
        image_names.append('Blurred Mask')
    if os.path.isfile(mask_path):
        img_mask, path, filename = pcv.readimage(mask_path)
        image_names.append('Masked Image')
    if os.path.isfile(roi_path):
        img_roi, path, filename = pcv.readimage(roi_path)
        image_names.append('ROI Objects')
    if os.path.isfile(analysis_path):
        img_analyze, path, filename = pcv.readimage(analysis_path)
        image_names.append('Analysed Objects')
    if os.path.isfile(landmark_path):
        img_marks, path, filename = pcv.readimage(landmark_path)
        image_names.append('Pseudolandmarks')
    if os.path.isfile(blur_path):
        img_blur, path, filename = pcv.readimage(blur_path)
        image_names.append('Gaussian blur')

    # Initialise plot
    length = len(image_names)
    rows = 1 if length <= 3 else 2 if length <= 6 else 3
    cols = length if length <= 3 else 3
    plotted = 1

    # Plot original image
    img_original, path, filename = pcv.readimage(img_path)
    plt.subplot(rows, cols, plotted)
    plt.imshow(img_original)
    plt.title('Original Image')
    plotted += 1

    # Plot blurred mask of image
    if img_maskblur is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_maskblur)
        plt.title("Blurred mask")
        plotted += 1

    # Plot masked image
    if img_mask is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_mask)
        plt.title("Masked image")
        plotted += 1

    # Plot roi image
    if img_roi is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_roi)
        plt.title("ROI objects")
        plotted += 1

    # Plot analyzed image
    if img_analyze is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_analyze)
        plt.title("Analyzed objects")
        plotted += 1

    # Plot image's landmarks
    if img_marks is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_marks)
        plt.title("Pseudolandmarks")
        plotted += 1

    # Plot image's true blur (assignment's PDF blur was a blur of the mask)
    if img_blur is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_blur)
        plt.title("Gaussian blur")
        plotted += 1

    # Display previous transformations along with original image on same plot
    plt.suptitle("Stored transformations of original image")
    plt.show()


@click.command()
@click.option('--src', default=None, help='Directory of original data')
@click.option('--dst', default=None, help="Directory of the transformed data")
@click.option('--type', default='all',
              help="Type of transformation requested, choose between"
                   + " ['all', 'blur', 'mask', 'roi', 'analysis', "
                   + "'pseudolandmarks', 'maskblur']")
@click.option('--separate', default=True,
              help="Separation of transformations in different directories")
@click.argument('file', required=False)
def main(file, src, dst, type, separate) -> None:
    # Check if requested type is acceptable
    known_types = ['all', 'blur', 'mask', 'roi', 'analysis',
                   'pseudolandmarks', 'maskblur']
    if type not in known_types:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        print(f"Requested type '{type}' was not recognised")
        ctx.exit()

    # Single file transformation
    if file is not None:
        if os.path.isfile(file) is False:
            return print(f"{file} does not exist or is not a file")
        if (filetype.guess(file) is None
           or filetype.guess(file).extension != 'jpg'):
            return print(f"{file} is not a jpeg image")
        img_dir = file
        if img_dir[0:2] == "./":
            img_dir = img_dir[2:]
        img_dir = img_dir[:img_dir.find('/')]
        transform_image(file, dst=f"./{img_dir}_transformed", type=type)
        plot_images(file, dst=f"./{img_dir}_transformed")

    # Directory transformation
    elif (src is not None and dst is not None):
        if os.path.isdir(src) is False:
            return print(f"{src} does not exist or is not a directory")
        if src[-1] == '/':
            src = src[:-1]
        if separate is True:
            if type == 'all':
                known_types.remove('all')
                known_types.remove('maskblur')
                for single_type in known_types:
                    transform_directory(src=src,
                                        dst=f"{dst}/{src}/{single_type}",
                                        type=single_type)
                    print(f"Finished applying {single_type}! Resulting images"
                          + f" can be found at {dst}/{src}/{single_type}\n")
            else:
                transform_directory(src=src,
                                    dst=f"{dst}/{src}/{type}",
                                    type=type)
        else:
            transform_directory(src=src, dst=f"{dst}/{src}", type=type)
    # Not enough arguments
    else:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        print("You must either provide a [src] and [dst] directory, or a "
              + "source file as arguments for the transformation")
        ctx.exit()


if __name__ == "__main__":
    main()
