import os
import filetype
from plantcv import plantcv as pcv
from shutil import copytree
import matplotlib.pyplot as plt
import click
from cli_transform import mask_image, transform_analysis, transform_gaussian_blur
from cli_transform import transform_masked, transform_pseudolandmarks, transform_roi
from cli_transform import transform_directory


def transform_image(img_path: str, dst: str, type: str) -> None:
    """
    Performs six image transformations on an image using PlantCV's functions
    Arguments:
        img_path (string): path to original image
        dst (string): path to destination directory
        type (string): type of transformation requested
    """
    # Initialisation
    img, path, filename = pcv.readimage(img_path)
    new_image_prefix = dst + '/'
    if img_path[0:2] == "./":
        img_path = img_path[2:]
    new_image_prefix += img_path[img_path.find('/') + 1:-4]
    new_image_directory = os.path.split(new_image_prefix)[0]
    pcv.params.debug_outdir = new_image_directory
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
        roi_img = transform_roi(img, dst=new_image_directory)
        pcv.print_image(roi_img, new_image_prefix + "_ROI_OBJECTS.JPG")

    # Analyse objects
    if type in ['analysis', 'all']:
        analyse_img = transform_analysis(img)
        pcv.print_image(analyse_img, new_image_prefix + "_ANALYZED.JPG")

    # Pseudolandmarks
    if type in ['pseudolandmarks', 'all']:
        pseudo_img = transform_pseudolandmarks(img, dst=new_image_directory)
        pcv.print_image(pseudo_img, new_image_prefix + "_PSEUDOLANDMARKS.JPG")


@click.command()
@click.option('--src', default=None, help='Directory of original data')
@click.option('--dst', default=None, help="Directory of the transformed data")
@click.option('--type', default='all',
              help="Type of transformation requested, choose between"
                   + " ['all', 'blur', 'mask', 'roi', 'analysis', "
                   + "'pseudolandmarks', 'maskblur']")
def main(src, dst, type) -> None:
    known_types = ['all', 'blur', 'mask', 'roi', 'analysis',
                   'pseudolandmarks', 'maskblur']
    if type not in known_types:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        print(f"Requested type '{type}' was not recognised")
        ctx.exit()
    if (src is not None and dst is not None and src != dst):
        if os.path.isdir(src) is False:
            return print(f"{src} does not exist or is not a directory")
        if src[-1] == '/':
            src = src[:-1]
        copytree(src, dst, dirs_exist_ok=True)
        if type == 'all':
            known_types.remove('all')
            known_types.remove('maskblur')
            for single_type in known_types:
                transform_directory(src=src, dst=dst, type=single_type)
                print(f"Finished applying {single_type}! Resulting images"
                        + f" can be found at {dst}\n")
        else:
            transform_directory(src=src, dst=src, type=type)
    else:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        print("You must either provide a [src] and [dst] directory, or a "
              + "source file as arguments for the transformation")
        ctx.exit()


if __name__ == "__main__":
    main()
