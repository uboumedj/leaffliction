import os
import sys
import filetype
import numpy as np
import click
from shutil import rmtree, copy
import joblib
from plantcv import plantcv as pcv
# from tensforflow.keras.preprocessing import image temporarily doesn't work ?
from keras._tf_keras.keras.preprocessing import image
from cli_transform import transform_gaussian_blur, transform_masked
from cli_transform import transform_roi, transform_analysis
from cli_transform import transform_pseudolandmarks, find
from cli_predict import soft_vote, hard_vote


def generate_transformed_images(image_path):
    """
    Creates transformations of the original image for the predictions
    Arguments:
        image_path (str): path to the original image
    """
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    img, path, filename = pcv.readimage(image_path)

    img_b = transform_gaussian_blur(img)
    pcv.print_image(img_b, "./tmp/BLURRED.JPG")
    img_p = transform_pseudolandmarks(img)
    pcv.print_image(img_p, "./tmp/PSEUDOLANDMARKS.JPG")
    img_m = transform_masked(img)
    pcv.print_image(img_m, "./tmp/MASKED.JPG")
    img_roi = transform_roi(img)
    pcv.print_image(img_roi, "./tmp/ROI_OBJECTS.JPG")
    img_a = transform_analysis(img)
    pcv.print_image(img_a, "./tmp/ANALYZED.JPG")
    copy(image_path, "./tmp")


def load_image(image_path):
    """
    Loads the specified requested image of a fruit into a tensor array
    Arguments:
        image_path (str): path to the requested image
    Returns
        A np.ndarray containing the requested image's data
    """
    img = image.load_img(image_path, target_size=(256, 256))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels), need to add a dimension because the model
    # expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor


@click.command()
@click.option('--image', default=None, help='Path to the image to predict')
@click.option('--model_path', default="leaffliction.joblib", help="Path to the trained model to use for prediction")
def main(image, model_path):
    if os.path.isfile(image) is False:
        return print(f"{image} does not exist or is not a file")
    if (filetype.guess(image) is None
       or filetype.guess(image).extension != 'jpg'):
        return print(f"{image} is not a jpeg image")

    path = find(model_path, ".")
    if path is None:
        return print(f'{model_path} model has not been trained')
    model = joblib.load(filename=path)

    generate_transformed_images(image)

    predictions = []
    transformations = sorted(os.listdir("./tmp"))
    print(transformations)
    for i in range(len(transformations)):
        prediction = model.predict(load_image("./tmp/" + transformations[i]))
        predictions.append(prediction[0])

    s_vote = soft_vote(predictions)
    h_vote = hard_vote(predictions)
    classes = sorted(os.listdir(os.path.dirname(os.path.dirname(image))))
    print(f'soft voting predicted : {classes[s_vote]}')
    print(f'hard voting predicted : {classes[h_vote]}')

    rmtree("./tmp")


if __name__ == "__main__":
    main()
