import os
import sys
import filetype
import numpy as np
import matplotlib.pyplot as plt
import joblib
import click
from plantcv import plantcv as pcv
# from keras.preprocessing import image temporrarily doesn't work ?
from keras._tf_keras.keras.preprocessing import image
from PIL import Image
from cli_transform import transform_gaussian_blur, transform_masked
from cli_transform import transform_roi, transform_analysis
from cli_transform import transform_pseudolandmarks, find


def plot_images(path, fruit, class_pred):
    # Open the images
    image = Image.open(path)
    b = Image.open(f"{fruit}_BLURRED.JPG")
    m = Image.open(f"{fruit}_MASKED.JPG")
    p = Image.open(f"{fruit}_PSEUDOLANDMARKS.JPG")
    r = Image.open(f"{fruit}_ROI_OBJECTS.JPG")
    a = Image.open(f"{fruit}_ANALYZED.JPG")

    plt.figure(figsize=(8, 6))

    plt.subplot(3, 2, 1)
    plt.imshow(image)
    plt.title('Original')

    plt.subplot(3, 2, 2)
    plt.imshow(b)
    plt.title('Gaussian Blur')

    plt.subplot(3, 2, 3)
    plt.imshow(m)
    plt.title('Mask')

    plt.subplot(3, 2, 4)
    plt.imshow(p)
    plt.title('Pseudolandmark')

    plt.subplot(3, 2, 5)
    plt.imshow(r)
    plt.title('ROI objects')

    plt.subplot(3, 2, 6)
    plt.imshow(a)
    plt.title('Analysis')

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)

    # center text
    fig = plt.gcf()
    fontsize = 14
    text_width = len(class_pred) * 0.1
    text_x = 0.5 - text_width / (2 * fig.get_figwidth())
    plt.text(text_x,
             0.5,
             class_pred,
             fontsize=fontsize,
             transform=fig.transFigure)

    plt.show()


def make_images(path, fruit):
    """
    Creates transformations of the original image for the predictions
    Arguments:
        path (str): path to the original image
        fruit (str): the type of fruit we're working with
    """
    pcv.params.debug_outdir = "."
    img, path, filename = pcv.readimage(path)

    img_b = transform_gaussian_blur(img)
    pcv.print_image(img_b, f"{fruit}_BLURRED.JPG")
    img_p = transform_pseudolandmarks(img)
    pcv.print_image(img_p, f"{fruit}_PSEUDOLANDMARKS.JPG")
    img_m = transform_masked(img)
    pcv.print_image(img_m, f"{fruit}_MASKED.JPG")
    img_roi = transform_roi(img)
    pcv.print_image(img_roi, f"{fruit}_ROI_OBJECTS.JPG")
    img_a = transform_analysis(img)
    pcv.print_image(img_a, f"{fruit}_ANALYZED.JPG")


def remove_images(fruit):
    """
    Removes the no longer necessary image files created during processing
    Arguments:
        fruit (str): the type of fruit we're working with
    """
    os.remove(f'./{fruit}_BLURRED.JPG')
    os.remove(f'./{fruit}_PSEUDOLANDMARKS.JPG')
    os.remove(f'./{fruit}_MASKED.JPG')
    os.remove(f'./{fruit}_ROI_OBJECTS.JPG')
    os.remove(f'./{fruit}_ANALYZED.JPG')


def load_image(type, fruit):
    """
    Loads the specified transformed image of a fruit into a tensor array
    Arguments:
        type (str): type of transformation to be loaded
        fruit (str): the type of fruit we're working with
    Returns
        A np.ndarray containing the transformed image's data
    """
    path = None

    if type == 'blur':
        path = f'./{fruit}_BLURRED.JPG'
    if type == 'pseudolandmarks':
        path = f'./{fruit}_PSEUDOLANDMARKS.JPG'
    if type == 'mask':
        path = f'./{fruit}_MASKED.JPG'
    if type == 'roi':
        path = f'./{fruit}_ROI_OBJECTS.JPG'
    if type == 'analysis':
        path = f'./{fruit}_ANALYZED.JPG'

    img = image.load_img(path, target_size=(128, 128))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels), need to add a dimension because the model
    # expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor


def print_image_summary(images, cols=8):
    for i in range(len(images)):
        channels = images[i].shape[-1]
        images_ = images[i][0]
        rows = channels // cols
        plt.figure(figsize=(cols*2,rows*2))
        for i in range(channels):
            plt.subplot(rows,cols+2,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(images_[:,:,i], cmap='gray')
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def soft_vote(predictions):
    """
    Computes the average prediction for each sample and class (soft vote)
    Arguments:
        predictions (np.ndarray): the predictions array
    """
    ensemble_prediction = np.mean(predictions, axis=0)

    # Convert ensemble predictions to class labels
    # (index of the maximum probability)
    max_index = np.argmax(ensemble_prediction)

    print(f'soft vote prediction percentage: {ensemble_prediction[max_index]}')

    return max_index


def hard_vote(preds):
    """
    Computes the majority vote for each sample and class (hard vote)
    Arguments:
        preds (np.ndarray): the predictions array
    """
    nb_classes = len(preds[0])

    pred_array = [pred for i in range(len(preds)) for pred in preds[i]]

    print(f'hard vote pred. percentage : {pred_array[np.argmax(pred_array)]}')

    return np.argmax(pred_array) % nb_classes


@click.command()
@click.option('--image', default=None, help='Path to the image to predict')
@click.option('--model_path', default="leaffliction.joblib", help="Path to the trained model to use for prediction")
def main(image, model_path):
    if os.path.isfile(image) is False:
        return print(f"Argument {image} is not a file")
    if (filetype.guess(image) is None
       or filetype.guess(image).extension != 'jpg'):
        return print(f"{image} is not a jpeg image")

    path = find(model_path, ".")
    if path is None:
        return print(f'{model_path} model has not been trained')
    models = joblib.load(filename=path)

    fruit = image.split("/", 1)[0]
    make_images(image, fruit)

    predictions = []
    transformations = sorted(os.listdir(os.path.dirname(path)+f'/{fruit}'))
    for i in range(len(models)):
        prediction = models[i].predict(load_image(transformations[i], fruit))
        predictions.append(prediction[0])

    s_vote = soft_vote(predictions)
    h_vote = hard_vote(predictions)
    classes = sorted(os.listdir(os.path.dirname(os.path.dirname(sys.argv[1]))))
    print(f'soft voting predicted : {classes[s_vote]}')
    print(f'hard voting predicted : {classes[h_vote]}')

    plot_images(sys.argv[1],
                fruit,
                classes[s_vote] if s_vote > h_vote else classes[h_vote])

    remove_images(fruit)


if __name__ == "__main__":
    main()
