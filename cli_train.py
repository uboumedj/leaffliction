import os
import sys
import numpy as np
import joblib
import click
import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks temporarily doesn't work ?
from keras._tf_keras.keras import layers, models, callbacks
# from tensorflow.keras.utils import image_dataset_from_directory temporarily doesn't work ?
from keras._tf_keras.keras.utils import image_dataset_from_directory


def make_model(dataset):
    """
    Builds a tensorflow CNN model adapted to the dataset. Here you can change
    the basic structure of the model used in this project
    Arguments:
        dataset (tf.data.Dataset object): the image dataset we want to train the model on
    Returns:
        A tensorflow model
    """
    model = models.Sequential()
    model.add(layers.Rescaling(1.0 / 255))
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(256, 256, 1)
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(len(dataset.class_names), activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def soft_vote(predictions):
    # Stack the predictions along the first axis to form a 3D array
    # (num_samples, num_models, num_classes)
    stacked_predictions = np.stack(predictions, axis=1)

    # Compute the average prediction for each sample and class (soft vote)
    ensemble_prediction = np.mean(stacked_predictions, axis=1)

    # Convert the ensemble predictions to class labels
    # (index of the maximum probability)
    ensemble_prediction = np.argmax(ensemble_prediction, axis=1)

    return ensemble_prediction


def hard_vote(predictions):
    # print(predictions)
    nb_classes = len(predictions[0][0])
    # print(f'nb_classes = {nb_classes}')
    # Stack the predictions along the first axis to form a 3D array
    # (num_samples, num_models, num_classes)
    stacked_pred = np.stack(predictions, axis=1)

    # Concatenate the innermost arrays along the last axis
    # (num_samples, num_models * num_classes)
    concatenated_pred = stacked_pred.reshape(stacked_pred.shape[0], -1)

    # Compute the majority vote for each sample and class (hard vote)
    ensemble_prediction = np.argmax(concatenated_pred, axis=-1)
    true_ensemble = []
    for i in range(len(ensemble_prediction)):
        true_ensemble.append(ensemble_prediction[i] % nb_classes)
    return true_ensemble


def print_accuracy(data, ensemble_prediction, mode="soft"):
    # Get the ground truth labels from the test dataset
    test_labels = np.concatenate([y for _, y in data], axis=0)
    # Now 'ensemble_prediction' contains the final ensemble prediction
    # for the test dataset.
    # Each element of 'ensemble_prediction' will be an array of probabilities
    # representing the ensemble's confidence for each class for
    # the corresponding sample.

    # Sum the equal values between ensemble and truth
    correct_predictions = np.sum(ensemble_prediction == test_labels)

    # Calculate the accuracy
    accuracy = correct_predictions / len(test_labels)

    print(f"Accuracy {mode} vote:", accuracy)
    return


@click.command()
@click.option('--data', default="./images", help='Directory of source data')
def main(data):
    if os.path.isdir(data) is False:
        return print(f"Argument {data} is not a directory")
    fruit = data.split('/', 1)[1]
    jl_name = os.path.join(data + '.joblib')
    data_acc = None
    predictions_validation = []
    subdirs = [elt for elt in os.listdir(data) if fruit not in elt]
    subdirs = sorted(subdirs)
    models = [0] * len(subdirs)
    for i in range(len(subdirs)):
        print(f'Training {subdirs[i]} model')
        data = image_dataset_from_directory(
            os.path.join(data, subdirs[i]),
            validation_split=0.2,
            subset="both",
            shuffle=True,
            seed=42,
            image_size=(256, 256),
        )
        train_data = data[0]
        validation_data = data[1]
        data_acc = data[1]

        model = make_model(train_data)

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=5
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        model.fit(
            train_data,
            epochs=3,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr]
        )

        models[i] = model
        predictions_validation.append(model.predict(validation_data))
        print()

    soft_vote_predictions = soft_vote(predictions_validation)
    hard_vote_predictions = hard_vote(predictions_validation)
    print_accuracy(data_acc, soft_vote_predictions, "soft")
    print_accuracy(data_acc, hard_vote_predictions, "hard")

    joblib.dump(models, jl_name)


if __name__ == "__main__":
    main()
