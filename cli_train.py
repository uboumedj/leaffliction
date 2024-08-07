import os
import sys
import numpy as np

import joblib

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import image_dataset_from_directory


def help():
    print("usage: python3 Train.py [folder of images]")


def make_model(dataset):
    model = models.Sequential()
    model.add(layers.Rescaling(1.0 / 255))
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(128, 128, 1),
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


def main():
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isdir(sys.argv[1]) is False:
        return print("Argument {} is not a directory".format(sys.argv[1]))

    fruit = sys.argv[1].split('/', 1)[1]
    print(fruit)
    jl_name = os.path.join(sys.argv[1] + '.joblib')
    data_acc = None
    predictions_validation = []
    subdirs = [elt for elt in os.listdir(sys.argv[1]) if fruit not in elt]
    subdirs = sorted(subdirs)
    models = [0] * len(subdirs)
    print(subdirs)
    for i in range(len(subdirs)):
        print(f'Training {subdirs[i]} model')
        # data preprocessing
        data = image_dataset_from_directory(
            os.path.join(sys.argv[1], subdirs[i]),
            validation_split=0.2,    # 0.8 for training, 0.2 for validation
            subset="both",
            shuffle=True,
            seed=42,
            image_size=(128, 128),   # 4x less memory and time than (256,256)
        )
        train_data = data[0]
        validation_data = data[1]
        data_acc = data[1]

        # create model
        model = make_model(train_data)

        # Create a learning rate scheduler callback.
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=5
        )
        # Create an early stopping callback.
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # fit model
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
