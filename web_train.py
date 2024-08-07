import os
import joblib
import click
from keras import callbacks
# from tensforflow.keras.preprocessing import image_dataset_from_directory temporarily doesn't work ?
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
from cli_train import make_model, soft_vote, hard_vote, print_accuracy


@click.command()
@click.option('--data', default="./images", help='Directory of source data')
@click.option('--model_path', default="./leaffliction.joblib", help='Path to saved model')
def main(data, model_path):
    """
    The web_train program trains a single CNN model on the whole dataset
    (Apples and Grapes) located inside a given directory. 
    """
    if os.path.isdir(data) is False:
        return print(f"Argument {data} is not a directory")

    # Data fetching and preprocessing
    print(f"Training model on dataset in {data}")
    model_full_path = os.path.join(model_path)
    data_acc = None
    predictions_validation = []
    subdirectories = [subdir for subdir in os.listdir(data)]
    classes = sorted(subdirectories)
    print(f"Data classes found : {classes}")

    print("Initialising model...")
    dataset = image_dataset_from_directory(
        os.path.join(data),
        validation_split=0.2,
        subset="both",
        shuffle=True,
        seed=42,
        image_size=(256, 256)
    )
    train_data = dataset[0]
    validation_data = dataset[1]
    data_acc = dataset[1]

    # Model creation and training
    model = make_model(train_data)

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=5
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print(f'Starting model training...')
    model.fit(
        train_data,
        epochs=3,
        validation_data=validation_data,
        callbacks=[early_stopping, reduce_lr]
    )

    # Model evaluation metrics
    predictions_validation.append(model.predict(validation_data))

    soft_vote_predictions = soft_vote(predictions_validation)
    hard_vote_predictions = hard_vote(predictions_validation)
    print_accuracy(data_acc, soft_vote_predictions, "soft")
    print_accuracy(data_acc, hard_vote_predictions, "hard")

    # Model save
    joblib.dump(model, model_full_path)


if __name__ == "__main__":
    main()
