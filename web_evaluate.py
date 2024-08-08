import os
import sys
import joblib
import click
from cli_train import soft_vote, hard_vote, print_accuracy
# from tensforflow.keras.preprocessing import image_dataset_from_directory temporarily doesn't work ?
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory


@click.command()
@click.option('--dataset', default=None, help='Path to the dataset on which evaluation is to be made')
@click.option('--model_path', default="leaffliction.joblib", help="Path to the trained model to use for prediction")
def main(dataset, model_path):
    if os.path.isdir(dataset) is False:
        return print(f"Argument {dataset} is not a directory")

    model_full_path = os.path.join(model_path)
    data_acc = None
    predictions_validation = []
    model = joblib.load(filename=model_full_path)
    data = image_dataset_from_directory(
        os.path.join(dataset),
        validation_split=0.2,
        subset="both",
        shuffle=True,
        seed=42,
        image_size=(256, 256),
        batch_size=2
    )
    validation_data = data[1]
    data_acc = data[1]

    predictions_validation.append(model.predict(validation_data))

    soft_vote_predictions = soft_vote(predictions_validation)
    hard_vote_predictions = hard_vote(predictions_validation)
    print_accuracy(data_acc, soft_vote_predictions, "soft")
    print_accuracy(data_acc, hard_vote_predictions, "hard")


if __name__ == "__main__":
    main()
