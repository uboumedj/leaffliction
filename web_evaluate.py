import os
import sys
import joblib
from cli_train import soft_vote, hard_vote, print_accuracy
# from tensforflow.keras.preprocessing import image_dataset_from_directory temporarily doesn't work ?
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory


def main():
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isdir(sys.argv[1]) is False:
        return print("Argument {} is not a directory".format(sys.argv[1]))

    fruit = sys.argv[1].split('/', 1)[1]
    jl_name = os.path.join(sys.argv[1] + '.joblib')
    data_acc = None
    predictions_validation = []
    subdirs = [elt for elt in os.listdir(sys.argv[1]) if fruit not in elt]
    subdirs = sorted(subdirs)
    models = joblib.load(filename=jl_name)
    for (model, subdir) in zip(models, subdirs):
        print(f'evaluating {subdir} model')
        # data preprocessing
        data = image_dataset_from_directory(
            os.path.join(sys.argv[1], subdir),
            validation_split=0.2,    # 0.8 for training, 0.2 for validation
            subset="both",
            shuffle=True,
            seed=42,
            image_size=(256, 256),   # 4x less memory and time than (256,256)
        )
        validation_data = data[1]
        data_acc = data[1]

        predictions_validation.append(model.predict(validation_data))
        print()

    soft_vote_predictions = soft_vote(predictions_validation)
    hard_vote_predictions = hard_vote(predictions_validation)
    print_accuracy(data_acc, soft_vote_predictions, "soft")
    print_accuracy(data_acc, hard_vote_predictions, "hard")


if __name__ == "__main__":
    main()
