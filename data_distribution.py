import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TypedDict
import click


class CountDict(TypedDict):
    directory_name: str
    number_of_elements: int


def get_class_count(dataset_path: str) -> CountDict:
    """
    Creates a dictionary with the sub-directories as keys and the number of
    images in each sub-directory as values.
    Arguments:
        dataset_path (str): path to the directory containing the dataset
    Returns:
        A CountDict representing the number of images in each sub-directory
    """
    count_dict: CountDict = CountDict()
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(subdir_path) is False:
            return print(f"{subdir_path} not a directory")
        count_dict[subdir] = len(os.listdir(subdir_path))
    return count_dict


def plot_dataset_classes(dataset_path: str):
    """
    Gets the image count of each sub-directory and plots them all
    Arguments:
        dataset_path (str): path to the directory containing the dataset
    """
    count_dict = get_class_count(dataset_path)
    names = list(count_dict.keys())
    values = list(count_dict.values())
    
    fig, axs = plt.subplots(figsize=(15,8), ncols=2)
    sns.barplot(x=values, y=names, hue=names, native_scale=True, ax=axs[0])
    axs[1] = plt.pie(values, labels=names)
    fig.tight_layout()
    plt.show()


@click.command()
@click.option('--dataset', default='./images', help='Path to the directory containing the dataset')
def main(dataset):
    if os.path.isdir(dataset) is False:
        return print(f"Argument {dataset} is not a directory")
    dataset_path = os.path.join(dataset, '')
    plot_dataset_classes(dataset_path)


if __name__ == "__main__":
    main()
