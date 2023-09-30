import pandas as pd

from bigwig_loader import config
from bigwig_loader.dataset import BigWigDataset
from bigwig_loader.download_example_data import download_example_data


def run():
    # Download some data to play with
    download_example_data()

    # created by running examples/create_train_val_test_intervals.py
    train_regions = pd.read_csv("train_regions.tsv", sep="\t")

    # now there is some example data here
    bigwig_dir = config.bigwig_dir
    reference_genome = config.reference_genome
    print("Loading from:", bigwig_dir)

    dataset = BigWigDataset(
        regions_of_interest=train_regions,
        collection=bigwig_dir,
        reference_genome_path=reference_genome,
        sequence_length=1000,
        center_bin_to_predict=1000,
        batch_size=256,
        batches_per_epoch=20,
        maximum_unknown_bases_fraction=0.1,
        sequence_encoder="onehot",
    )

    for input, target in dataset:
        print(input)
        print(target)


if __name__ == "__main__":
    run()
