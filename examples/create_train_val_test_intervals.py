from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection


def create_train_val_test(
    val_chromosomes=("chr6", "chr7"),
    test_chromosomes=("chr8", "chr9"),
    threshold=2,
    merge_allow_gap=1000,
):
    bigwig_path = config.bigwig_dir
    print(bigwig_path)
    collection = BigWigCollection(bigwig_path)
    print(collection)
    merged_intervals = collection.intervals(
        "standard",
        exclude_chromosomes=val_chromosomes + test_chromosomes,
        merge=True,
        threshold=threshold,
        merge_allow_gap=merge_allow_gap,
    )
    print("Train intervals:", merged_intervals)
    merged_intervals.to_csv("train_regions.tsv", sep="\t", index=None)
    merged_intervals = collection.intervals(
        include_chromosomes=val_chromosomes,
        merge=True,
        threshold=threshold,
        merge_allow_gap=merge_allow_gap,
    )
    print("Validation intervals:", merged_intervals)
    merged_intervals.to_csv("validation_regions.tsv", sep="\t", index=None)
    merged_intervals = collection.intervals(
        include_chromosomes=test_chromosomes,
        merge=True,
        threshold=threshold,
        merge_allow_gap=merge_allow_gap,
    )
    print("Test intervals:", merged_intervals)
    merged_intervals.to_csv("test_regions.tsv", sep="\t", index=None)


if __name__ == "__main__":
    create_train_val_test()
