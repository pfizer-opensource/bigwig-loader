import pytest

from bigwig_loader.dataset_new import Dataset


@pytest.fixture
def dataset(bigwig_path, reference_genome_path, merged_intervals):
    dataset = Dataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=2000,
        center_bin_to_predict=1000,
        window_size=4,
        batch_size=265,
        batches_per_epoch=10,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=2,
    )
    return dataset


# @pytest.mark.timeout(10)
def test_output_shape(dataset):
    for i, (sequence, values) in enumerate(dataset):
        print(i, "---", flush=True)
        assert values.shape == (265, 2, 250)


if __name__ == "__main__":
    from bigwig_loader.collection import BigWigCollection
    from bigwig_loader.download_example_data import get_example_bigwigs_files
    from bigwig_loader.download_example_data import get_reference_genome

    bigwig_path = get_example_bigwigs_files()
    collection = BigWigCollection(bigwig_path, first_n_files=2)

    merged_intervals = collection.intervals(
        ["chr3", "chr4", "chr5"], exclude_chromosomes=["chr4"], merge=True, threshold=2
    )
    ds = Dataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=get_reference_genome(),
        sequence_length=2000,
        center_bin_to_predict=1000,
        window_size=4,
        batch_size=265,
        batches_per_epoch=10,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=2,
    )

    test_output_shape(ds)
    print("done")
