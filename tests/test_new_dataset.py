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


@pytest.fixture
def dataset_with_track_sampling(bigwig_path, reference_genome_path, merged_intervals):
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
        sub_sample_tracks=1,
    )
    return dataset


# @pytest.mark.timeout(10)
def test_output_shape(dataset):
    for i, (sequence, values) in enumerate(dataset):
        print(i, "---", flush=True)
        assert values.shape == (265, 2, 250)


def test_output_shape_sub_sampled_tracks(dataset_with_track_sampling):
    for i, (sequence, values, track_indices) in enumerate(dataset_with_track_sampling):
        print(i, "---", flush=True)
        assert len(track_indices) == 1
        assert values.shape == (265, 1, 250)


def test_batch_return_type(bigwig_path, reference_genome_path, merged_intervals):
    from bigwig_loader.batch import Batch

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
        sub_sample_tracks=1,
        return_batch_objects=True,
    )
    for i, batch in enumerate(dataset):
        assert isinstance(batch, Batch)
        assert batch.track_indices is not None


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
        sub_sample_tracks=1,
    )

    test_output_shape_sub_sampled_tracks(ds)
    print("done")
