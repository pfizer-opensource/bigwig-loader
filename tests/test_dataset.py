import pytest


@pytest.fixture
def pytorch_dataset(bigwig_path, reference_genome_path, merged_intervals):
    from bigwig_loader.dataset import BigWigDataset

    dataset = BigWigDataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=2000,
        center_bin_to_predict=1000,
        window_size=4,
        batch_size=256,
        batches_per_epoch=4,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=2,
    )
    return dataset


def test_output_shape(pytorch_dataset):
    for sequence, target in pytorch_dataset:
        assert target.shape == (256, 2, 250)
