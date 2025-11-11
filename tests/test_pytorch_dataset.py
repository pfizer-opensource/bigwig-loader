from math import isnan

import pandas as pd
import pytest

from bigwig_loader import config

torch = pytest.importorskip("torch")


@pytest.fixture
def pytorch_dataset(bigwig_path, reference_genome_path, merged_intervals):
    from bigwig_loader.pytorch import PytorchBigWigDataset

    dataset = PytorchBigWigDataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=1000,
        center_bin_to_predict=1000,
        batch_size=256,
        batches_per_epoch=4,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=2,
    )
    return dataset


def test_pytorch_dataset(pytorch_dataset):
    for sequence, target in pytorch_dataset:
        assert target.shape == (256, 1000, 2)


def test_input_and_target_is_torch_tensor(pytorch_dataset):
    sequence, target = next(iter(pytorch_dataset))
    assert isinstance(sequence, torch.Tensor)
    assert isinstance(target, torch.Tensor)


@pytest.mark.parametrize("default_value", [0.0, 4.0, 5.6, torch.nan])
def test_pytorch_dataset_with_window_function(
    default_value, bigwig_path, reference_genome_path, merged_intervals
):
    from bigwig_loader.pytorch import PytorchBigWigDataset

    center_bin_to_predict = 2048
    window_size = 128
    reduced_dim = center_bin_to_predict // window_size

    batch_size = 16

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin({"chr1", "chr3", "chr5"})]
    chromosomes = list(df["chr"])[:batch_size]
    centers = list(df["center"])[:batch_size]

    position_sampler = [(chrom, center) for chrom, center in zip(chromosomes, centers)]

    dataset = PytorchBigWigDataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=center_bin_to_predict * 2,
        center_bin_to_predict=center_bin_to_predict,
        window_size=1,
        batch_size=batch_size,
        batches_per_epoch=1,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=3,
        custom_position_sampler=position_sampler,
        default_value=default_value,
        return_batch_objects=True,
    )

    dataset_with_window = PytorchBigWigDataset(
        regions_of_interest=merged_intervals,
        collection=bigwig_path,
        reference_genome_path=reference_genome_path,
        sequence_length=center_bin_to_predict * 2,
        center_bin_to_predict=center_bin_to_predict,
        window_size=window_size,
        batch_size=batch_size,
        batches_per_epoch=1,
        maximum_unknown_bases_fraction=0.1,
        first_n_files=3,
        custom_position_sampler=position_sampler,
        default_value=default_value,
        return_batch_objects=True,
    )

    print(dataset_with_window._dataset.bigwig_collection.bigwig_paths)

    for batch, batch_with_window in zip(dataset, dataset_with_window):
        print(batch)
        print(batch_with_window)
        print(batch.chromosomes)
        print(batch_with_window.chromosomes)
        print(batch.starts)
        print(batch_with_window.starts)
        print(batch.ends)
        print(batch_with_window.ends)
        # expected = batch.values.reshape(
        #     batch.values.shape[0], batch.values.shape[1], reduced_dim, window_size
        # )
        expected = batch.values.reshape(
            batch.values.shape[0],
            reduced_dim,
            window_size,
            batch.values.shape[-1],
        )
        if not isnan(default_value) or default_value == 0:
            expected = torch.nan_to_num(expected, nan=default_value)
        expected = torch.nanmean(expected, axis=-2)
        # print("---")
        # print("expected")
        # print(expected)
        # print("batch_with_window")
        # print(batch_with_window.values)
        assert torch.allclose(expected, batch_with_window.values, equal_nan=True)

        # TODO: I need the bigwig file with empty intervals
        if isnan(default_value):
            assert torch.isnan(batch_with_window.values).any()
