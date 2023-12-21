import pytest

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
        assert target.shape == (256, 2, 1000)


def test_input_and_target_is_torch_tensor(pytorch_dataset):
    sequence, target = next(iter(pytorch_dataset))
    assert isinstance(sequence, torch.Tensor)
    assert isinstance(target, torch.Tensor)
