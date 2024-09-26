import pandas as pd
import torch
from torch.utils.data import DataLoader

from bigwig_loader import config
from bigwig_loader.download_example_data import download_example_data
from bigwig_loader.pytorch import PytorchBigWigDataset

download_example_data()
example_bigwigs_directory = config.bigwig_dir
reference_genome_file = config.reference_genome

train_regions = pd.DataFrame(
    {"chrom": ["chr1", "chr2"], "start": [0, 0], "end": [1000000, 1000000]}
)

dataset = PytorchBigWigDataset(
    regions_of_interest=train_regions,
    collection=example_bigwigs_directory,
    reference_genome_path=reference_genome_file,
    sequence_length=1000,
    center_bin_to_predict=500,
    window_size=1,
    batch_size=32,
    super_batch_size=1024,
    batches_per_epoch=20,
    maximum_unknown_bases_fraction=0.1,
    sequence_encoder="onehot",
    n_threads=4,
    return_batch_objects=True,
)

dataloader = DataLoader(dataset, num_workers=0, batch_size=None)


class MyTerribleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, batch):
        return self.linear(batch).transpose(1, 2)


model = MyTerribleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def poisson_loss(pred, target):
    return (pred - target * torch.log(pred.clamp(min=1e-8))).mean()


for batch in dataloader:
    # batch.sequences.shape = n_batch (32), sequence_length (1000), onehot encoding (4)
    pred = model(batch.sequences)
    # batch.values.shape = n_batch (32), n_tracks (2) center_bin_to_predict (500)
    loss = poisson_loss(pred[:, :, 250:750], batch.values)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
