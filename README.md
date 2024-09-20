# Dataloader for BigWig files

Faster batched dataloading of BigWig files and corresponding sequence data powered by GPU for deep learning applications.
This library is meant for loading batches of data with the same dimensionality, which allows for some assumptions that can
speed up the loading process. As can be seen from the plot below, when loading a small amount of data, pyBigWig is very fast,
but does not exploit the batched nature of data loading for machine learning.

In the benchmark below we also created PyTorch dataloaders (with set_start_method('spawn')) using pyBigWig to compare to
the realistic scenario where multiple CPUs would be used per GPU. We see that the throughput of the CPU dataloader does
not go up linearly with the number of CPUs, and therefore it becomes hard to get the needed throughput to keep the GPU,
training the neural network,saturated during the learning steps.


![benchmark.png](images%2Fbenchmark.png)

This is the problem bigwig-loader solves. This is an example of how to use bigwig-loader:

```python
import pandas as pd
from bigwig_loader.dataset import BigWigDataset
from bigwig_loader import config
from bigwig_loader.download_example_data import download_example_data

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
    window_size=1,
    batch_size=256,
    super_batch_size=1024,
    batches_per_epoch=20,
    maximum_unknown_bases_fraction=0.1,
    sequence_encoder="onehot",
)

for encoded_sequences, epigenetics_profiles in dataset:
    print(encoded_sequences)
    print(epigenetics_profiles)

```

See the examples directory for more examples.

### Installation

1. `git clone git@github.com:pfizer-opensource/bigwig-loader`
2. `cd bigwig-loader`
3. create the conda environment" `conda env create -f environment.yml`

In this environment you should be able to run `pytest -v` and see the tests
succeed. NOTE: you need a GPU to use bigwig-loader!

## Development

This section guides you through the steps needed to add new functionality. If
anything is unclear, please open an issue.

### Environment

1. `git clone git@github.com:pfizer-opensource/bigwig-loader`
2. `cd bigwig-loader`
3. create the conda environment" `conda env create -f environment.yml`
4. `pip install -e '.[dev]'`
5. run `pre-commit install` to install the pre-commit hooks


## Citing

If you use this library, consider citing:

Retel, Joren Sebastian, Andreas Poehlmann, Josh Chiou, Andreas Steffen, and Djork-Arné Clevert. “A Fast Machine Learning Dataloader for Epigenetic Tracks from BigWig Files.” Bioinformatics 40, no. 1 (January 1, 2024): btad767. https://doi.org/10.1093/bioinformatics/btad767.

```bibtex
@article{
    retel_fast_2024,
    title = {A fast machine learning dataloader for epigenetic tracks from {BigWig} files},
    volume = {40},
    issn = {1367-4811},
    url = {https://doi.org/10.1093/bioinformatics/btad767},
    doi = {10.1093/bioinformatics/btad767},
    abstract = {We created bigwig-loader, a data-loader for epigenetic profiles from BigWig files that decompresses and processes information for multiple intervals from multiple BigWig files in parallel. This is an access pattern needed to create training batches for typical machine learning models on epigenetics data. Using a new codec, the decompression can be done on a graphical processing unit (GPU) making it fast enough to create the training batches during training, mitigating the need for saving preprocessed training examples to disk.The bigwig-loader installation instructions and source code can be accessed at https://github.com/pfizer-opensource/bigwig-loader},
    number = {1},
    urldate = {2024-02-02},
    journal = {Bioinformatics},
    author = {Retel, Joren Sebastian and Poehlmann, Andreas and Chiou, Josh and Steffen, Andreas and Clevert, Djork-Arné},
    month = jan,
    year = {2024},
    pages = {btad767},
}
```
