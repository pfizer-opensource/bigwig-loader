from typing import Generator

import pandas as pd

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.streamed_dataset import StreamedDataloader


def main() -> None:
    bigwig_path = config.bigwig_dir

    collection = BigWigCollection(bigwig_path, first_n_files=2, use_cufile=True)

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]

    # def input_generator():
    #     for i in range(1000):
    #         batch = df.sample(10)
    #         chrom, start, end = (
    #             list(batch["chr"]),
    #             list(batch["center"] - 500),
    #             list(batch["center"] + 500),
    #         )
    #
    #         yield (chrom, start, end)
    def input_generator() -> (
        Generator[tuple[list[str], list[int], list[int]], None, None]
    ):
        chrom, start, end = (
            list(df["chr"]),
            list(df["center"] - 500),
            list(df["center"] + 500),
        )

        for i in range(1000):
            yield (chrom[:10], start[:10], end[:10])

    data_loader = StreamedDataloader(
        input_generator=input_generator(),
        collection=collection,
        num_threads=4,
        queue_size=10,
    )

    for i, batch in enumerate(data_loader):
        print(i, batch)
        if i == 10:
            data_loader.stop()
            break


if __name__ == "__main__":
    main()
