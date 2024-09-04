from typing import Generator

import pandas as pd

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.streamed_dataset import StreamedDataloader


def main() -> None:
    bigwig_path = config.bigwig_dir

    collection = BigWigCollection(bigwig_path, first_n_files=2)

    df = pd.read_csv(config.example_positions, sep="\t")
    df = df[df["chr"].isin(collection.get_chromosomes_present_in_all_files())]

    def input_generator() -> (
        Generator[tuple[list[str], list[int], list[int]], None, None]
    ):
        while True:
            batch = df.sample(20)
            batch = batch.sort_values(by=["chr", "center"])

            chrom, start, end = (
                list(batch["chr"]),
                list(batch["center"] - 500),
                list(batch["center"] + 500),
            )
            print(len(chrom), len(start), len(end))

            yield (chrom, start, end)

    data_loader = StreamedDataloader(
        input_generator=input_generator(),
        collection=collection,
        num_threads=4,
        queue_size=4,
        slice_size=3,
    )

    for i, output in enumerate(data_loader):
        print(i, output, flush=True)
        if i == 2000:
            data_loader.stop()
            break
    print("Done!")


if __name__ == "__main__":
    main()
