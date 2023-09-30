"""
This script benchmarks both pyBigWig and bigwig-loader
by processing the same batches of intervals over and over.
There is also not anything else going on except loading
epigenetic profiles: no sampling, no loading genetic
sequence.

In this script pyBigWig is processing data in the main
process. A benchmark script with worker processes for
pyBigWig can be found in:

02_benchmark_minimal_pybigwig_multiple_workers.py

For a benchmark script that involves the entire loader
including loading randomly sampling intervals and loading
the genetic sequence see

03_benchmark_entire_loader.py
04_benchmark_entire_loader_pybigwig.py
05_benchmark_entire_loader_pybigwig_multiple_workers.py

Using bigwig-loader:
1 0.057052651047706605 0.00031709791153868065
2 0.057504883874207734 0.00017417943769643463
4 0.05825563352555037 0.00019529730686256182
8 0.06154786795377731 0.0003019231155274113
16 0.06676486572250724 0.00022556050291397823
32 0.07972694663330912 0.0076502898637687425
64 0.09993992112576962 0.00027788778186448623
128 0.1499797949567437 0.00045689017021743373
256 0.2352875482290983 0.0023178531225548945
384 0.27881404412910343 0.0018045323118200274
512 0.3368280011229217 0.0006013540903082275
640 0.4124560124240816 0.002849675913360234
768 0.4759079207666218 0.01028077048736988
896 0.5377236870117486 0.0015763286466726138
1024 0.5991766246035695 0.0009644582857066636
1152 0.6788586993701756 0.0019231941816597235
1280 0.7360170400701463 0.0013234154183372115
1408 0.7953401518054306 0.0021116091192682047
1536 0.8512347645126284 0.0014503305936706486
1664 0.9066219345666469 0.0011453341982749716
1792 0.9620565695688128 0.0014472743645979804
1920 1.0174510131590069 0.001081436407676258
2048 1.074055849108845 0.0019284532899694303
2176 1.129074206855148 0.0026376567781361444
2304 1.1839054396376014 0.001356973514175882
2432 1.228180939052254 0.001749555462204235
2560 1.2815975906327366 0.001366976399215406
2688 1.333120484650135 0.002277772508031197
2816 1.3867404093034565 0.002437589407029183
2944 1.4343125496990978 0.002801285683084347

Using pybigwig:
1 0.007712579797953367 1.3239532677819585e-05
2 0.015228871442377567 8.263952207858359e-05
4 0.030153942853212358 0.00019700686679982924
8 0.060477635357528925 0.0002358331421623131
16 0.12558735040947794 0.00035793705119275106
32 0.25060572251677515 0.0009981795971422773
64 0.4963857516646385 0.0015567211998296923
128 0.9931910111568868 0.006337946291953
256 1.9887389127165078 0.003353472583919708
384 2.978631450049579 0.004801139675230364
512 3.9880202571861445 0.009407197750658897
640 4.9852978646755215 0.00611967283515103
768 5.983420168235898 0.009951637939808588
896 7.002280553430319 0.03586572903432073
1024 7.970334761962294 0.007562513363487385
1152 8.959011241234839 0.007723535605142787
1280 9.959012242592872 0.010164891096016896
1408 10.94503095811233 0.009721375058046907
1536 11.942723208200187 0.013671800221479928
1664 12.922414517775177 0.01037941394057453
1792 13.911210288107394 0.011497199485000838
1920 14.922799918893725 0.019355535832356417
2048 15.89705225303769 0.016396645425789023
2176 16.869348569959403 0.01744892580180062
2304 17.84538498856127 0.017183208089130094
2432 18.836720233224334 0.01547382450705791
2560 19.837676095962525 0.016545546754101328
2688 20.812969361618162 0.017382408372612606
2816 21.805524553731082 0.019967367162665957
2944 22.810655508097263 0.026718328948077626

"""

import time

import cupy as cp
import numpy as np
import pandas as pd
import pyBigWig

from bigwig_loader import config
from bigwig_loader.collection import BigWigCollection
from bigwig_loader.collection import interpret_path


class PyBigWigCollection:
    def __init__(self, bigwig_path, first_n_files=None):
        bigwig_paths = sorted(interpret_path(bigwig_path))
        self.bigwigs = [pyBigWig.open(str(bw)) for bw in bigwig_paths][:first_n_files]

    def get_batch(self, chromosomes, starts, ends):
        out = np.stack(
            [
                bw.values(chrom, start, end, numpy=True)
                for chrom, start, end in zip(chromosomes, starts, ends)
                for bw in self.bigwigs
            ]
        )

        out = out.reshape(
            (out.shape[0] // len(self.bigwigs), len(self.bigwigs), out.shape[1])
        )

        return cp.asarray(out)


def some_intervals():
    df = pd.read_csv("example_data/benchmark_positions.tsv", sep="\t")
    start = df["center"].values - 500
    return list(df["chr"]), start, start + 1000


def run_benchmark(collection, batch_sizes, chromosomes, starts, ends):
    for batch_size in batch_sizes:
        # print("batch_size", batch_size)

        chrom = chromosomes[:batch_size]
        start = starts[:batch_size]
        end = ends[:batch_size]

        # burn in
        for _ in range(20):
            collection.get_batch(chrom, start, end)

        elapsed = []
        start_time = time.perf_counter()
        for _ in range(20):
            collection.get_batch(chrom, start, end)
            end_time = time.perf_counter()
            elapsed.append(end_time - start_time)
            start_time = end_time
        # print("Seconds per batch:", np.mean(elapsed))
        # print("standard deviation:", np.std(elapsed))
        print(batch_size, np.mean(elapsed), np.std(elapsed))


def run_all_benchmarks(bigwig_path=config.bigwig_dir):
    print("Loading from:", config.bigwig_dir)
    pybigwig_collection = PyBigWigCollection(bigwig_path, first_n_files=None)
    bigwig_loader_collection = BigWigCollection(bigwig_path, first_n_files=None)
    chromosomes, starts, ends = some_intervals()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64] + [128 * i for i in range(1, 48)]

    print("Using bigwig-loader:")
    run_benchmark(bigwig_loader_collection, batch_sizes, chromosomes, starts, ends)

    print("Using pybigwig:")
    run_benchmark(pybigwig_collection, batch_sizes, chromosomes, starts, ends)


if __name__ == "__main__":
    run_all_benchmarks()
