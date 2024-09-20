import gzip
import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import BinaryIO

from bigwig_loader import config
from bigwig_loader.path import get_bigwig_files_from_path

LOGGER = logging.getLogger(__name__)


def download_example_data() -> None:
    get_reference_genome()
    get_example_bigwigs_files()


def get_reference_genome(reference_genome_path: Path = config.reference_genome) -> Path:
    compressed_file = reference_genome_path.with_suffix(".fasta.gz")
    if reference_genome_path.exists():
        return reference_genome_path
    elif compressed_file.exists():
        # subprocess.run(["bgzip", "-d", compressed_file])
        unzip_gz_file(compressed_file, reference_genome_path)
    else:
        LOGGER.info("Need reference genome for tests. Downloading it from ENCODE.")
        url = "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
        urllib.request.urlretrieve(url, compressed_file)
        # subprocess.run(["bgzip", "-d", compressed_file])
        unzip_gz_file(compressed_file, reference_genome_path)
    return reference_genome_path


def unzip_gz_file(compressed_file_path: Path, output_file_path: Path) -> Path:
    with gzip.open(compressed_file_path, "rb") as gz_file:
        with open(output_file_path, "wb") as output_file:
            shutil.copyfileobj(gz_file, output_file)
    return output_file_path


EXAMPLE_FILES = {
    "ENCFF270YCY.bigWig": (
        "https://www.encodeproject.org/files/ENCFF270YCY/@@download/ENCFF270YCY.bigWig",
        "64886b78f4cd70531fceffbaeafd4adb",
    ),
    "ENCFF270ISQ.bigWig": (
        "https://www.encodeproject.org/files/ENCFF270ISQ/@@download/ENCFF270ISQ.bigWig",
        "5fef60f9f1e43b9a17075c352650865e",
    ),
}


def checksum_md5(f: BinaryIO, *, chunk_size: int = 10 * 1024 * 1024) -> str:
    """return the md5sum"""
    m = hashlib.md5(b"", usedforsecurity=False)
    for data in iter(lambda: f.read(chunk_size), b""):
        m.update(data)
    return m.hexdigest()


def get_example_bigwigs_files(bigwig_dir: Path = config.bigwig_dir) -> Path:
    bigwig_dir.mkdir(parents=True, exist_ok=True)
    available_files = [pth.name for pth in get_bigwig_files_from_path(bigwig_dir)]
    if len(available_files) < 2:
        for fn, (url, md5) in EXAMPLE_FILES.items():
            file = bigwig_dir / fn
            if not file.exists():
                urllib.request.urlretrieve(url, file)
            with file.open(mode="rb") as f:
                if checksum_md5(f) != md5:
                    raise RuntimeError(f"{fn} has incorrect checksum!")
    return bigwig_dir
