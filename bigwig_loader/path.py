import os
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Union

from upath import UPath


def get_bigwig_files_from_path(
    path: Path, file_extensions: Iterable[str] = (".bigWig", ".bw"), crawl: bool = True
) -> list[Path]:
    """
    For a path object, get all bigwig files. If the path is directly pointing
    to a bigwig file, a list with just that file is returned. In case of a
    directory all files are gathered with file extensions that are part of
    file_extensions.
    Args:
        path: or upath.Path object. Either directory or BigWig file
        file_extensions: used to find all BigWig files in a directory.
            Default: (".bigWig", ".bw")
        crawl: whether to find BigWig files in subdirectories. Default: True.

    Returns: list of paths to BigWig files.

    """
    if not path.exists():
        raise FileNotFoundError(f"No such file or directory: {path}")
    elif path.is_dir():
        if crawl:
            pattern = "**/*"
        else:
            pattern = "*"
        return [
            file
            for extension in file_extensions
            for file in path.glob(f"{pattern}{extension}")
        ]
    return [path]


def interpret_path(
    bigwig_path: Union[
        Union[str, "os.PathLike[Any]"], Iterable[Union[str, "os.PathLike[Any]"]]
    ],
    file_extensions: Iterable[str] = (".bigWig", ".bw"),
    crawl: bool = True,
) -> list[Path]:
    """
    Get all bigwig files for a path. Also excepts strings.
    If the path is directly pointing to a bigwig file, a list with just that
    file is returned. In case of a directory all files are gathered with file
    extensions that are part of file_extensions.
    Args:
        bigwig_path: str, Path or upath.Path object. Either directory or BigWig file
        file_extensions: used to find all BigWig files in a directory.
            Default: (".bigWig", ".bw")
        crawl: whether to find BigWig files in subdirectories. Default: True.

    Returns: list of paths to BigWig files.

    """
    if isinstance(bigwig_path, str) or isinstance(bigwig_path, os.PathLike):
        return get_bigwig_files_from_path(
            UPath(bigwig_path), file_extensions=file_extensions, crawl=crawl
        )

    elif isinstance(bigwig_path, Iterable):
        return [
            path
            for element in bigwig_path
            for path in interpret_path(
                element, file_extensions=file_extensions, crawl=crawl
            )
        ]
    raise ValueError(
        f"Can not interpret {bigwig_path} as path or a collection of paths."
    )


def subpaths(path: Path, depth: Optional[int] = None) -> Iterable[Path]:
    """Iteratively breaks up a path like
    /home/jim/bigwig_files/liver/ENCFF1234.bigWig into
    subpaths with less and parent directories. After this is done
    to the full path, the same thing will be done to the parent of
    path.
    Example:
        $ subpaths(Path("/home/jim/bigwig_files/liver/ENCFF1234.bigWig"), depth=3)
        bigwig_files/liver/ENCFF1234.bigWig
        bigwig_files/liver/ENCFF1234
        liver/ENCFF1234.bigWig
        liver/ENCFF1234
        ENCFF1234.bigWig
        ENCFF1234
        bigwig_files/liver
        liver
    ```
    """
    parts = path.parts
    stemmed = parts[:-1] + (path.stem,)
    if depth is not None:
        parts = parts[-depth:]
        stemmed = stemmed[-depth:]
    for i in range(len(parts) - 1):
        sub_base = parts[: len(parts) - i]
        sub_stemmed = stemmed[: len(stemmed) - i]
        for j in range(len(sub_base)):
            yield Path(*sub_base[j:])
            if sub_stemmed != sub_base:
                yield Path(*sub_stemmed[j:])


def match_key_to_path(path: Path, keys: Iterable[str | Path]) -> Optional[str | Path]:
    """
    Find  the key in keys that best describes the Path. Direct matches
    like file_name.bigWig to file_name.bigWig are the simplest case. After
    that also matches like file_name to file_name.bigWig are considered.
    When a longer path is, like /home/jim/bigwig_files/liver/ENCFF1234.bigWig,
    first it's checked whether the full path is in keys. If not, it continues to
    check whether jim/bigwig_files/liver/ENCFF1234.bigWig is in keys. After that
    bigwig_files/liver/ENCFF1234.bigWig etc, until the path is just the file name/stem.
    When there is still no match, we try the same procedure with the parent directory
    as input. This is handy is you want to associate all liver bigwig files with the
    scaling factor for liver for instance.

    Args:
        path: Path object
        keys: Iterable of strings or Path objects

    Returns: key that best matches the path

    """
    if not keys:
        return None
    for subpath in subpaths(path):
        if str(subpath) in keys:
            return str(subpath)
        if subpath in keys:
            return subpath
    return None


def map_path_to_value(
    path: Path, value_dict: dict[Union[str | Path], Any], default: Any = None
) -> Any:
    key = match_key_to_path(path=path, keys=value_dict.keys())
    if key is None:
        return default
    return value_dict[key]


if __name__ == "__main__":
    example = Path("/home/jim/bigwig_files/liver/ENCFF1234.bigWig")
    for p in subpaths(example):
        print(p)
    # print(list(subpaths(example)))
