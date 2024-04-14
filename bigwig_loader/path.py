import os
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Union

from upath import UPath


def subpaths(path: Path) -> Iterable[Path]:
    parts = path.parts
    stemmed = parts[:-1] + (path.stem,)
    for i in range(len(parts)):
        yield Path(*parts[i:])
        yield Path(*stemmed[i:])


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


def match_key_to_path(path: Path, keys: Iterable[str | Path]) -> str | Path | None:
    for subpath in subpaths(path):
        if subpath in keys:
            return subpath
        elif str(subpath) in keys:
            return str(subpath)
        elif subpath.stem in keys:
            return subpath.stem
        elif str(subpath.stem) in keys:
            return str(subpath.stem)
    return None


def match_keys_to_paths(
    keys: list[str | Path], paths: list[Path], allow_unmatched_paths: bool = False
) -> dict[Path, str | Path | None]:
    """
    Matches keys to paths, starting with thw full path
    Args:
        keys:
        paths:
        allow_unmatched_paths:

    Returns:

    """
    mapping = {}
    for path in paths:
        key = match_key_to_path(path, keys)
        if key is None and not allow_unmatched_paths:
            raise ValueError(f"No key found for {path}")
        mapping[path] = key
    return mapping


if __name__ == "__main__":
    example = Path("/home/jim/projects/my_dir/text_file.txt")
    print(list(subpaths(example)))
