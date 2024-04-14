from pathlib import Path

from bigwig_loader.path import subpaths


def test_subpaths():
    example = Path("/home/jim/projects/my_dir/text_file.txt")
    assert list(subpaths(example)) == [
        Path("/home/jim/projects/my_dir/text_file.txt"),
        Path("/home/jim/projects/my_dir/text_file"),
        Path("jim/projects/my_dir/text_file.txt"),
        Path("jim/projects/my_dir/text_file"),
        Path("projects/my_dir/text_file.txt"),
        Path("projects/my_dir/text_file"),
        Path("my_dir/text_file.txt"),
        Path("my_dir/text_file"),
        Path("text_file.txt"),
        Path("text_file"),
    ]
