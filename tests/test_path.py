from pathlib import Path

from bigwig_loader.path import map_path_to_value
from bigwig_loader.path import subpaths


def test_subpaths():
    example = Path("/home/jim/bigwig_files/liver/ENCFF1234.bigWig")
    assert list(subpaths(example)) == [
        Path("/home/jim/bigwig_files/liver/ENCFF1234.bigWig"),
        Path("/home/jim/bigwig_files/liver/ENCFF1234"),
        Path("home/jim/bigwig_files/liver/ENCFF1234.bigWig"),
        Path("home/jim/bigwig_files/liver/ENCFF1234"),
        Path("jim/bigwig_files/liver/ENCFF1234.bigWig"),
        Path("jim/bigwig_files/liver/ENCFF1234"),
        Path("bigwig_files/liver/ENCFF1234.bigWig"),
        Path("bigwig_files/liver/ENCFF1234"),
        Path("liver/ENCFF1234.bigWig"),
        Path("liver/ENCFF1234"),
        Path("ENCFF1234.bigWig"),
        Path("ENCFF1234"),
        Path("/home/jim/bigwig_files/liver"),
        Path("home/jim/bigwig_files/liver"),
        Path("jim/bigwig_files/liver"),
        Path("bigwig_files/liver"),
        Path("liver"),
        Path("/home/jim/bigwig_files"),
        Path("home/jim/bigwig_files"),
        Path("jim/bigwig_files"),
        Path("bigwig_files"),
        Path("/home/jim"),
        Path("home/jim"),
        Path("jim"),
        Path("/home"),
        Path("home"),
    ]


def test_map_path_to_value_absolute_paths():
    value_dict = {
        "ENCFF1234": 1234,
        "ENCFF5678": 5678,
        "liver": 5000,
        "brain": 6000,
        "liver/specific.bigWig": 1111,
        "kidney/other": 2222,
        "liver/ENCFF4444": 4444,
        "brain/ENCFF4444": 5555,
    }
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/liver/ENCFF1234.bigWig"), value_dict
        )
        == 1234
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/brain/ENCFF1234.bigWig"), value_dict
        )
        == 1234
    )

    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/liver/ENCFF4444.bigWig"), value_dict
        )
        == 4444
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/brain/ENCFF4444.bigWig"), value_dict
        )
        == 5555
    )

    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/liver/ENCFF5678.bigWig"), value_dict
        )
        == 5678
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/liver/ENCFF5000.bigWig"), value_dict
        )
        == 5000
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/brain/brain_track.bigWig"), value_dict
        )
        == 6000
    )

    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/liver/ENCFF5000.bigWig"), value_dict, default=1
        )
        == 5000
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/liver/specific.bigWig"), value_dict, default=1
        )
        == 1111
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/kidney/other.bigWig"), value_dict, default=1
        )
        == 2222
    )

    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/kidney/ENCFF9101.bigWig"), value_dict
        )
        is None
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/kidney/ENCFF9101.bigWig"),
            value_dict,
            default=0,
        )
        == 0
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/kidney/ENCFF9101.bigWig"),
            value_dict,
            default=1,
        )
        == 1
    )
    assert (
        map_path_to_value(
            Path("/home/jim/bigwig_files/kidney/ENCFF9101.bigWig"),
            value_dict,
            default=None,
        )
        is None
    )


def test_map_path_to_value_relative_paths():
    value_dict = {
        "ENCFF1234": 1234,
        "ENCFF5678": 5678,
        "liver": 5000,
        "brain": 6000,
        "liver/specific.bigWig": 1111,
        "kidney/other": 2222,
    }
    assert (
        map_path_to_value(Path("bigwig_files/liver/ENCFF1234.bigWig"), value_dict)
        == 1234
    )
    assert (
        map_path_to_value(Path("bigwig_files/brain/ENCFF1234.bigWig"), value_dict)
        == 1234
    )

    assert (
        map_path_to_value(Path("bigwig_files/liver/ENCFF5678.bigWig"), value_dict)
        == 5678
    )
    assert (
        map_path_to_value(Path("bigwig_files/liver/ENCFF5000.bigWig"), value_dict)
        == 5000
    )
    assert (
        map_path_to_value(Path("bigwig_files/brain/brain_track.bigWig"), value_dict)
        == 6000
    )

    assert (
        map_path_to_value(
            Path("bigwig_files/liver/ENCFF5000.bigWig"), value_dict, default=1
        )
        == 5000
    )
    assert (
        map_path_to_value(
            Path("bigwig_files/liver/specific.bigWig"), value_dict, default=1
        )
        == 1111
    )
    assert (
        map_path_to_value(
            Path("bigwig_files/kidney/other.bigWig"), value_dict, default=1
        )
        == 2222
    )

    assert (
        map_path_to_value(Path("bigwig_files/kidney/ENCFF9101.bigWig"), value_dict)
        is None
    )
    assert (
        map_path_to_value(
            Path("bigwig_files/kidney/ENCFF9101.bigWig"), value_dict, default=0
        )
        == 0
    )
    assert (
        map_path_to_value(
            Path("bigwig_files/kidney/ENCFF9101.bigWig"), value_dict, default=1
        )
        == 1
    )
    assert (
        map_path_to_value(
            Path("bigwig_files/kidney/ENCFF9101.bigWig"), value_dict, default=None
        )
        is None
    )
