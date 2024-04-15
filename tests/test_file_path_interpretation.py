import pytest

from bigwig_loader.path import interpret_path


@pytest.fixture(scope="session")
def directory_structure(tmp_path_factory):
    path = tmp_path_factory.mktemp("root_one")
    sub_dir_one = path / "one"
    sub_dir_two = path / "two"
    sub_dir_one.mkdir()
    sub_dir_two.mkdir()
    (path / "heart.bw").write_text("")
    (sub_dir_one / "kidney.bw").write_text("")
    (sub_dir_one / "lung.bw").write_text("")
    (sub_dir_two / "eye.bigWig").write_text("")
    (sub_dir_two / "misspelled.biggwigg").write_text("")
    return path


def test_interpret_path_string(directory_structure):
    assert set(interpret_path(str(directory_structure))) == {
        directory_structure / "heart.bw",
        directory_structure / "one" / "kidney.bw",
        directory_structure / "one" / "lung.bw",
        directory_structure / "two" / "eye.bigWig",
    }


def test_interpret_path(directory_structure):
    assert set(interpret_path(directory_structure)) == {
        directory_structure / "heart.bw",
        directory_structure / "one" / "kidney.bw",
        directory_structure / "one" / "lung.bw",
        directory_structure / "two" / "eye.bigWig",
    }


def test_interpret_path_no_walk(directory_structure):
    assert interpret_path(directory_structure, crawl=False) == [
        directory_structure / "heart.bw"
    ]


def test_interpret_path_custom_extension(directory_structure):
    assert interpret_path(directory_structure, file_extensions=(".biggwigg",)) == [
        directory_structure / "two" / "misspelled.biggwigg",
    ]


def test_interpret_sequence_of_pathlib(directory_structure):
    assert set(
        interpret_path(
            [
                directory_structure / "heart.bw",
                directory_structure / "one",
                directory_structure / "two",
            ]
        )
    ) == {
        directory_structure / "heart.bw",
        directory_structure / "one" / "kidney.bw",
        directory_structure / "one" / "lung.bw",
        directory_structure / "two" / "eye.bigWig",
    }


def test_interpret_sequence_of_pathlib_2(directory_structure):
    assert set(
        interpret_path([directory_structure / "one", directory_structure / "two"])
    ) == {
        directory_structure / "one" / "kidney.bw",
        directory_structure / "one" / "lung.bw",
        directory_structure / "two" / "eye.bigWig",
    }
