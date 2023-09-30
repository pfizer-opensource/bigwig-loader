import os

from dotenv import find_dotenv
from pydantic.functional_validators import BeforeValidator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from typing_extensions import Annotated
from upath import UPath

ENCODE_BASE_URL: str = "https://www.encodeproject.org/"


def string_to_path(v: str) -> UPath:
    return UPath(os.path.expandvars(v))


Path = Annotated[UPath, BeforeValidator(string_to_path)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="bigwig_loader_",
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    example_data_dir: Path = Path(__file__).parent.parent / "example_data"
    example_positions: Path = example_data_dir / "some_positions.tsv"
    reference_genome_url: str = (
        ENCODE_BASE_URL
        + "files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
    )
    reference_genome: Path = (
        example_data_dir / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
    )
    bigwig_dir: Path = example_data_dir / "bigwig"

    def __str__(self) -> str:
        return super().__str__().replace(" ", "\n")
