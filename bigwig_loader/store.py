from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
from kvikio.cufile import CuFile


class BigWigStore:
    def __init__(
        self,
        path: Union[str, Path],
        chunk_offsets: npt.NDArray[np.int64],
        chunk_sizes: npt.NDArray[np.int64],
        use_cufile: bool = True,
    ) -> None:
        self.path = path
        if use_cufile:
            self.file_handle = CuFile(path, flags="r")
        else:
            self.file_handle = open(path, "rb", buffering=0)
        self.chunk_offsets = chunk_offsets
        self.chunk_sizes = chunk_sizes

    def get_offsets_and_sizes(
        self, keys: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        return self.chunk_offsets[keys], self.chunk_sizes[keys]
