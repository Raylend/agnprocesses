# file io is a temporary solution for passing data between C extensions and python code

from pathlib import Path

from typing import Tuple, Iterable


CUR_DIR = Path(__file__).parent

DATA_DIR = CUR_DIR / 'data'


def get_io_paths(name: str, subdir_names: Iterable[str] = ('input', 'output')) -> Tuple[Path, Path]:
    data_files_dir = CUR_DIR / name
    ext_subdirs = [data_files_dir / subdir for subdir in subdir_names]

    for dir in [data_files_dir, *ext_subdirs]:
        dir.mkdir(exist_ok=True)
    
    return tuple(ext_subdirs)