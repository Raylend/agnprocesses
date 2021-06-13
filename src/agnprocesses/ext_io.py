# file io is a temporary solution for passing data between C extensions and python code

from pathlib import Path

from typing import Tuple


CUR_DIR = Path(__file__).parent


def get_io_paths(name: str) -> Tuple[Path, Path]:
    ext_io_dir = CUR_DIR / name
    ext_input = ext_io_dir / "input"
    ext_output = ext_io_dir / "output"

    for dir in [ext_io_dir, ext_input, ext_output]:
        dir.mkdir(exist_ok=True)
    
    return ext_input, ext_output
