from agnprocesses.ext.pgamma import pgamma
from agnprocesses.ext.bh import bh
from agnprocesses.ext.ggir import rate
from agnprocesses.ext.ggpp import pair


def test_extensions_import():
    for ext_fun in [pgamma, bh, rate, pair]:
        print(ext_fun.__doc__)
