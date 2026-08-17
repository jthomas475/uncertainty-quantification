import numpy as np
import os
from contextlib import contextmanager


def safe_save(path, data):
    if not path.endswith(".npy"):
        path += ".npy"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if os.path.exists(path):
        raise FileExistsError(
            f"file '{path}' already exists"
        )
    np.save(path, data)
    return path

@contextmanager
def quiet_func():
    devnull = os.open(os.devnull, flags=os.O_WRONLY)

    saved_out = os.dup(1) # save to standard output (stdout)
    saved_err = os.dup(2) # save to standard error (stderr) (note that dup(0) saves to standard input (stdin))

    os.dup2(devnull, 1)
    os.dup2(devnull,2)

    try:
        yield

    finally:
        os.dup2(saved_out,1)
        os.dup2(saved_err,2)

        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)
