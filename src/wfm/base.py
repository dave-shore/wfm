from torch import float16, set_default_dtype as torch_set_default_dtype, set_num_threads as torch_set_num_threads, get_default_dtype, finfo
from os import cpu_count

ALLOWED_LIBRARIES = [
    "numpy",
    "torch",
    "jax"
]

BASE_BATCH_SIZE = 256

torch_set_default_dtype(float16)
torch_set_num_threads(cpu_count() * 2 // 5)

EPS = finfo(get_default_dtype()).eps
UB = finfo(get_default_dtype()).max