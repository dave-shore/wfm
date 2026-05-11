from torch import float16, set_default_dtype as torch_set_default_dtype

ALLOWED_LIBRARIES = [
    "numpy",
    "torch",
    "jax"
]

EPS = 1e-6
UB = 1e6
BASE_BATCH_SIZE = 512

torch_set_default_dtype(float16)