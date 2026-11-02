"""Forest Cover Type Classification package.

The thread limit is set here, before numpy / scikit-learn / LightGBM are
imported: on macOS scikit-learn and LightGBM ship separate OpenMP
runtimes, and with all cores busy their spinning threads slow
HistGradientBoosting down by an order of magnitude. Override with
FCT_THREADS=<n>.
"""

import os

THREADS = int(os.environ.get("FCT_THREADS", "4"))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(THREADS))
