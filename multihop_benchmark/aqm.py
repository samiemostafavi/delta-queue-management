import os
from pathlib import Path

from loguru import logger

from .core import run_core

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_aqm(params):

    from qsimpy_aqm.delta import PredictorAddresses
    from qsimpy_aqm.newdelta import Horizon, NewDeltaQueue
    from qsimpy_aqm.oo import OfflineOptimumQueue
    from qsimpy_aqm.random import HeavyTailGamma

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=params["service_seed"],
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=params["gpd_concentration"],
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=params["arrivals_number"],
    )
    if params["module"][0] == "offline-optimum":
        queue = OfflineOptimumQueue(
            name="queue",
            service_rp=service,
            horizon=Horizon(
                max_length=10,
                min_length=None,
                arrival_rate=None,
            ),
            debug_all=False,
            debug_drops=False,
        )
    elif params["module"][1] == "gmm" or params["module"][1] == "gmevm":
        queue = NewDeltaQueue(
            name="queue",
            service_rp=service,
            predictor_addresses=PredictorAddresses(
                h5_address=params["predictor_path_h5"],
                json_address=params["predictor_path_json"],
            ),
            horizon=Horizon(
                max_length=10,
                min_length=None,
                arrival_rate=None,
            ),
            limit_drops=[0, 1, 2, 3],
            gradient_check=True,
            debug_drops=False,
            do_not_drop=False,
        )
    else:
        raise ("unknown predictor or AQM")

    df, results = run_core(params, queue)
    return results
