import os
from pathlib import Path

from loguru import logger

from .core import run_core

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def add_delta_params(params):

    # call by ref
    params["predictor_addr_h5"] = (
        params["main_path"] + "predictors/" + params["predictor_type"] + "/model.h5"
    )
    params["predictor_addr_json"] = (
        params["main_path"] + "predictors/" + params["predictor_type"] + "/model.json"
    )


def run_newdelta(params, return_dict, module_label: str):

    from qsimpy_aqm.delta import PredictorAddresses
    from qsimpy_aqm.newdelta import Horizon, NewDeltaQueue
    from qsimpy_aqm.random import HeavyTailGamma

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=params["service_seed"],
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=0.2,
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=params["arrivals_number"],
    )
    queue = NewDeltaQueue(
        name="queue",
        service_rp=service,
        predictor_addresses=PredictorAddresses(
            h5_address=params["predictor_addr_h5"],
            json_address=params["predictor_addr_json"],
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

    run_core(params, return_dict, queue, module_label)
