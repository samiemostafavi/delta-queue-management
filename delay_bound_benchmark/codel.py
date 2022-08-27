import json
import os
import time

import numpy as np
import polars as pl
from loguru import logger

from .core import run_core


def add_codel_params(params):

    # get parameters
    with open(params["project_path"] + "codel_tune_results.json") as info_json_file:
        info = json.load(info_json_file)

    for entry in info:
        if entry["quantile_key"] == params["quantile_key"]:
            # call by reference
            params["codel_interval"] = entry["interval"]
            params["codel_target"] = entry["target"]


def run_codel(params, return_dict, module_label: str):

    from qsimpy_aqm.codel import CodelQueue
    from qsimpy_aqm.random import HeavyTailGamma

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=params["service_seed"],
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=0.1,
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=params["arrivals_number"],
    )
    queue = CodelQueue(
        name="queue",
        service_rp=service,
        interval=params["codel_interval"],
        target=params["codel_target"],
    )

    run_core(params, return_dict, queue, module_label)
