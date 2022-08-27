import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from loguru import logger

from .core import run_noaqm_core


def add_noaqm_params(params):
    pass


def run_noaqm(params, return_dict, module_label: str):

    from qsimpy.simplequeue import SimpleQueue
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
    queue = SimpleQueue(
        name="queue",
        service_rp=service,
        # queue_limit=10, #None
    )

    df = run_noaqm_core(params, return_dict, queue, module_label)

    df = df.select([pl.col("end2end_delay")])
    res = df.quantile(
        quantile=params["quantile_key"],
        interpolation="nearest",
    )
    return_dict[params["run_number"]]["quantile_key"] = params["quantile_key"]
    return_dict[params["run_number"]]["quantile_value"] = res[0, 0]
    return_dict[params["run_number"]]["until"] = params["until"]

    records_path = params["records_path"] + module_label + "/"
    os.makedirs(records_path, exist_ok=True)

    resultjson = json.dumps(return_dict[params["run_number"]].copy())
    with open(
        records_path + f"{params['run_number']}_noaqm_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(resultjson)
