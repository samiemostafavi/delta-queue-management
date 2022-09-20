import json
import os
import time

import numpy as np
import polars as pl
from loguru import logger

from .core import run_core


def add_offlineoptimum_params(params):
    pass


def run_offlineoptimum(params, return_dict, module_label: str):

    from qsimpy_aqm.newdelta import Horizon
    from qsimpy_aqm.oo import OfflineOptimumQueue
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

    run_core(params, return_dict, queue, module_label)
