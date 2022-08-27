import json
import os
import time
from functools import partial

import numpy as np
import polars as pl
from loguru import logger

from .core import run_core


def tune_codel(params, return_dict, module_label: str):

    from scipy.optimize import Bounds, minimize

    # define optimization function
    opt_func = partial(
        run_codel,
        params=params,
        return_dict=return_dict,
        module_label=module_label,
    )

    # define constraints
    # e.g. define constraints 0 <= x1 <= 1.0 and -0.5 <= x2 <= 2.0
    # bounds = Bounds([0, -0.5], [1.0, 2.0])
    bounds = Bounds(
        [params["interval_bounds"][0], params["target_bounds"][0]],
        [params["interval_bounds"][1], params["target_bounds"][1]],
    )

    logger.info(f"{params['run_number']}: Tuning started")

    x0 = np.array([params["interval_initial"], params["target_initial"]])
    # res = minimize(
    #    opt_func,
    #    x0,
    #    method='trust-constr',
    #    options={'verbose': 1},
    #    bounds=bounds
    # )
    res = minimize(
        opt_func,
        x0,
        method="nelder-mead",
        options={"xatol": 1e-8, "disp": True},
        bounds=bounds,
    )
    logger.info(
        f"{params['run_number']}: Tuning finished."
        + f" interval={res.x[0]}, target={res.x[1]}"
    )
    return_dict[params["run_number"]]["interval"] = res.x[0]
    return_dict[params["run_number"]]["target"] = res.x[1]

    records_path = params["records_path"] + module_label + "/"
    os.makedirs(records_path, exist_ok=True)
    resultjson = json.dumps(return_dict[params["run_number"]].copy())
    with open(
        records_path + f"{params['run_number']}_{module_label}_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(resultjson)


def run_codel(x, params, return_dict, module_label: str):

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
        be_quiet=True,
    )
    queue = CodelQueue(
        name="queue",
        service_rp=service,
        interval=x[0],
        target=x[1],
    )

    return run_core(params, return_dict, queue, module_label)
