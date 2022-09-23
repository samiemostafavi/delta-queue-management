import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from loguru import logger

from .core import run_core


def run_noaqm(params):

    from qsimpy.simplequeue import SimpleQueue
    from qsimpy_aqm.random import HeavyTailGamma

    results = {}
    results["run_number"] = params["run_number"]

    # results are call by ref
    if params["run_noaqm"]:

        logger.info(f"{params['run_number']}: Running noaqm")

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
        queue = SimpleQueue(
            name="queue",
            service_rp=service,
            # queue_limit=10, #None
        )

        # make a copy of params for noaqm
        noaqm_params = params.copy()
        noaqm_params["module_label"] = "noaqm"
        df, misc = run_core(noaqm_params, queue)

        df = df.select([pl.col("end2end_delay")])
        res = df.quantile(
            quantile=params["quantile_key"],
            interpolation="nearest",
        )

        results["misc"] = misc
        results["quantile_key"] = noaqm_params["quantile_key"]
        results["quantile_value"] = res[0, 0]
        results["until"] = noaqm_params["until"]

        records_path = noaqm_params["records_path"] + noaqm_params["module_label"] + "/"
        os.makedirs(records_path, exist_ok=True)

        resultjson = json.dumps(results)
        with open(
            records_path + f"{noaqm_params['run_number']}_noaqm_result.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(resultjson)

        # results["delay_bound"] = results[params["run_number"]]["quantile_value"]

    else:

        logger.info(f"{params['run_number']}: Not running noaqm, reading results")
        noaqm_res_path = (
            params["records_path"]
            + "noaqm/"
            + str(params["run_number"])
            + "_noaqm_result.json"
        )

        # Opening result json file
        with open(noaqm_res_path) as json_file:
            noaqm_res = json.load(json_file)

        logger.info(
            f"Loaded noaqm {params['run_number']}: "
            + f"quantile_value={noaqm_res['quantile_value']}, "
        )

        results["misc"] = noaqm_res["misc"]
        results["quantile_key"] = params["quantile_key"]
        results["until"] = params["until"]
        results["quantile_value"] = noaqm_res["quantile_value"]
        # params["delay_bound"] = noaqm_res["quantile_value"]
        # logger.info(f"Set delay_bound={params['delay_bound']}")

    return results

    # logger.info(f"{params['run_number']}: Running {main_benchmark_name}")
    # main_benchmark(params, return_dict, main_benchmark_name)
