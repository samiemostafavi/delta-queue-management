import json
import os

from loguru import logger

from .core import run_core


def add_deepq_params(params):

    # call by ref

    path = params["records_path"] + "deepq/"
    all_files = os.listdir(path)

    jsonfile = None
    zipfile = None
    for f in all_files:
        if f.endswith("queue.json"):
            jsonfile = path + f
        elif f.endswith("queue.zip"):
            zipfile = path + f

    assert jsonfile is not None
    assert zipfile is not None

    logger.info(f"{params['run_number']}: found DeepQ json and zip files.")

    params["model_addr_json"] = jsonfile
    params["model_addr_zip"] = zipfile


def run_deepq(params, return_dict, module_label: str):

    from qsimpy_aqm.dqn import DQNQueue
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

    # Opening JSON file
    with open(params["model_addr_json"]) as file:
        queue_json = json.load(file)

    queue = DQNQueue.parse_raw(json.dumps(queue_json))
    queue.service_rp = service
    queue.rl_model_address = params["model_addr_zip"]

    logger.info(f"{params['run_number']}: DeepQ reconstructed.")

    return run_core(params, return_dict, queue, module_label)
