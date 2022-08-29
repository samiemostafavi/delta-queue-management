import json

from .core import run_core


def add_deepq_params(params):

    # get parameters
    with open(params["project_path"] + "codel_tune_results.json") as info_json_file:
        info = json.load(info_json_file)

    for entry in info:
        if entry["quantile_key"] == params["quantile_key"]:
            # call by reference
            params["delay_ref"] = entry["target"]


def train_deepq(params, return_dict, module_label: str):

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
    queue = DQNQueue(
        name="queue",
        service_rp=service,
        t_interval=params["interval"],
        delta=params["delta"],
        delay_ref=params["delay_ref"],
    )

    return run_core(params, return_dict, queue, module_label)
