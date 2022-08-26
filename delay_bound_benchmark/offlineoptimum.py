import json
import os
import time

import numpy as np
import polars as pl
from loguru import logger


def run_offlineoptimum(params, return_dict):

    from qsimpy.core import Model, TimedSource
    from qsimpy.polar import PolarSink
    from qsimpy.random import Deterministic
    from qsimpy_aqm.newdelta import Horizon
    from qsimpy_aqm.oo import OfflineOptimumQueue
    from qsimpy_aqm.random import HeavyTailGamma

    records_path = params["records_path"] + "offline-optimum/"
    os.makedirs(records_path, exist_ok=True)

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Offline Optimum AQM benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate=params["arrival_rate"],  # 0.09 low or 0.095 high
        seed=params["arrival_seed"],
        dtype="float64",
    )
    delay_bount = return_dict[params["run_number"]]["quantile_value"]
    source = TimedSource(
        name="start-node",
        arrival_rp=arrival,
        task_type="0",
        delay_bound=delay_bount,
    )
    model.add_entity(source)

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
    model.add_entity(queue)

    # Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
    sink = PolarSink(
        name="sink",
        batch_size=10000,
    )
    # define postprocess function: the name must be 'user_fn'

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        df["service_delay"] = df["end_time"] - df["service_time"]
        df["queue_delay"] = df["service_time"] - df["queue_time"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    # Wire start-node, queue, end-node, and sink together
    source.out = queue.name
    queue.out = sink.name
    queue.drop = sink.name

    # Setup task records
    model.set_task_records(
        {
            "timestamps": {
                source.name: {
                    "task_generation": "start_time",
                },
                queue.name: {
                    "task_reception": "queue_time",
                    "service_start": "service_time",
                    "service_end": "end_time",
                },
            },
            "attributes": {
                source.name: {
                    "task_generation": {
                        queue.name: {
                            "queue_length": "queue_length",
                        },
                    },
                },
            },
        }
    )

    modeljson = model.json()
    with open(
        records_path + f"{params['run_number']}_oo_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    service_mean = np.mean(queue.service_rp._pregenerated_samples)
    logger.info(
        f"Service mean: {service_mean}, "
        + f"arrival mean: {1.00/arrival.rate}, utilization: {service_mean*arrival.rate}"
    )
    return_dict[params["run_number"]]["utilization"] = service_mean * arrival.rate

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        logger.info(
            f"{params['run_number']}: Simulation progress "
            + f"{100.0*float(model.env.now)/float(params['until'])}% done"
        )

    for step in np.arange(
        0, params["until"], params["until"] * params["report_state"], dtype=int
    ):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params["until"])
    end = time.time()
    logger.info(
        "{0}: OO: Run finished in {1} seconds".format(params["run_number"], end - start)
    )

    logger.info(
        "{0}: OO: Source generated {1} tasks".format(
            params["run_number"], source.get_attribute("tasks_generated")
        )
    )
    logger.info(
        "{0}: OO: Queue completed {1}, dropped {2}".format(
            params["run_number"],
            queue.get_attribute("tasks_completed"),
            queue.get_attribute("tasks_dropped"),
        )
    )
    logger.info(
        "{0}: OO: Sink received {1} tasks".format(
            params["run_number"], sink.get_attribute("tasks_received")
        )
    )

    # Process the collected data
    df = sink.received_tasks
    # print(df)

    df.write_parquet(
        file=records_path + f"{params['run_number']}_oo_records.parquet",
        compression="snappy",
    )

    dropped_df = df.filter(pl.col("end_time") == -1)
    passed_df = df.filter(pl.col("end_time") != -1)
    delayed_df = passed_df.filter(pl.col("end2end_delay") > pl.col("delay_bound"))

    failed_ratio = (dropped_df.height + delayed_df.height) / df.height
    logger.info(
        f"{params['run_number']}: OO: total={df.height}, "
        + f"passed={passed_df.height}, dropped={dropped_df.height} "
        + f"delayed={delayed_df.height}, failed ratio={failed_ratio} ",
    )

    return_dict[params["run_number"]]["total"] = df.height
    return_dict[params["run_number"]]["passed"] = passed_df.height
    return_dict[params["run_number"]]["dropped"] = dropped_df.height
    return_dict[params["run_number"]]["delayed"] = delayed_df.height
    return_dict[params["run_number"]]["failed_ratio"] = failed_ratio

    resultjson = json.dumps(return_dict[params["run_number"]].copy())
    with open(
        records_path + f"{params['run_number']}_oo_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(resultjson)
