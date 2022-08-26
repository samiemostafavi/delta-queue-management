import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def plot_main(plot_args):

    logger.info(f"Drawing delay-bound benchmarks with args: {plot_args}")

    # set project folder
    project_folder = plot_args["project_folder"]

    # read utilization factor
    with open(project_folder + "info.json") as info_json_file:
        info = json.load(info_json_file)
    utilization = info["utilization"]

    # read project paths inside
    project_paths = [
        project_folder + name
        for name in os.listdir(project_folder)
        if os.path.isdir(os.path.join(project_folder, name))
    ]

    # limit
    # project_paths = ['projects/delta_benchmark/p8_results']
    logger.info(f"All project folders: {project_paths}")

    results = pd.DataFrame(columns=["delay target", "no-aqm", *plot_args["models"]])

    quantile_keys = []
    for project_path in project_paths:

        # get quantile key
        with open(project_path + "/info.json") as info_json_file:
            info = json.load(info_json_file)
        quantile_key = float(info["quantile_key"])
        quantile_keys.append(quantile_key)

        logger.info(
            f"Starting to import parquet files in: {project_path} with "
            + f"quantile key:{quantile_key}"
        )

        res_arr = []
        for key in plot_args["models"]:
            logger.info(f"AQM method: {key}")

            total_tasks = 0
            dropped_tasks = 0
            delayed_tasks = 0
            records_path = project_path + "/" + key + "/"
            all_files = os.listdir(records_path)
            for f in all_files:
                if f.endswith("result.json"):
                    with open(records_path + f) as info_json_file:
                        info = json.load(info_json_file)
                        total_tasks += info["total"]
                        dropped_tasks += info["dropped"]
                        delayed_tasks += info["delayed"]

            res_arr.append((dropped_tasks + delayed_tasks) / total_tasks)

        results.loc[len(results)] = [
            str(quantile_key),
            1.00 - quantile_key,
            *res_arr,
        ]

    results.sort_values(by=["delay target"], inplace=True)
    plt.style.use(["science", "ieee"])
    ax = results.plot(x="delay target", y=["no-aqm", *plot_args["models"]], kind="bar")
    ax.set_title(f"Utilization factor: {utilization:.3f}")
    ax.set_yscale("log")
    # ax.set_yticks(1.00 - np.array(quantile_keys))
    ax.set_xlabel("Target delay")
    ax.set_ylabel("Failed tasks ratio")
    # draw the legend
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(project_folder + "result.eps", format=plot_args["type"])


"""
bars = [str(par) for par in bench_params.values()]


y_pos = np.arange(len(bars))
fig, ax = plt.subplots()
ax.bar(
    y_pos,
    results,
    label="delta",
)
ax.bar(
    y_pos,
    1.00 - np.array(list(bench_params.values())),
    label="no-aqm",
)
# fix x axis
# ax.set_xticks(range(math.ceil(minx),math.floor(maxx),100))
plt.xticks(y_pos, bars)
plt.yticks(y_pos, list(bench_params.values()))
ax.set_yscale("log")
ax.set_xlabel("Target delay")
ax.set_ylabel("Failed tasks ratio")

# draw the legend
ax.legend()
ax.grid()

fig.savefig("result.png")
"""
