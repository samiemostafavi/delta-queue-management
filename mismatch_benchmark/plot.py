import getopt
import json
import os
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import scienceplots

def parse_plot_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hp:m:t:",
            ["project=", "models=", "type="],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m predictors_benchmark -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m delay_bound_benchmark plot "
                + "-a <arrival rate> -u <until> -l <label>",
            )
            sys.exit()
        elif opt in ("-p", "--project"):
            # project folder setting
            p = Path(__file__).parents[0]
            args_dict["project_folder"] = str(p) + "/" + arg + "_results/"
        elif opt in ("-m", "--models"):
            args_dict["models"] = [s.strip() for s in arg.split(",")]
        elif opt in ("-t", "--type"):
            args_dict["type"] = arg

    return args_dict


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

    res_dict = {}
    leg_dict = {}
    quantile_keys = []
    # iterate over quantile keys
    for project_path in project_paths:

        # get quantile key
        with open(project_path + "/info.json") as info_json_file:
            info = json.load(info_json_file)
        quantile_key = float(info["quantile_key"])
        qkey = info["quantile_key"]
        quantile_keys.append(quantile_key)

        logger.info(
            f"Starting to import parquet files in: {project_path} with "
            + f"quantile key:{quantile_key}"
        )

        res_dict[qkey] = {}
        leg_dict[qkey] = {}
        for mkey in plot_args["models"]:
            logger.info(f"AQM method: {mkey}")
            res_dict[qkey][mkey] = {}
            leg_dict[qkey][mkey] = {}

            # read utilization factor
            with open(project_path + "/" + mkey + "/info.json") as info_json_file:
                info = json.load(info_json_file)
            leg_dict[qkey][mkey]["legend_label"] = info["legend_label"]

            if "gmm" in mkey or "gmevm" in mkey or "delta" in mkey:
                ensembles = []
                files_folders = os.listdir(project_path + "/" + mkey)
                for file_folder in files_folders:
                    if not os.path.isfile(
                        os.path.join(project_path + "/" + mkey, file_folder)
                    ):
                        ensembles.append(file_folder)

                logger.info(f"found the following ensembles for {mkey}: {ensembles}")

                for ensemble in ensembles:
                    total_tasks = 0
                    dropped_tasks = 0
                    delayed_tasks = 0
                    records_path = project_path + "/" + mkey + "/" + ensemble + "/"
                    all_files = os.listdir(records_path)
                    for f in all_files:
                        if f.endswith("result.json"):
                            with open(records_path + f) as info_json_file:
                                info = json.load(info_json_file)
                                total_tasks += info["total"]
                                dropped_tasks += info["dropped"]
                                delayed_tasks += info["delayed"]

                    res_dict[qkey][mkey][ensemble] = (
                        dropped_tasks + delayed_tasks
                    ) / total_tasks

                model_results = list(res_dict[qkey][mkey].values())
                res_dict[qkey][mkey] = {
                    "min": min(model_results),
                    "max": max(model_results),
                    "mean": mean(model_results),
                }

            elif mkey == "oo":
                # oo
                total_tasks = 0
                dropped_tasks = 0
                delayed_tasks = 0
                records_path = project_path + "/" + mkey + "/"
                all_files = os.listdir(records_path)
                for f in all_files:
                    if f.endswith("result.json"):
                        with open(records_path + f) as info_json_file:
                            info = json.load(info_json_file)
                            total_tasks += info["total"]
                            dropped_tasks += info["dropped"]
                            delayed_tasks += info["delayed"]

                res_dict[qkey][mkey] = (dropped_tasks + delayed_tasks) / total_tasks

            elif mkey == "noaqm":
                # noaqm
                res_dict[qkey]["noaqm"] = 1.00 - quantile_key

    print(leg_dict)

    plt.style.use(["science", "ieee", "bright"])

    col_width = 2
    def_x = np.arange(len(list(res_dict.keys()))) + col_width

    fig, ax = plt.subplots()
    # iterate over the schemes
    q = 0   
    print(len(plot_args["models"]))
    for idx, scheme in enumerate(plot_args["models"]):

        # calculate x position
        num_schemes = len(plot_args["models"])
        bar_width = col_width / num_schemes / 3
        if num_schemes % 2 == 0:
            new_idx = idx - num_schemes / 2
            if new_idx < 0:
                x = def_x + (new_idx + 0.5) * bar_width
            else:
                new_idx = new_idx + 1
                x = def_x + (new_idx - 0.5) * bar_width
        else:
            new_idx = idx - (num_schemes - 1) / 2
            x = def_x + new_idx * bar_width

        targets = list(res_dict.keys())
        targets.sort()

        # fill y
        y = []

        color_choice =  ['greenyellow', 'lawngreen', 'limegreen', 'darkgreen', 'mediumseagreen', 'mediumspringgreen']
        if "gmm" in scheme or "gmevm" in scheme or "delta" in scheme:
            min_error = []
            max_error = []
            for target_delay in targets:
                min_error.append(
                    res_dict[target_delay][scheme]["mean"]
                    - res_dict[target_delay][scheme]["min"]
                )
                max_error.append(
                    res_dict[target_delay][scheme]["max"]
                    - res_dict[target_delay][scheme]["mean"]
                )
                y.append(res_dict[target_delay][scheme]["mean"])
            error = [min_error, max_error]
            ax.errorbar(x, y, yerr=error, fmt="", linestyle="")  # ecolor='grey'
            ax.bar(x, y, width=bar_width, label=leg_dict[target_delay][scheme]["legend_label"], color=color_choice[q], linestyle='dashed')
            #ax.plot(x, y, '--o', label=leg_dict[target_delay][scheme]["legend_label"])
            q = q +1
        elif "noaqm" in scheme:
            for target_delay in targets:
                y.append(res_dict[target_delay][scheme])
            ax.bar(x, y, width=bar_width, label=leg_dict[target_delay][scheme]["legend_label"], color='red')
            #ax.plot(x, y, '--o', label=leg_dict[target_delay][scheme]["legend_label"])
        else:
            for target_delay in targets:
                y.append(res_dict[target_delay][scheme])
            ax.bar(x, y, width=bar_width, label=leg_dict[target_delay][scheme]["legend_label"], color='deepskyblue')
        

    # ax = results.plot(x="delay target", y=["no-aqm", *plot_args["models"]], kind="bar")
    #ax.set_title(f"Utilization factor: {utilization:.3f}")
    #ax.set_xlim(0,3)
    ax.set_yscale("log")
    # ax.set_yticks(1.00 - np.array(quantile_keys))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1 : len(targets) + 1] = targets
    ax.set_xticklabels(labels)
    
    ax.set_xlabel("Target delay")
    ax.set_ylabel("Failed tasks ratio")
    # draw the legend
    ax.legend(fontsize=7, facecolor="white", frameon = True)
    #ax.legend(facecolor="black")
    #ax.legend(['No AQM', '','Mean =3.33', '', 'Mean =5', '', 'Mean =10*', '', 'Mean =20', '', 'Mean =80', '','OO', ''],fontsize=7)
    #ax.legend(fontsize=7, facecolor="black")
    ax.grid()
    plt.tight_layout()
    plt.savefig(
        project_folder + "result." + plot_args["type"], format=plot_args["type"]
    )
