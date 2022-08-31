[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# delta-queue-management
To reproduce the simulations and results of the paper "Delta: Predictive Queue Management for Time Sensitive Stochastic Networking"

# Start

Create a Python 3.9 virtual environment using `virtualenv`

        $ python -m virtualenv --python=python3.9 ./venv
        $ source venv/bin/activate

Install dependencies

        $ pip install -Ur requirements.txt

# Usage

Tune CoDel, note that `until` argument above 100k makes the run much longer

        $ python -m tune_codel -a 0.09 -u 100000 -l tune_lowutil --target-bounds 1,300 --target-initial 20 --interval-bounds 1,300 --interval-initial 10 --run-noaqm
        $ python -m tune_codel -a 0.095 -u 100000 -l tune_highutil --target-bounds 1,300 --target-initial 20 --interval-bounds 1,300 --interval-initial 10 --run-noaqm

Train Deep Queue Network implementation as follows. We set the `delay_ref` parameter to the target delay obtained by tuning CoDel.

        $ python -m train_deepq -a 0.09 -u 100000 -e 1000000 -l lowutil --interval 2 --delta 0.8 --run-noaqm
        $ python -m train_deepq -a 0.095 -u 100000 -e 1000000 -l highutil --interval 2 --delta 0.9 --run-noaqm

Run delay bound benchmarks

        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module delta --run-noaqm
        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module offline-optimum
        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module codel
        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module deepq
        
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module delta --run-noaqm
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module offline-optimum
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module codel
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module deepq

Plot the delay bound benchmark results

        $ python -m delay_bound_benchmark plot --project lowutil --models deepq,codel,delta,offline-optimum --type png
        $ python -m delay_bound_benchmark plot --project highutil --models deepq,codel,delta,offline-optimum --type png

Run delta models benchmarks

        $ python -m delta_models_benchmark run -a 0.089 -u 1000000 -l lowutil -m gmevm --run-noaqm
        $ python -m delta_models_benchmark run -a 0.089 -u 1000000 -l lowutil -m gmm --run-noaqm

        $ python -m delta_models_benchmark run -a 0.094 -u 1000000 -l highutil -m gmevm --run-noaqm
        $ python -m delta_models_benchmark run -a 0.094 -u 1000000 -l highutil -m gmm --run-noaqm

Plot delta models benchmark results

        $ python -m delta_models_benchmark plot --project lowutil --models gmm,gmevm,offline-optimum --type png
        $ python -m delta_models_benchmark plot --project highutil --models gmm,gmevm,offline-optimum --type png

# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

