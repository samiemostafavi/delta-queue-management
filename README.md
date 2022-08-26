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

Run benchmarks

        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module delta --run-noaqm
        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module offline-optimum
        
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module delta --run-noaqm
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module offline-optimum

Plot the results

        $ python -m delay_bound_benchmark plot --project lowutil --models delta,offline-optimum
        $ python -m delay_bound_benchmark plot --project highutil --models delta,offline-optimum

# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

