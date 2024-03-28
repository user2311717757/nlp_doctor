# Overview

This is the repository accompanying ICLR 2024 paper ["Beyond Memorization: Violating Privacy via Inference with Large Language Models"](https://arxiv.org/abs/2310.07298).

## Getting started

The easiest way to get started with the project is to simply run the `demo.py` script. This will load a synthetic example and run a single query on it. This will give you a brief overview of the the main prompt and outputs.

Before running the demo, you need to install the environment. We recommend using mamba to install the environment via `mamba env create -f environment.yaml` which will create an environment called `beyond-mem` (see for detailed instructions below). You can then activate the environment via `conda activate beyond-mem`.

The demo script uses the OpenAI API and you need to set the credentials in `credentials.py` to use the OpenAI-API (we provide a template in `credentials_template.py`). You can adapt the code directly in the `demo.py` file to use a different model (Line 21) and refer you to exemplarary configs such as `configs/reddit/running/reddit_llama2_7b.yaml` for reference.

If you want to run other experiments, you can use the `main.py` script. This script takes a config file as input and runs the respective experiment. We provide a set of sample configs in the `configs` folder. You can run the script via `python ./main.py --config_path <your_config>`. For more detail we refer to the documentation below.
