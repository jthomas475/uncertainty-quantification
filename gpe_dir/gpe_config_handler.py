'''
File that configures all relevant information for the gpe workflow
'''
import json
from typing import Optional
from dataclasses import dataclass, field

from .gpe_param_handler import discover_params, ParamIdentifier, BLOCK_REGISTRY, scan_blocks
from .gpe_observables import get_observables, get_volumes

# should establish yaml handling? svzerotuner implementation uses it

@dataclass
class ConfigHandler:
    input_file: str
    bounds: tuple=(0.6,1.4)
    params: list = field(default_factory=list)
    param_bounds: dict = field(default_factory=dict)
    num_samples: int=32
    n_jobs: int=1
    max_gpe_wave_reps: int=20
    max_waves: int=5
    samples_per_wave: int=512
    gpe_cutoff: float=3.0

    observables: list = field(default_factory=list)
    observable_fields: tuple = ("pressure", "flow")
    observable_metrics: tuple = ("max","mean")
    calc_second_order: bool = True
    failure_threshold: float=0.05
    quiet_solver: bool = True
    output_dir: str = "../GPE_Results"
    raw_dir: str = "../GPE_RawOutputs"
    wave_dir: str = "../GPE_WaveResults"


    @classmethod
    def load(cls, input):
        if isinstance(input,ConfigHandler):
            return input
        
        if isinstance(input, dict):
            cls(**input)

        with open(input, "r") as file:
            return cls(**json.load(file))

def get_container(data, block, param):
    for container in BLOCK_REGISTRY:
        for name, values, _ in scan_blocks(data, container):

            if name == block and param in values:
                return container

def baseline_val(data, container, block, param):
    for name, values, _ in scan_blocks:
        if name == block and param in values:
            val = values[param]
            return val
    raise KeyError(f"{param} not found in {container} of {block}")

def get_params(data, config):
    if config.params:
        param_specs = []

        for param in config.params:
            container = param.get("container")

            if container is None:
                container = get_container(data, param["block"], param["param"])
            baseline = baseline_val(data, container, param["block"], param["param"])

            param_specs.append(ParamIdentifier(container=container, block=param["block"], param=param["param"], baseline=baseline, bounds=tuple.param.get("bounds", config.bounds), mode=param.get("mode","scale")))

    else:
        param_specs = discover_params(data, config.bounds)

    for param_spec in param_specs:
        for key, bound in config.param_bounds.items():
            if key in (param_spec.id, param_spec.param, f"{param_spec.block}:{param_spec.param}"):
                param_spec.bounds = tuple(bound)

    return param_specs


def generate_observables(config, baseline_results):
    return get_observables(baseline_results, fields=tuple(config.observable_fields),metrics=tuple(config.observable_metrics))


