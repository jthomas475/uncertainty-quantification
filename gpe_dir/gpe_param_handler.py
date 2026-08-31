'''
File for handling parameters of a model.
Discovers tunable parameters and format them as objects of the ParamIdentifier class.
'''

import json
from typing import Optional
import copy
from dataclasses import dataclass, field


# Create dictionary of svzerodsolver blocks and their corresponding name and field
BLOCK_REGISTRY = {
    "vessels": ("vessel_name", "zero_d_element_values"),
    "valves": ("name", "params"),
    "junctions": ("junction_name", "junction_values"),
    "chambers": ("name", "values"),
    "boundary_conditions": ("bc_name", "bc_values"),
    "closed_loop_blocks": ("closed_loop_type", "parameters"),
}

def is_scalar(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool)

'''
ParamIdentifier discovers tunable parameters and its associated bounds

'''
@dataclass
class ParamIdentifier:
    container: str # which overall cardiac type, such as "vessels" or "valves"
    block: str # block identifier, such as the vessel_name or bc_name
    param: str # the parameter key (parameter being investigated)
    baseline: float # baseline value of the tunable parameter
    bounds: tuple = (0.6, 1.4) # bounds of the scaler value that will be used for the parameter
    mode: str = "scale" # perturbation type; 'scale' mode generates value that is product of sampled scaler (w/in bounds) and baseline parameter value
    

    @property
    def id(self):
        return f"{self.container}-{self.block}:{self.param}"

    def value(self,x):
        return self.baseline*x if self.mode == "scale" else x

def is_tuneable_list(mylist):
    return isinstance(mylist, list) and all(is_scalar(i) for i in mylist)


def scan_blocks(data, container):
    if not data:
        raise ValueError(f"No data provided")

    elif container not in BLOCK_REGISTRY:
        raise KeyError(f"{container} not listed in block registry")

    name, value = BLOCK_REGISTRY[container]

    for block in data.get(container, []) or []:
        values = block.get(value, dict)
        
        if isinstance(values, dict):
            yield block.get(name), values, block


def discover_params(data, bounds=(0.6,1.4), containers=None):
    containers = containers or list(BLOCK_REGISTRY)
    param_specs = []

    for container in containers:
        for name, values, _ in scan_blocks(data, container):
            for key, val in values.items():

                if key.startswith("_"):
                    continue

                if is_scalar(val):
                    param_specs.append(ParamIdentifier(container, name, key, float(val),tuple(bounds)))

    return param_specs


def apply_scaler(data, param_specs, row):
    perturbed = copy.deepcopy(data)

    for param_spec, scaler in zip(param_specs, row):
        for name, values, _ in scan_blocks(perturbed, param_spec.container):
            if name == param_spec.block and param_spec.param in values:
                values[param_spec.param] = param_spec.value(scaler)

    return perturbed

