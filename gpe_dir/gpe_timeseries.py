import numpy as np


SERIES_METRICS = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "last": lambda y: y[-1]
}

# return all unique variable names (no duplicates) in the "name" column of the results
def list_variables(results):
    return sorted(results["name"].unique())

# Does this work for all model schema in simvascular/zerodsolver? Need to check
def get_series(results, variable):
    cols = results.column
    vector = results[results["name"] == variable]
    return vector["y"].to_numpy(), vector["time"].to_numpy()

def get_metric(y, metric):
    if metric not in SERIES_METRICS:
        raise ValueError(f"{metric} not in dictionary. Valid metric options are: {list(SERIES_METRICS)}")

    return float(SERIES_METRICS[metric](y))