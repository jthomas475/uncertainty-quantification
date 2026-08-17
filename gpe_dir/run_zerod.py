import numpy as np
import pysvzerod as zerod
import time

from joblib import Parallel, delayed
from SALib.sample import sobol as sobol_sample

from .gpe_param_handler import apply_scaler
from .misc_helpers import safe_save, quiet_func


def replace_median(output):
    Y = output.copy()

    for j in range(Y.shape[1]):
        col = Y[:,j]
        mask = np.isnan(col)

        if mask.any() and not mask.all():
            col[mask] = np.nanmedian(col)

    return Y


def evaluate_single(data, param_row, param_specs, observables):
    observables_outputs = np.full(len(observables), np.nan)
    perturbed_data = apply_scaler(data, param_specs, param_row)

    try:
        with quiet_func():
            results = zerod.simulate(perturbed_data)

        for idx, observable in enumerate(observables):
            observables_outputs[idx] = observable.evaluate(results)

        return results, observables_outputs, None
    
    except (RuntimeError, ValueError, KeyError) as e:
        return None, observables_outputs, str(e)


def evaluate_model_parallel(data, sample_params, param_specs, observables, n_jobs=1):
    num_samples, num_observables = len(sample_params), len(observables)

    outputs = np.full((num_samples, num_observables), np.nan)

    raw_results = [None] * num_samples

    failures = []

    successes = []

    param_ids = [param.id for param in param_specs]

    timeInitial = time.time()

    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(evaluate_single)(data, sample_params[idx], param_specs, observables) for idx in range(num_samples)
    )

    for idx, (raw_result, output, e) in enumerate(results):
        outputs[idx] = output

        raw_results[idx] = raw_result

        if e is not None:
            failures.append((idx, dict(zip(param_ids, sample_params[idx].tolist())), e))

            print(f"fail {len(failures)} sample {idx} \n Error Message: {e} \n ")

        else:
            successes.append((idx, dict(zip(param_ids, sample_params[idx].tolist()))))

        curr = idx+1
        
        if curr == num_samples or curr % max(1, num_samples // 20) == 0:
            time_elapsed = time.time() - timeInitial

            rate = curr / time_elapsed if time_elapsed else 0
            eta = (num_samples-curr) / rate if rate else 0
            print(f"[{idx}/{num_samples}] {len(failures)} failed, {time_elapsed:.0f}s elapsed, ~{eta:.0f}s left \n\n\n", flush=True)

    return raw_results, outputs, failures, successes

