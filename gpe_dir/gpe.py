'''
Gaussian process emulator driver

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pysvzerod as zerod
import json
import os
import copy
import time
import sys
import pandas as pd


from SALib.sample import sobol as sobol_sample
from joblib import Parallel, delayed
from contextlib import contextmanager

from gpytGPE.utils.design import read_labels

from GSA_library import kfold_cross_validation_training
from GSA_library import global_sobol_sensitivity_analysis
from GSA_library import gsa_parameters_ranking

from GSA_library.plotting import * # imports all public names, such as functions, variables, classes, from plotting
from GSA_library.gsa_plotting import * 

from .gpe_config_handler import ConfigHandler, get_params, generate_observables
from .gpe_timeseries import get_series, get_metric
from .gpe_param_handler import apply_scaler, discover_params
from sklearn.model_selection import train_test_split

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.metrics import IndependentStandardError as ISE
import torchmetrics
from pathlib import Path



def safe_save(path, data):
    if not path.endswith(".npy"):
        path += ".npy"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if os.path.exists(path):
        raise FileExistsError(
            f"file '{path}' already exists"
        )
    np.save(path, data)
    return path

@contextmanager
def quiet_func():
    devnull = os.open(os.devnull, flags=os.O_WRONLY)

    saved_out = os.dup(1) # save to standard output (stdout)
    saved_err = os.dup(2) # save to standard error (stderr) (note that dup(0) saves to standard input (stdin))

    os.dup2(devnull, 1)
    os.dup2(devnull,2)

    try:
        yield

    finally:
        os.dup2(saved_out,1)
        os.dup2(saved_err,2)

        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)

def replace_median(output):
    Y = output.copy()

    for j in range(Y.shape(1)):
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
    
    except (RuntimeError, ValueError) as e:
        return None, observables_outputs, str(e)




def evaluate_model_parallel(data, sample_params, param_specs, observables, n_jobs=1):
    num_samples, num_observables = len(sample_params), len(observables)

    observables_df = pd.DataFrame(index=range(num_samples), columns=range(num_observables))

    raw_results = [None] * num_samples

    failures = []

    successes = []

    param_ids = [param.id for param in param_specs]

    timeInitial = time.time()

    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(evaluate_single)(data, sample_params[idx], param_specs, observables) for idx in range(num_samples)
    )

    for idx, (raw_result, output, e) in enumerate(results):
        observables_df.iloc[idx] = output
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

    return raw_results, observables_df.tonumpy(), failures, successes


def get_training_data(config):
    config = ConfigHandler.load(config)

    with open(config.input_file, "r") as file:
        data = json.load(file)

    

    param_specs = get_params(data, config)

    baseline_results = zerod.simulate(data)
    observables = generate_observables(config, baseline_results)

    num_params = len(param_specs)
    problem = {
        "num_vars": len(param_specs),
        "names": [param_spec.id for param_spec in param_specs],
        "bounds": [list(param_spec.bounds) for param_spec in param_specs],
    }

    samples = sobol_sample.sample(problem, config.num_samples, calc_second_order=config.calc_second_order)

    time_initial = time.time()

    raw_results, outputs, failures, successes = evaluate_model_parallel(data, samples, param_specs, observables, config, config.n_jobs)

    time_elapsed = time.time() - time_initial
    mins, secs = np.divmod(time_elapsed, 60)

    print(f"Producing training data ")

    observable_names = [observable.name for observable in observables]

    safe_save(os.path.join(config.raw_dir, f"raw_gpe_training_data_{time.strftime("%d %m %Y_%H %M")}"),{"inputs":samples, "raw_results":raw_results,"outputs": outputs, "observable_names": observable_names, "param_names": problem["names"]})


    failure_rate = (len(failures) / len(samples))
    print(f"Failure rate: {failure_rate}")

    if failure_rate > 0.05:
        print("WARNING: Failure rate > 0.05. Sensitivity indices may be unreliable.")

    outputs_clean = replace_median(outputs)

    training_data = {
        "problem": problem,
        "specs": param_specs,
        "inputs": samples,
        "raw_results": raw_results,
        "observable_names": observable_names,
        "outputs": outputs_clean,
        "failures": failures,
        }
    
    return training_data


# Function established by Strocchi et al (made available in their open source github repository Strocchi_etal_2023_GSA. Specifically their emulation_step_by_step.py file in the GSA_+library)
def gaussian_emulator(training_data):
    x = training_data["inputs"]
    y = training_data["raw_results"]

    x_, x_test, y_, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.2, random_state=42)


    num_samples = len(x)
    power = np.log2(num_samples)
    timestamp = time.strftime("%d%m%Y_%H %M")

    data_path = Path("./gpe_train_data")
    data_path.mkdir(exist_ok=True)

    x_path = str(data_path / f"x_data_{timestamp}")
    y_path = str(data_path / f"y_data_{timestamp}")
    
    safe_save(x_path, x_train)
    safe_save(y_path, y_train)

    savepath = Path("./gpe_models")
    savepath.mkdir(exist_ok=True)


    emulator = GPEmul(x_train, y_train)
    emulator.train(x_val, y_val, savepath=savepath, save_losses=True, watch_metric="MSE") # could test out other regression metrics

    emulator.save(filename=str(savepath / f"gpe_{timestamp}.pth"))



    mean_list = []
    std_list = []
    mse_list = []
    ise_list = []

    y_pred_mean, y_pred_std = emulator.predict(x_test)
    mean_list.append(y_pred_mean)
    std_list.append(y_pred_std)

    mse = torchmetrics.MeanSquaredError(emulator.tensorize(y_pred_mean), emulator.tensorize(y_test))

    mse_list.append(mse)

    ise = ISE(
        emulator.tensorize(y_test),
        emulator.tensorize(y_pred_mean),
        emulator.tensorize(y_pred_std),
    )

    ise_list.append(ise)
    print(f"\nStatistics on test set for GPE trained with validation:")
    print(f"\tMSE = {mse:.4f}")
    print(f"\tISE = {ise:.2f} %\n")

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 4))

    ci = 2  # ~95% confidance interval

    inf_bound = []
    sup_bound = []

    for i, (m, s) in enumerate(zip(mean_list, std_list)):
        l = np.argsort(m)  # for the sake of a better visualisation
        inf_bound.append((m - ci * s).min())  # same
        sup_bound.append((m + ci * s).max())  # same

        axes[i].scatter(
            np.arange(1, len(l) + 1),
            y_test[l],
            facecolors="none",
            edgecolors="C0",
            label="observed",
        )
        axes[i].scatter(
            np.arange(1, len(l) + 1),
            m[l],
            facecolors="C0",
            s=16,
            label="predicted",
        )
        axes[i].errorbar(
            np.arange(1, len(l) + 1),
            m[l],
            yerr=ci * s[l],
            c="C0",
            ls="none",
            lw=0.5,
            label=f"uncertainty ({ci} STD)",
        )

        axes[i].set_xticks([])
        axes[i].set_xticklabels([])
        axes[i].set_ylabel("Simulation data", fontsize=12)
        axes[i].set_title(
            f"GPE with validation | MSE = {mse_list[i]:.4f} | ISE = {ise_list[i]:.2f} %",
            fontsize=12,
        )
        axes[i].legend(loc="upper left")

    axes[0].set_ylim([np.min(inf_bound), np.max(sup_bound)])
    axes[1].set_ylim([np.min(inf_bound), np.max(sup_bound)])

    fig.tight_layout()
    plt.savefig(
        savepath + "inference_on_testset.pdf", bbox_inches="tight", dpi=1000
    )

    return savepath, x_path, y_path

# Function that uses the trained gpe to generate simulation input vectors
def generate_predictions(config, training_data, gpe_savepath, x_path, y_path):
    if not os.path.exists(gpe_savepath):
        raise Exception(f"Path to trained GPE cannot be found: {gpe_savepath}")

    x_train = np.load(x_path)
    y_train = np.load(y_path)

    gaussian_emulator = GPEmul.load(x_train, y_train, gpe_savepath)


    problem = training_data["problem"]
    
    samples = sobol_sample.sample(problem, 1024, calc_second_order=config.calc_second_order)



    means, sigmas = gaussian_emulator.predict(samples) # means and standard deviations of gpe-predicted probability distribution over possible values

    gpe_predictions = {
        "inputs": samples,
        "predicted_mean": means,
        "predicted_std": sigmas,
    }

    return gpe_predictions


    # optimize gpe using training loss and use validation loss to evaluate generalization and provide insight on how the model performs on new data and when to stop training
    # for i in range(512):
    # # train model
    #     train(gpe, x_train, y_train)

    # # evaluate on validation data
    #     val_accuracy = evaluate(model, x_val, y_val)

    #     if val_accuracy > best_accuracy:
    #         save_gpe(gpe)

