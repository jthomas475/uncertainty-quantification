'''
Gaussian process emulator driver

'''

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import pysvzerod as zerod

from SALib.sample import sobol as sobol_sample

from GSA_library.plotting import * # imports all public names, such as functions, variables, classes, from plotting
from GSA_library.gsa_plotting import * 

from .gpe_config_handler import ConfigHandler, get_params, generate_observables
from sklearn.model_selection import train_test_split

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.metrics import IndependentStandardError as ISE

import torchmetrics

from pathlib import Path

from .misc_helpers import safe_save
from .run_zerod import replace_median, evaluate_model_parallel

'''
Overall GPE architecture:
1) Generate initial training data
2) train gpe
3) sample potential simulation input vectors
4) compute implausibility via history matching
5) keep non-implausible samples
6) evaluate the zerod model
7) append new taining data
8) repeat
'''

def get_training_data(config):
    config = ConfigHandler.load(config)

    with open(config.input_file, "r") as file:
        data = json.load(file)

    
    param_specs = get_params(data, config)

    baseline_results = zerod.simulate(data)
    observables = generate_observables(config, baseline_results)

    base_obs_vals = np.array([observable.evaluate(baseline_results) for observable in observables])
    base_obs_var = (0.05 * base_obs_vals)**2

    num_params = len(param_specs)
    problem = {
        "num_vars": len(param_specs),
        "names": [param_spec.id for param_spec in param_specs],
        "bounds": [list(param_spec.bounds) for param_spec in param_specs],
    }

    samples = sobol_sample.sample(problem, config.num_samples, calc_second_order=config.calc_second_order)

    time_initial = time.time()

    raw_results, outputs, failures, successes = evaluate_model_parallel(data, samples, param_specs, observables, config.n_jobs)

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
        "obs": observables,
        "observable_names": observable_names,
        "outputs": outputs_clean,
        "failures": failures,
        "data": data,
        "base_obs_vals": base_obs_vals,
        "base_obs_vars": base_obs_var,
        }
    
    return training_data


def predict_samples(emulators, problem, n_samples, calc_second_order, observable_names):
    samples = sobol_sample.sample(problem, n_samples, calc_second_order=calc_second_order)

    pred_means = np.zeros((len(samples), len(observable_names)))
    pred_stds = np.zeros((len(samples), len(observable_names)))

    for j, obs_name in enumerate(observable_names):
        mean, sigma = emulators[obs_name].predict(samples)

        pred_means[:,j] = mean
        pred_stds[:,j] = sigma

    return {
        "samples": samples,
        "pred_means": pred_means,
        "pred_std": pred_stds
    }

# Function adapted from Strocchi et al (made available in their open source github repository Strocchi_etal_2023_GSA. Specifically their emulation_step_by_step.py file in the GSA_+library)
def gaussian_emulator(training_data):
    x = training_data["inputs"]
    y = training_data["outputs"]
    observable_names = training_data["observable_names"]

    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.2, random_state=42)


    num_samples = len(x)
    timestamp = time.strftime("%d%m%Y_%H %M")

    data_path = Path("./gpe_train_data")
    data_path.mkdir(exist_ok=True)

    x_path = str(data_path / f"x_data_{timestamp}")
    y_path = str(data_path / f"y_data_{timestamp}")
    
    safe_save(x_path, x_train)
    safe_save(y_path, y_train)

    savepath = Path(f"./gpe_models_{timestamp}")
    savepath.mkdir(exist_ok=True)

    emulators = {}

    height = 9.36111
    width = 5.91667
    n_obs = len(observable_names)

    fig, axes = plt.subplots(1, n_obs, figsize=(n_obs * width, height / 2))
    if n_obs == 1:
        axes = [axes]  

    ci = 2  # 95% confidence interval
    inf_bound, sup_bound = [], []

    for j, obs_name in enumerate(observable_names):
        emulator = GPEmul(x_train, y_train[:,j])
        emulator.train(x_val, y_val[:,j], savepath= savepath/obs_name, save_losses=True, watch_metric="MSE") # could test out other regression metrics and kernels
        emulator.save(filename=str(savepath / f"gpe_{obs_name}_{timestamp}.pth"))
        emulators[obs_name] = emulator

        y_pred_mean, y_pred_std = emulator.predict(x_test)
        

        mse = torchmetrics.MeanSquaredError(emulator.tensorize(y_pred_mean), emulator.tensorize(y_test[:,j]))

        ise = ISE(
            emulator.tensorize(y_test[:,j]),
            emulator.tensorize(y_pred_mean),
            emulator.tensorize(y_pred_std),
        )

        print(f"\nStatistics on test set for GPE trained with validation for observable {obs_name}:")
        print(f"\tMSE = {mse:.4f}")
        print(f"\tISE = {ise:.2f} %\n")
    
        l = np.argsort(y_pred_mean)
        inf_bound.append((y_pred_mean - ci * y_pred_std).min())
        sup_bound.append((y_pred_mean + ci * y_pred_std).max())

        axes[j].scatter(np.arange(1, len(l) + 1), y_test[l, j], facecolors="none", edgecolors="C0", label="observed")
        axes[j].scatter(np.arange(1, len(l) + 1), y_pred_mean[l], facecolors="C0", s=16, label="predicted")
        axes[j].errorbar(np.arange(1, len(l) + 1), y_pred_mean[l], yerr=ci * y_pred_std[l], c="C0", ls="none", lw=0.5, label=f"uncertainty ({ci} STD)")
        axes[j].set_xticks([])
        axes[j].set_xticklabels([])
        axes[j].set_ylabel("Simulation data", fontsize=12)
        axes[j].set_title(f"{obs_name} | MSE = {mse:.4f} | ISE = {ise:.2f} %", fontsize=12)
        axes[j].legend(loc="upper left")

    for idx, ax in enumerate(axes):
        ax.set_ylim([np.min(inf_bound[idx]), np.max(sup_bound[idx])])

    fig.tight_layout()
    plt.savefig(savepath / "inference_on_testset.pdf", bbox_inches="tight", dpi=1000)

    return savepath, x_path, y_path, emulators


def history_match(preds, obs, obs_vars, cutoff, model_discrepency=0.0):
    var_total = (preds["pred_std"]**2 + obs_vars + model_discrepency)

    imp = np.abs(preds["pred_means"] - obs) / np.sqrt(var_total)
    imp_max = np.max(imp, axis=1)

    nimp_idxs = np.where(imp_max < cutoff)[0]
    imp_idxs = np.where(imp_max >= cutoff)[0]

    return {
        "imp": imp,
        "imp_max": imp_max,
        "nimp_idx": nimp_idxs,
        "imp_idx": imp_idxs,
    }

# cutoff is the implausability cutoff to determine what counts as non-implausible vs implausiblee
# batch_size is num of candidate points want GPE to predict at once
# max_reps explained above
def get_preds_NIMP(gaussian_emulator, problem, obs, obs_var, observable_names, N, calc_second_order, cutoff, batch_size=2048, max_reps=20):
    accepted = []

    reps = 0

    while sum(len(x) for x in accepted) < N:
        reps +=1
        if reps > max_reps:
            raise RuntimeError(f"Unable to find {N} non-implausible samples after {max_reps} repetitions. Only acquired {sum(len(x) for x in accepted)} non-implausible samples")

        #  Potential alternative to raising error if too restrictive (normal to have shrinking NIMP region)
        # if reps > max_reps:
        #     if accepted:
        #         return np.vstack(accepted)
        #     return np.empty((0, problem["num_vars"]))

        preds = predict_samples(gaussian_emulator, problem, batch_size, calc_second_order, observable_names) 
        history = history_match(preds, obs, obs_var, cutoff)

        if len(history["nimp_idx"]) > 0:
            accepted.append(preds["samples"][history["nimp_idx"]])

    accepted = np.vstack(accepted)

    return accepted[:N]

def append_training_data(training_data, new_inputs, new_outputs):
    training_data["inputs"] = np.vstack([training_data["inputs"],new_inputs])

    training_data["outputs"] = np.vstack([training_data["outputs"],new_outputs])

    return training_data

def history_match_wave(config):
    training_data = get_training_data(config)
    max_waves = config.max_wave
    samples_per_wave = config.samples_per_wave
    cutoff = config.cutoff


    for wave in range(max_waves):
        print(f"\nWave {wave}\n")

        _, _, _, emulators = gaussian_emulator(training_data)

        nimp_samples = get_preds_NIMP(emulators, training_data["problem"], training_data["base_obs_vals"], training_data["base_obs_vars"], training_data["observable_names"], samples_per_wave, config.calc_second_order, cutoff, max_reps=config.max_gpe_wave_reps)


        if len(nimp_samples)==0:
            print("No non-implausible samples")
            break

        _, new_outputs, _, _ = evaluate_model_parallel(data=training_data["data"], sample_params=nimp_samples, param_specs=training_data["specs"], observables=training_data["obs"], n_jobs=config.n_jobs)

        print(f"Accepted {len(nimp_samples)} of {samples_per_wave} samples ({100 * len(nimp_samples) / samples_per_wave}%)")

        new_outputs = replace_median(new_outputs)

        training_data = append_training_data(training_data, nimp_samples, new_outputs)

        timestamp = time.strftime("%d %m %Y")
        safe_save(os.path.join(config.raw_dir, f"wave{wave}_training_{timestamp}"), training_data)

    # retrain emulators on final dataset, so returned emulators reflects entire training data acquisition
    _, _, _, emulators = gaussian_emulator(training_data)
        
    return {
        "training_data": training_data,
        "emulators": emulators,
    }

