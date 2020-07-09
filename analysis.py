# science/math libraries
import numpy as np
import matplotlib.pyplot as plt

# general libraries
import sys
import platform
import h5py as h5
import json


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        k: v[()] if type(v) is h5._hl.dataset.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }


def plot_peak_analysis(peaks, params=None):
    """Build a multi-panel plot of manipulations of synapse B peak voltage
    depolarizations following the event at synapse A.
    """
    if params is not None:
        inds = params["idx_range"]
        xaxis = np.arange(-inds, inds + 2) * params["seg_step"] * 1000
        xlabel = "Microns"
    else:
        xaxis = np.arange(peaks["EI"].shape[0])
        xlabel = "Segment"

    fig, ax = plt.subplots(1, 4)
    ax[0] = plot_peaks(ax[0], peaks, xaxis, xlabel)
    ax[1] = plot_peaks_sub_none(ax[1], peaks, xaxis, xlabel)
    ax[2] = plot_diffs(ax[2], peaks, xaxis, xlabel)
    ax[3] = plot_peak_suppression(ax[3], peaks, xaxis, xlabel)
    fig.tight_layout()

    return fig, ax


def plot_peaks(ax, peaks, xaxis, xlabel):
    """Peak depolarization at varying synapse B locations, in each of the
    synapse A conditions (simultaneous E+I, I alone, E alone, and no release).
    """
    for cond in ["EI", "I", "E", "None"]:
        ax.plot(xaxis, peaks[cond], label=cond)
    ax.legend()
    ax.set_title("Peak Voltage at Pure E Synapse")
    ax.set_xlabel(xlabel)
    return ax


def plot_spike_probs(probs, params=None):
    """Plot spike probability data (chance of ANY spikes occurring, usually 1)
    """
    if params is not None:
        inds = params["idx_range"]
        xaxis = np.arange(-inds, inds + 2) * params["seg_step"] * 1000
        xlabel = "Microns"
    else:
        xaxis = np.arange(probs["EI"].shape[0])
        xlabel = "Segment"

    fig, ax = plt.subplots(1)
    for cond in ["EI", "I", "E", "None"]:
        ax.plot(xaxis, probs[cond], label=cond)
    ax.legend()
    ax.set_title("Spike probability vs Pure E location")
    ax.set_xlabel(xlabel)

    return fig, ax


def plot_peaks_sub_none(ax, peaks, xaxis, xlabel):
    """Peak depolarization at varying synapse B locations, but with the peak
    depolarization in the quiet (no release) synapse A condition subtracted."""
    for cond in ["EI", "I", "E"]:
        ax.plot(xaxis, peaks[cond] - peaks["None"], label=cond)
    ax.legend()
    ax.set_title("None Condition Peak Voltage Subtracted")
    ax.set_xlabel(xlabel)
    return ax


def plot_diffs(ax, peaks, xaxis, xlabel):
    """Peak depolarization at varying synapse B locations for the EI condition,
    with an estimated 'influence' of each mono-release (E or I alone) condition
    subtracted. A window in to the impact of the E and I components of the
    corelease event.
    """
    sub_I = peaks["EI"] - (peaks["None"] - peaks["I"])
    sub_E = peaks["EI"] - (peaks["E"] - peaks["None"])

    ax.plot(xaxis, sub_I, label="EI sub I influence")
    ax.plot(xaxis, sub_E, label="EI sub E influence")
    ax.plot(xaxis, peaks["None"], label="None Condition (reference)")
    ax.legend()
    ax.set_title("EI Condition With Mono-Release Subtracted")
    ax.set_xlabel(xlabel)
    return ax


def plot_peak_suppression(ax, peaks, xaxis, xlabel, rest=-64.2):
    """Suppression ratio of the peak response amplitude (voltage difference
    from the resting potential) at syn_B by the event at syn_A for each of the
    conditions.
    """
    sub_bsln = {cond: pks - rest for cond, pks in peaks.items()}

    suppr = {
        cond: (pks - sub_bsln["None"]) / (pks + sub_bsln["None"])
        for cond, pks in sub_bsln.items()
    }

    for cond in ["EI", "I", "E", "None"]:
        ax.plot(xaxis, suppr[cond], label=cond)
    ax.legend()
    ax.set_title("Peak Suppression Ratio")
    ax.set_xlabel(xlabel)

    return ax


def plot_trace_examples(recs, params):
    """Voltage recording examples from syn_A -> syn_B interaction experiment.
    Including recordings at syn_B at different relative positions to syn_A,
    and the response to stimulation at syn_A on its own.
    """
    xaxis = np.arange(recs["EI"].shape[2]) / 10  # 10kHz sampling

    fig, ax = plt.subplots(1, 2)
    ax[0] = plot_synB_recs(
        ax[0], recs["EI"], [1, 2, 8], xaxis, params["idx_range"],
        params["seg_step"]
    )
    ax[1] = plot_synA_recs(ax[1], recs, xaxis, params["idx_range"])
    fig.tight_layout()

    return fig, ax


def plot_synB_recs(ax, cond_recs, positions, xaxis, idx_range, seg_step):
    """Voltage response at syn_B for a given syn_A condition, at varying
    relative positions (number of segments away).
    """
    for p in positions:
        dist = np.round(p * seg_step*1000, decimals=2)
        idx = p + idx_range
        ax.plot(xaxis, cond_recs[idx, idx, :], label="%dμ from site" % dist)

    ax.set_xlim(15, 50)
    ax.set_title("Mean Responses at synapse B (EI condition)")
    ax.set_ylabel("Membrane Voltage (at synapse B)")
    ax.set_xlabel("Time (ms)")
    ax.legend()

    return ax


def plot_synA_recs(ax, recs, xaxis, idx_A):
    """Voltage response at the site of syn_A, in the absense of any activity
    from syn_B.
    """
    for cond in ["EI", "I", "E"]:
        ax.plot(xaxis, recs[cond][idx_A, idx_A, :], label=cond)

    ax.set_xlim(15, 50)
    ax.set_title("Mean Responses at synapse A (no B)")
    ax.set_ylabel("Membrane Voltage (at synapse A)")
    ax.set_xlabel("Time (ms)")
    ax.legend()

    return ax


if __name__ == "__main__":
    if platform.system() == "Linux":
        basest = "/media/geoff/Data/NEURONoutput/corelease/"
    else:
        basest = "D:\\NEURONoutput\\corelease\\"

    if len(sys.argv) == 1:
        fname = "spiking_15Na_00045E_0024I_del5"
    else:
        fname = sys.argv[1]

    with h5.File(basest + fname + ".h5", "r") as pckg:
        data = unpack_hdf(pckg)

    # decode json of model params to dict
    if "params" in data.get("model", {}):
        data["params"] = json.loads(data["model"]["params"])
        if "experiment" in data["model"]:
            data["params"].update(json.loads(data["model"]["experiment"]))
        data["params"].update(json.loads(data["model"]["experiment"]))
    else:
        data["params"] = None

    fig1, ax1 = plot_peak_analysis(data["peaks"], data["params"])
    fig2, ax2 = plot_trace_examples(data["mean_recs"], data["params"])
    if "spike_probs" in data:
        fig3, ax3 = plot_spike_probs(data["spike_probs"], data["params"])

    # TODO: try stronger E, and see if spike probability is a useful way
    # to look at this. Also, get dend segs smaller for shorter A to B distance
