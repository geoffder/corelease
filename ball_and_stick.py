from neuron import h, gui

# science/math libraries
import numpy as np
import matplotlib.pyplot as plt

# general libraries
import platform
import h5py as h5
import json


class Model(object):
    def __init__(self, params=None, attach_soma=True):
        self.attach_soma = attach_soma
        self.set_default_params()
        self.set_hoc_params()
        if params is not None:
            self.update_params(params)
        self.nz_seed = 1  # noise seed for HHst

        self.create_neuron()  # builds and connects soma and dendrite

    def set_default_params(self):
        # hoc environment parameters
        self.tstop = 100  # [ms]
        self.steps_per_ms = 10  # [10 = 10kHz]
        self.dt = .1  # [ms, .1 = 10kHz]
        self.v_init = -65
        self.celsius = 36.9

        # soma physical properties
        self.soma_L = 10
        self.soma_diam = 10
        self.soma_nseg = 1
        self.soma_Ra = 100

        # dendrite physical properties
        self.dend_nseg = 80
        self.seg_step = 1 / self.dend_nseg
        self.dend_diam = .5
        self.dend_L = 1200
        self.dend_Ra = 100

        # soma active properties
        self.activeSOMA = True
        self.somaNa = .15  # [S/cm2]
        self.somaK = .035  # [S/cm2]
        self.somaKm = .003  # [S/cm2]
        self.soma_gleak_hh = .0001667  # [S/cm2]
        self.soma_eleak_hh = -60.0  # [mV]
        self.soma_gleak_pas = .0001667  # [S/cm2]
        self.soma_eleak_pas = -60  # [mV]

        # dend compartment active properties
        self.activeDEND = True
        self.dendNa = .03  # [S/cm2] .03
        self.dendK = .025  # [S/cm2]
        self.dendKm = .003  # [S/cm2]
        self.dend_gleak_hh = 0.0001667  # [S/cm2]
        self.dend_eleak_hh = -60.0  # [mV]
        self.dend_gleak_pas = .0001667  # [S/cm2]
        self.dend_eleak_pas = -60  # [mV]

        # membrane noise
        self.dend_nzFactor = .1  # default NF_HHst = 1
        self.soma_nzFactor = .1

        # synaptic properties
        self.synprops = {
            "E": {
                "tau1": .1,  # excitatory conductance rise tau [ms]
                "tau2": 4,  # excitatory conductance decay tau [ms]
                "rev": 0,  # excitatory reversal potential [mV]
                "weight": .00023,  # weight of excitatory NetCons [uS] .00023
            },
            "I": {
                "tau1": .5,  # inhibitory conductance rise tau [ms]
                "tau2": 12,  # inhibitory conductance decay tau [ms]
                "rev": -65,  # inhibitory reversal potential [mV]
                "weight": .0024,  # weight of inhibitory NetCons [uS]
            }
        }

    def update_params(self, params):
        """Update self members with key-value pairs from supplied dict."""
        for k, v in params.items():
            self.__dict__[k] = v

    def get_params_dict(self):
        params = self.__dict__.copy()
        # remove the non-param entries (model objects)
        for key in ["soma", "dend", "syns"]:
            params.pop(key)
        return params

    def set_hoc_params(self):
        """Set hoc NEURON environment model run parameters."""
        h.tstop = self.tstop
        h.steps_per_ms = self.steps_per_ms
        h.dt = self.dt
        h.v_init = self.v_init
        h.celsius = self.celsius

    def create_soma(self):
        """Build and set membrane properties of soma compartment"""
        soma = nrn_section("soma")
        soma.L = self.soma_L
        soma.diam = self.soma_diam
        soma.nseg = self.soma_nseg
        soma.Ra = self.soma_Ra

        if self.activeSOMA:
            soma.insert('HHst')
            soma.gnabar_HHst = self.somaNa
            soma.gkbar_HHst = self.somaK
            soma.gkmbar_HHst = self.somaKm
            soma.gleak_HHst = self.soma_gleak_hh
            soma.eleak_HHst = self.soma_eleak_hh
            soma.NF_HHst = self.soma_nzFactor
        else:
            soma.insert('pas')
            soma.g_pas = self.soma_gleak_pas
            soma.e_pas = self.soma_eleak_hh

        return soma

    def create_dend(self):
        """Build and set membrane properties of dend compartment"""
        dend = nrn_section("dend")
        dend.nseg = self.dend_nseg
        dend.diam = self.dend_diam
        dend.L = self.dend_L
        dend.Ra = self.dend_Ra

        if self.activeDEND:
            dend.insert("HHst")
            dend.gnabar_HHst = self.dendNa
            dend.gkbar_HHst = self.dendK
            dend.gkmbar_HHst = self.dendKm
            dend.gleak_HHst = self.dend_gleak_hh
            dend.eleak_HHst = self.dend_eleak_hh
            dend.NF_HHst = self.dend_nzFactor
        else:
            dend.insert('pas')
            dend.g_pas = self.soma_gleak_pas
            dend.e_pas = self.soma_eleak_hh

        return dend

    def create_synapses(self):
        # access hoc compartment
        self.dend.push()

        # create *named* hoc objects for each synapse (for gui compatibility)
        h("objref e_syns[%i], i_syns[%i]" % (self.dend_nseg, self.dend_nseg))

        # complete synapses are made up of a NetStim, Syn, and NetCon
        self.syns = {
            "E": {"stim": [], "syn": h.e_syns, "con": []},
            "I": {"stim": [], "syn": h.i_syns, "con": []},
        }

        for i in range(self.dend_nseg):
            # 0 -> 1 position dendrite section
            pos = np.round((i + 1) * self.seg_step, decimals=5)

            for src in ["E", "I"]:
                # Synapse object (source of conductance)
                self.syns[src]["syn"][i] = h.Exp2Syn(pos)
                self.syns[src]["syn"][i].tau1 = self.synprops[src]["tau1"]
                self.syns[src]["syn"][i].tau2 = self.synprops[src]["tau2"]
                self.syns[src]["syn"][i].e = self.synprops[src]["rev"]

                # Network Stimulus object (activates synaptic event)
                self.syns[src]["stim"].append(h.NetStim(pos))
                self.syns[src]["stim"][i].interval = 0
                self.syns[src]["stim"][i].number = 0
                self.syns[src]["stim"][i].noise = 0

                # Network Connection object (connects stimulus to synapse)
                self.syns[src]["con"].append(
                    h.NetCon(
                        self.syns[src]["stim"][i],
                        self.syns[src]["syn"][i],
                        0,  # threshold
                        0,  # delay [ms]
                        self.synprops[src]["weight"],  # conductance strength
                    )
                )

        # remove section from access stack
        h.pop_section()

    def create_neuron(self):
        # create compartments (using parameters in self.__dict__)
        self.soma = self.create_soma()
        self.dend = self.create_dend()

        # generate synapses on dendrite
        self.create_synapses()

        # wire up compartments
        if self.attach_soma:
            self.dend.connect(self.soma)

    def run(self):
        h.init()  # reset model state, initialize voltage

        # update HHst membrane noise seeds
        if self.activeSOMA:
            self.soma.seed_HHst = self.nz_seed
            self.nz_seed += 1

        if self.activeDEND:
            self.dend.seed_HHst = self.nz_seed
            self.nz_seed += 1

        h.run()

    def sequence_trial(self, idx_a, syn_a=["E", "I"], syn_b=["E"], delay=10,
                       idx_range=20):
        # turn on the corelease synapse
        for src in syn_a:
            self.syns[src]["stim"][idx_a].number = 1
            self.syns[src]["stim"][idx_a].start = 20

        dend_recs = [
            h.Vector().record(self.dend(
                np.round((i + 1) * self.seg_step, decimals=5)
            )._ref_v)
            # for i in range(self.dend_nseg)
            for i in range(idx_a-idx_range, idx_a+idx_range+2)
        ]
        rec_mats = []

        for i in range(idx_a-idx_range, idx_a+idx_range+2):
            # turn on syn_b conductances at delay (skip modifying syn_a)
            if i != idx_a:
                for src in syn_b:
                    self.syns[src]["stim"][i].number = 1
                    self.syns[src]["stim"][i].start = 20 + delay

            self.run()

            # store recordings and clear the hoc vectors for the next run
            rec_mats.append(np.array(dend_recs))
            for rec in dend_recs:
                rec.resize(0)

            # turn the syn_b synapse back off (skip modifying syn_a)
            if i != idx_a:
                for src in syn_b:
                    self.syns[src]["stim"][i].number = 0

        # turn the corelease synapse back off
        for src in syn_a:
            self.syns[src]["stim"][idx_a].number = 0

        return rec_mats


def nrn_section(name):
    """Create NEURON hoc section, and return a corresponding python object."""
    h("create " + name)
    return h.__getattribute__(name)


def windows_gui_fix():
    """Workaround GUI bug in windows preventing graphs drawing during run"""
    h('''
        proc advance() {
            fadvance()
            nrnpython("")
        }
    ''')


def min_max_scaling(arr):
    """Normalize on a 0 -> 1 scale."""
    arr = arr - arr.min()
    return arr / (arr.max() + .00001)


def mean_norm(arr):
    return (arr - arr.mean()) / arr.var()


def interaction_experiment(model, site, delay=5, idx_range=39, trials=10):
    """
    Run multi-trial experiment of sequence trials, and return the mean of
    the recordings of each dendritic segment for each condition, along with
    the peak voltages at the second (moving) synapse for each position.
    """
    trial_recs = {
        cond if len(cond) else "None": np.stack([
            neuron.sequence_trial(
                site, syn_a=list(cond), delay=delay, idx_range=idx_range
            )
            for _ in range(trials)
        ], axis=0)
        for cond in ["EI", "I", "E", ""]
    }

    mean_recs = {
        cond: np.mean(recs, axis=0).squeeze()
        for cond, recs in trial_recs.items()
    }

    peaks = {
        cond: np.stack([
            recs[i, i, :]
            if i != idx_range else np.full_like(recs[i, i, :], np.nan)
            for i in range(len(recs))
        ]).max(axis=1)
        for cond, recs in mean_recs.items()
    }

    spike_probs = {
        cond: calc_spike_probs(recs)
        for cond, recs in trial_recs.items()
    }

    return {"mean_recs": mean_recs, "peaks": peaks, "spike_probs": spike_probs}


def calc_spike_probs(cond_recs, threshold=20):
    """Pare down recording matrix to just the rec for the stimulated site on
    each trial. Determine is a spike happened during each of these recordings,
    then return the proportion of trials in which a spike occured.

    shape progression:
    (trials, syn B, rec sites, time)
    -> (trials, syn B, time)
    -> (trials, syn B)
    -> (syn B,)
    """
    cond_recs = np.array([
        [trial[i, i, :] for i in range(len(trial))]
        for trial in cond_recs
    ])

    did_spike = (cond_recs > threshold).any(axis=2)

    return did_spike.sum(axis=0) / did_spike.shape[0]


def pack_hdf(pth, data_dict):
    """
    Takes data organized in a python dict, and creates an hdf5 with the same
    structure.
    """
    pckg = h5.File(pth + '.h5', 'w')

    for key, dataset in data_dict.items():
        set_grp = pckg.create_group(key)
        for cond, data in dataset.items():
            set_grp.create_dataset(cond, data=data)

    pckg.close()


if __name__ == "__main__":
    if platform.system() == "Linux":
        basest = "/media/geoff/Data/NEURONoutput/corelease/"
    else:
        basest = "D:\\NEURONoutput\\corelease\\"
        windows_gui_fix()

    # spiking
    # change_params = {
    #     "dendNa": .15,
    #     "synprops": {
    #         "E": {"tau1": .1, "tau2": 4, "rev": 0, "weight": .00045},
    #         "I": {"tau1": .5, "tau2": 12, "rev": -65, "weight": .0024}
    #     }
    # }

    # strong inputs
    # change_params = {
    #     "dendNa": .15,
    #     "synprops": {
    #         "E": {"tau1": .1, "tau2": 4, "rev": 0, "weight": .00046},
    #         "I": {"tau1": .5, "tau2": 12, "rev": -65, "weight": .0036}
    #     }
    # }

    # weak inputs
    change_params = {
        "synprops": {
            "E": {"tau1": .1, "tau2": 4, "rev": 0, "weight": .00023},
            "I": {"tau1": .5, "tau2": 12, "rev": -65, "weight": .0012}
        }
    }

    neuron = Model(change_params, attach_soma=False)

    h.xopen("ball_stick.ses")  # open neuron gui session

    # experimental conditions
    conds = {
        "ref_pos": neuron.dend_nseg // 2 - 1,  # middle
        "delay": 5,
        "idx_range": 18,
        "trials": 10,
    }

    results = interaction_experiment(neuron, *list(conds.values()))

    results["model"] = {}
    results["model"]["experiment"] = json.dumps(conds)
    results["model"]["params"] = json.dumps(neuron.get_params_dict())

    pack_hdf(basest + input("Experiment Package name: "), results)

    plt.plot(results["peaks"]["EI"], label="ei")
    plt.plot(results["peaks"]["I"], label="i")
    plt.plot(results["peaks"]["None"], label="none")
    plt.plot(results["peaks"]["E"], label="e")
    plt.legend()
