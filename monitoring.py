import numpy as np
import numpy.typing as npt
from typing import Iterable
from scipy.special import logsumexp


class Monitor:

    def __init__(self):
        """This class should do two things:
        1. Compute metrics for integrator snippets.
        2. Check if we reached termination based on those metrics.
        """
        self.proportion_moved = 1.0
        self.proportion_resampled = 1.0
        self.particle_diversity = 1.0
        self.median_index_proportion = 1.0
        self.median_path_diversity = 1.0

    def update_metrics(self, attributes: dict):
        pass

    def terminate(self):
        raise NotImplementedError


class MonitorSingleIntSnippet(Monitor):

    def __init__(self, terminal_metric: float = 1e-2, metric: str = 'pm'):
        """This is the usual monitor for a single integrator snippet."""
        super().__init__()
        assert metric in {'pm', 'mip', 'mpd', None}, "Metric must be one of 'pm', 'mip', 'mpd' or None."
        self.terminal_metric = terminal_metric  # could be pm, mid, mpd
        self.metric = metric
        # choose the metric to test during termination based on self.metric
        match self.metric:
            case "pm":
                self.grab_metric = lambda: self.proportion_moved
            case "mip":
                self.grab_metric = lambda: self.median_index_proportion
            case "mpd":
                self.grab_metric = lambda: self.median_path_diversity
            case None:
                self.grab_metric = lambda: np.inf
        self.esjd = None
        self.ess_mubar = None

    def update_metrics(self, attributes: dict):
        """Updates the following:

        1. Expected Squared Jump Distance
        2. Proportion of Particles Moved
        3. Proportion of Particles resampled
        4. Particle Diversity
        5. Median Index Proportion
        6. Median Path Diversity
        """
        N = attributes['N']
        T = attributes['T']
        self.proportion_moved = np.sum(attributes['trajectory_indices'] >= 1) / N
        self.proportion_resampled = len(np.unique(attributes['indices'])) / N
        self.particle_diversity = (len(np.unique(attributes['particle_indices'])) - 1) / (N - 1)
        self.median_index_proportion = np.median(attributes['trajectory_indices']) / T
        self.median_path_diversity = np.sqrt(self.particle_diversity * self.median_index_proportion)
        # Compute ESS for mu bar. This requires obtaining the folded weights from the unfolded ones
        logw_folded = logsumexp(attributes['logw'], axis=1) - np.log(T+1)
        self.ess_mubar = np.exp(2*logsumexp(logw_folded) - logsumexp(2*logw_folded))

    def terminate(self):
        return self.grab_metric() <= self.terminal_metric


class MonitorMixtureIntSnippet:

    def __init__(self, *monitors: Monitor):
        self.monitors = monitors

    def update_metrics(self, attributes: dict):
        """For each monitor we need to update metrics, but we need the monitor itself to know its position relative
        to the other monitors."""
        for ix in range(len(self.monitors)):
            # for each monitor, we require to pass a dictionary with the following keys
            # N, T, trajectory_indices, indices, particle_indices
            # of which N, T are common for all but the other are specific to each integrator/monitor
            keys = {'trajectory_indices', 'indices', 'particle_indices'}
            filtered_attributes = {k: v for k, v in attributes.items() if k not in keys}
            for key in keys:
                filtered_attributes[key] = attributes[key][attributes['iotas'] == ix]
            self.monitors[ix].update_metrics(filtered_attributes)

    def terminate(self):
        """Terminates if any of its sub-monitors terminates."""
        return np.any([monitor.terminate() for monitor in self.monitors])
