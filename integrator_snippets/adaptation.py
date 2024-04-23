import numpy as np

class AdaptationStrategy:
    """Base class for adaptation strategies."""

    def adapt(self, attributes: dict) -> dict:
        """Adapts the parameters of the integrator(s).

        Parameters
        ----------
        :param attributes: Dictionary with the attributes of the integrator snippet
        :type attributes: dict
        :return: Dictionary with the adapted parameters
        :rtype: dict
        """
        raise NotImplementedError("Adapt method must be implemented in subclasses.")


class DummyAdaptation(AdaptationStrategy):
    """Dummy adaptation strategy, does nothing."""

    def adapt(self, attributes: dict) -> dict:
        """Does nothing."""
        return {'step_size': attributes['integrator'].step_size}


class SingleStepSizeAdaptorSA(AdaptationStrategy):
    """Adapts step size using stochastic approximations."""

    def __init__(self, target_metric_value: float, metric: str = 'pm', lr: float = 0.5,
                 min_step: float = 1e-30, max_step: float = 100.0):
        """Initialize SingleStepSizeAdaptorSA."""
        super().__init__()
        self.target_metric_value = target_metric_value
        self.metric = metric
        self.lr = lr
        self.min_step = min_step
        self.max_step = max_step
        self.metric_key_mapping = {
            'pm': "proportion_moved",
            'mip': "median_index_proportion",
            'mpd': "median_path_diversity"
        }

    def adapt(self, attributes: dict) -> dict:
        """Adapt step size."""
        metric_key = self.metric_key_mapping.get(self.metric)
        if not metric_key:
            raise ValueError("Invalid metric provided.")
        
        log_step = np.log(attributes['integrator'].step_size)
        metric_value = attributes['monitor'].__dict__[metric_key]
        
        new_step_size = np.clip(
            np.exp(log_step + self.lr * (metric_value - self.target_metric_value)),
            a_min=self.min_step,
            a_max=self.max_step
        )
        
        return {'step_size': new_step_size}


class MixtureStepSizeAdaptorSA(AdaptationStrategy):
    """Adapts the step size of a mixture of integrators using stochastic approximations."""

    def __init__(self, *adaptors: AdaptationStrategy):
        """Initialize MixtureStepSizeAdaptorSA."""
        super().__init__()
        self.adaptors = adaptors

    def adapt(self, attributes: dict) -> dict:
        """Adapt step sizes."""
        adaptation_dict = {}
        for ix, adaptor in enumerate(self.adaptors):
            filtered_attributes = {
                'integrator': attributes['integrators'].integrators[ix],
                'monitor': attributes['monitors'].monitors[ix]
            }
            adaptation_dict[ix] = adaptor.adapt(filtered_attributes)
        return adaptation_dict


class SingleStepSizeSATMinAdaptor(AdaptationStrategy):
    """Adapts step size using stochastic approximation and T based on the median index proportion."""

    def __init__(self, target_metric_value: float, metric: str = 'pm', lr: float = 0.5,
                 min_step: float = 1e-30, max_step: float = 100.0):
        """Initialize SingleStepSizeSATMinAdaptor."""
        super().__init__()
        self.target_metric_value = target_metric_value
        self.metric = metric
        self.lr = lr
        self.min_step = min_step
        self.max_step = max_step
        self.metric_key_mapping = {
            'pm': "proportion_moved",
            'mip': "median_index_proportion",
            'mpd': "median_path_diversity"
        }

    def adapt(self, attributes: dict) -> dict:
        """Adapt step size."""
        metric_key = self.metric_key_mapping.get(self.metric)
        if not metric_key:
            raise ValueError("Invalid metric provided.")

        log_step = np.log(attributes['integrator'].step_size)
        metric_value = attributes['monitor'].__dict__[metric_key]
        
        new_step_size = np.clip(
            np.exp(log_step + self.lr * (metric_value - self.target_metric_value)),
            a_min=self.min_step,
            a_max=self.max_step
        )

        return {'step_size': new_step_size, 'T': np.max(attributes['k_resampled'])}
