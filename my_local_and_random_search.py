from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
# from smac.acquisition.maximizer.local_search import LocalSearch
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

from weighted_mo_hpi.experiments.my_local_search import MyLocalSearch

logger = get_logger(__name__)


class MyLocalAndSortedRandomSearch(AbstractAcquisitionMaximizer):
    """Implement SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points according, to the acquisition
    function, uses the acquisition function to sort randomly sampled configurations.
    Random configurations are interleaved by the main SMAC code.

    The Random configurations are interleaved to circumvent issues from a constant prediction
    from the Random Forest model at the beginning of the optimization process.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
    challengers : int, defaults to 5000
        Number of challengers.
    max_steps: int | None, defaults to None
        [LocalSearch] Maximum number of steps that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        [LocalSearch] number of steps during a plateau walk before local search terminates.
    local_search_iterations: int, defauts to 10
        [Local Search] number of local search iterations.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
        seed: int = 0,
        path_to_run=None
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        self._random_search = RandomSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            seed=seed,
        )

        self._local_search = MyLocalSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed,
            path_to_run=path_to_run
        )

        self._local_search_iterations = local_search_iterations

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:  # noqa: D102
        """Returns the used acquisition function."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function
        self._random_search._acquisition_function = acquisition_function
        self._local_search._acquisition_function = acquisition_function

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "random_search": self._random_search.meta,
                "local_search": self._local_search.meta,
            }
        )

        return meta

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self._random_search._maximize(
            previous_configs=previous_configs,
            n_points=n_points,
            _sorted=True,
        )

        next_configs_by_local_search = self._local_search._maximize(
            previous_configs=previous_configs,
            n_points=self._local_search_iterations,
            additional_start_points=next_configs_by_random_search_sorted,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = next_configs_by_random_search_sorted + next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        first_five = [f"{_[0]} ({_[1].origin})" for _ in next_configs_by_acq_value[:5]]

        logger.debug(f"First 5 acquisition function values of selected configurations:\n{', '.join(first_five)}")

        return next_configs_by_acq_value