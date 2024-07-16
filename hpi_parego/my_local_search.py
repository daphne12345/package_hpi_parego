from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from deepcave.runs.converters.smac3v2 import SMAC3v2Run

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer import LocalSearch
from smac.utils.configspace import (
    convert_configurations_to_array,
    get_one_exchange_neighbourhood,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

from hpi_parego.fanova import fANOVAWeighted

logger = get_logger(__name__)


def update(cfg, hp, value):
    cfg_change =  copy.copy(cfg)
    cfg_change[hp] = value
    return cfg_change




class MyLocalSearch(LocalSearch):
    """Implementation of SMAC's local search.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction
    challengers : int, defaults to 5000
        Number of challengers.
    max_steps: int | None, defaults to None
        Maximum number of iterations that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        Number of steps during a plateau walk before local search terminates.
    vectorization_min_obtain : int, defaults to 2
        Minimal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC.
    vectorization_max_obtain : int, defaults to 64
        Maximal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC.
    seed : int, defaults to 0
    """

    def __init__(
            self,
            configspace: ConfigurationSpace,
            acquisition_function: AbstractAcquisitionFunction | None = None,
            challengers: int = 5000,
            max_steps: int | None = None,
            n_steps_plateau_walk: int = 10,
            vectorization_min_obtain: int = 2,
            vectorization_max_obtain: int = 64,
            seed: int = 0,
            path_to_run=None
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        self._max_steps = max_steps
        self._n_steps_plateau_walk = n_steps_plateau_walk
        self._vectorization_min_obtain = vectorization_min_obtain
        self._vectorization_max_obtain = vectorization_max_obtain
        self.path_to_run = path_to_run

    def _maximize(
            self,
            previous_configs: list[Configuration],
            n_points: int,
            additional_start_points: list[tuple[float, Configuration]] | None = None,
    ) -> list[tuple[float, Configuration]]:
        """Start a local search from the given startpoints. Iteratively collect neighbours
        using Configspace.utils.get_one_exchange_neighbourhood and evaluate them.
        If the new config is better than the current best, the local search is coninued from the
        new config.

        Quit if either the max number of steps is reached or
        no neighbor with a higher improvement was found or the number of local steps self._n_steps_plateau_walk
        for each of the starting point is depleted.


        Parameters
        ----------
        previous_configs : list[Configuration]
            Previous configuration (e.g., from the runhistory).
        n_points : int
            Number of initial points to be generated.
        additional_start_points : list[tuple[float, Configuration]] | None
            Additional starting points.

        Returns
        -------
        list[Configuration]
            Final candidates.
        """
        hps = self._calculate_hpi(previous_configs)
        # reduced_cfgs = self._convert_full_to_reduced_configs(previous_configs, hps)
        # converted_configs, cfg_dict = self._set_irrelevant_to_default(previous_configs, hps)
        # additional_start_points = [(point, self._set_irrelevant_to_default([cfg], hps)[0]) for (point, cfg) in additional_start_points]
        # init_points = self._get_initial_points(converted_configs, n_points, additional_start_points) #TODO X needs to be reduced
        init_points = self._get_initial_points(previous_configs, n_points, additional_start_points) #TODO X needs to be reduced
        converted_configs, cfg_dict = self._set_irrelevant_to_default(init_points, hps)
        configs_acq = self._search(converted_configs)
        # configs_acq = self._convert_reduced_to_full_configs_hpi(configs_acq, previous_configs[0].config_space)

        # Shuffle for random tie-break
        configs_acq = self.reverse_configs(configs_acq, cfg_dict)
        self._rng.shuffle(configs_acq)

        # Sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])
        for a, inc in configs_acq:
            inc.origin = "Acquisition Function Maximizer: Local Search"

        configs_acq = self.reverse_configs(configs_acq, cfg_dict)

        return configs_acq

    def _calculate_hpi(self, configs):
        weighting = self._acquisition_function._theta
        X = convert_configurations_to_array(configs)
        Y = self._acquisition_function._model.predict(X)
        # TODO get meta info or get path to current run
        # run = SMAC3v2Run(name='name', configspace=previous_configs[0].config_space, objectives=['1-accuracy', 'time'], meta=self.meta)
        print(self.path_to_run)
        run = SMAC3v2Run.from_path(self.path_to_run)
        fanova = fANOVAWeighted(run)
        fanova.train_model(X, Y, weighting)
        df_res = pd.DataFrame(fanova.get_importances(hp_names=None)).loc[0:1].T.reset_index()
        hps = df_res[df_res[0] > df_res[0].quantile(0.5)][
            'index'].to_list()  # select hps over the 50% quantile of importance
        return hps

    def _convert_full_to_reduced_configs(self, previous_configs, hps):
        # create reduced cs
        cs = ConfigurationSpace()
        full_cs = previous_configs[0].config_space
        cs.add_hyperparameters([full_cs[hp] for hp in hps])

        # create list of known configs with reduced space
        reduced_cfgs = [Configuration(cs, dict((hp, cfg[hp]) for hp in hps)) for cfg in previous_configs]
        return reduced_cfgs

    def _set_irrelevant_to_default(self, previous_configs, hps):
        hps_default = list(set(previous_configs[0].config_space.get_hyperparameter_names())-set(hps))
        def_cfg = previous_configs[0].config_space.get_default_configuration()
        for hp in hps_default:
            converted_configs = [update(cfg, hp, def_cfg[hp]) for cfg in previous_configs]
        cfg_dict = dict(zip(converted_configs, previous_configs))
        return converted_configs, cfg_dict

    def reverse_configs(self, configs_acq, cfg_dict):
        cfgs = []
        for c, cfg in configs_acq:
            if cfg in cfg_dict:
                cfgs.append((c, cfg_dict[cfg]))
            else:
                cfgs.append((c,cfg))
        return cfgs

    def _convert_reduced_to_full_configs_hpi(self, reduced_configs, full_cs):
        #TODO test
        full_configs = [Configuration(full_cs, cfg) for cfg in reduced_configs]
        return full_configs

    def _search(
        self,
        start_points: list[Configuration],
    ) -> list[tuple[float, Configuration]]:
        """Optimize the acquisition function.

        Execution:
        1. Neighbour generation strategy for each of the starting points is according to
        ConfigSpace.utils.get_one_exchange_neighbourhood.
        2. Each of the starting points create a local search, that can be active.
        if it is active, request a neighbour of its neightbourhood and evaluate it.
        3. Comparing the acquisition function of the neighbors with the acquisition value of the
        candidate.
        If it improved, then the candidate is replaced by the neighbor. And this candidate is
        investigated again with two new neighbours.
        If it did not improve, it is investigated with twice as many new neighbours
        (at most self._vectorization_max_obtain neighbours).
        The local search for a starting point is stopped if the number of evaluations is larger
        than self._n_steps_plateau_walk.


        Parameters
        ----------
        start_points : list[Configuration]
            Starting points for the search.

        Returns
        -------
        list[tuple[float, Configuration]]
            Candidates with their acquisition function value. (acq value, candidate)
        """
        assert self._acquisition_function is not None

        # Gather data structure for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]

        candidates = start_points
        # Compute the acquisition value of the candidates
        num_candidates = len(candidates)
        acq_val_candidates_ = self._acquisition_function(candidates)

        if num_candidates == 1:
            acq_val_candidates = [acq_val_candidates_[0][0]]
        else:
            acq_val_candidates = [a[0] for a in acq_val_candidates_]

        # Set up additional variables required to do vectorized local search:
        # whether the i-th local search is still running
        active = [True] * num_candidates
        # number of plateau walks of the i-th local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = [0] * num_candidates
        # tracking the number of steps for logging purposes
        local_search_steps = [0] * num_candidates
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = [0] * num_candidates
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = [0] * num_candidates
        # how many neighbors were obtained for the i-th local search. Important to map the individual acquisition
        # function values to the correct local search run
        obtain_n = [self._vectorization_min_obtain] * num_candidates
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(candidates):
            neighborhood_iterators.append(
                # get_one_exchange_neighbourhood implementational details:
                # https://github.com/automl/ConfigSpace/blob/05ab3da2a06c084ba920e8e4e3f62f2e87e81442/ConfigSpace/util.pyx#L95
                # Return all configurations in a one-exchange neighborhood.
                #
                #     The method is implemented as defined by:
                #     Frank Hutter, Holger H. Hoos and Kevin Leyton-Brown
                #     Sequential Model-Based Optimization for General Algorithm Configuration
                #     In Proceedings of the conference on Learning and Intelligent
                #     Optimization(LION 5)
                get_one_exchange_neighbourhood(inc, seed=self._rng.randint(low=0, high=100000))
            )
            local_search_steps[i] += 1

        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq: list[list[Configuration]] = [[] for _ in range(num_candidates)]

        num_iters = 0
        while np.any(active):
            num_iters += 1
            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = [False] * num_candidates
            # Used to request a new neighborhood for the candidates of the i-th local search
            new_neighborhood = [False] * num_candidates

            # gather all neighbors
            neighbors = []
            for i, neighborhood_iterator in enumerate(neighborhood_iterators):
                if active[i]:
                    neighbors_for_i = []
                    for j in range(obtain_n[i]):
                        try:
                            n = next(neighborhood_iterator)
                            neighbors_generated[i] += 1
                            neighbors_for_i.append(n)
                        except ValueError as e:
                            # `neighborhood_iterator` raises `ValueError` with some probability when it reaches
                            # an invalid configuration.
                            logger.debug(e)
                            new_neighborhood[i] = True
                        except StopIteration:
                            new_neighborhood[i] = True
                            break
                    obtain_n[i] = len(neighbors_for_i)
                    neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                start_time = time.time()
                acq_val = self._acquisition_function(neighbors)
                end_time = time.time()
                times.append(end_time - start_time)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = np.asarray([acq_val])

                # Comparing the acquisition function of the neighbors with the acquisition value of the candidate
                acq_index = 0
                # Iterating the all i local searches
                for i in range(num_candidates):
                    if not active[i]:
                        continue

                    # And for each local search we know how many neighbors we obtained
                    for j in range(obtain_n[i]):
                        # The next line is only true if there was an improvement and we basically need to iterate to
                        # the i+1-th local search
                        if improved[i]:
                            acq_index += 1
                        else:
                            neighbors_looked_at[i] += 1

                            # Found a better configuration
                            if acq_val[acq_index] > acq_val_candidates[i]:
                                is_valid = False
                                try:
                                    neighbors[acq_index].is_valid_configuration()
                                    is_valid = True
                                except (ValueError, ForbiddenValueError) as e:
                                    logger.debug("Local search %d: %s", i, e)

                                if is_valid:
                                    # We comment this as it just spams the log
                                    # logger.debug(
                                    #     "Local search %d: Switch to one of the neighbors (after %d configurations).",
                                    #     i,
                                    #     neighbors_looked_at[i],
                                    # )
                                    candidates[i] = neighbors[acq_index]
                                    acq_val_candidates[i] = acq_val[acq_index]
                                    new_neighborhood[i] = True
                                    improved[i] = True
                                    local_search_steps[i] += 1
                                    neighbors_w_equal_acq[i] = []
                                    obtain_n[i] = 1
                            # Found an equally well performing configuration, keeping it for plateau walking
                            elif acq_val[acq_index] == acq_val_candidates[i]:
                                neighbors_w_equal_acq[i].append(neighbors[acq_index])

                            acq_index += 1

            # Now we check whether we need to create new neighborhoods and whether we need to increase the number of
            # plateau walks for one of the local searches. Also disables local searches if the number of plateau walks
            # is reached (and all being switched off is the termination criterion).
            for i in range(num_candidates):
                if not active[i]:
                    continue

                if obtain_n[i] == 0 or improved[i]:
                    obtain_n[i] = 2
                else:
                    obtain_n[i] = obtain_n[i] * 2
                    obtain_n[i] = min(obtain_n[i], self._vectorization_max_obtain)

                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < self._n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            candidates[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= self._n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        candidates[i],
                        seed=self._rng.randint(low=0, high=100000),
                    )

        logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps,
            neighbors_looked_at,
            np.mean(times),
        )

        return [(a, i) for a, i in zip(acq_val_candidates, candidates)]