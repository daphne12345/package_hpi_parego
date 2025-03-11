from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, NormalFloatHyperparameter, UniformIntegerHyperparameter, NormalIntegerHyperparameter, Constant

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acqusition_maximizer import AbstractAcquisitionMaximizer
from smac.acquisition.maximizer.local_search import LocalSearch
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.utils.logging import get_logger

from hpi_parego.fanova import fANOVAWeighted
from smac.utils.configspace import convert_configurations_to_array
from deepcave.runs.converters.smac3v2 import SMAC3v2Run
import pandas as pd
import copy
from typing import Dict
from hypershap import HPIGame
import shapiq
import numpy as np

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class MyLocalAndSortedRandomSearchConfigSpace(AbstractAcquisitionMaximizer):
    """Implement SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points according to the acquisition
    function, uses the acquisition function to sort randomly sampled configurations.
    Random configurations are interleaved by the main SMAC code.

    The Random configurations are interleaved to circumvent issues from a constant prediction
    from the Random Forest model at the beginning of the optimization process.

    Parameters
    ----------
    configspace : ConfigurationSpace
    uniform_configspace : ConfigurationSpace
        A version of the user-defined ConfigurationSpace where all parameters are uniform (or have their weights removed
        in the case of a categorical hyperparameter). Can optionally be given and sampling ratios be defined via the
        `prior_sampling_fraction` parameter.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
    challengers : int, defaults to 5000
        Number of challengers.
    max_steps: int | None, defaults to None
        [LocalSearch] Maximum number of steps that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        [LocalSearch] number of steps during a plateau walk before local search terminates.
    local_search_iterations: int, defauts to 10
        [Local Search] number of local search iterations.
    prior_sampling_fraction: float, defaults to 0.5
        The ratio of random samples that are taken from the user-defined ConfigurationSpace, as opposed to the uniform
        version (needs `uniform_configspace`to be defined).
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
        uniform_configspace: ConfigurationSpace | None = None,
        prior_sampling_fraction: float | None = None,
        adjust_cs=True,
        constant=True,
        hpi_method='hypershap',
        adjust_previous_cfgs=False,
        set_to_default=False,
        path_to_run=None
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )
        print('PATH', path_to_run)

        self.path_to_run=path_to_run
        self._original_cs = copy.deepcopy(configspace)
        self.adjust_cs = adjust_cs # whether to adjust the configspace for sampling
        self.constant = constant # whether to set the unimportant hyperparameters to the cosntant default value or a distribution with very unlikely other values
        self.hpi=hpi_method # fanova or hypershap
        self.adjust_previous_cfgs = adjust_previous_cfgs # whther to adjust the previous configs for search
        self.set_to_default = set_to_default # whether to adjust the previous configs by setting the unimportant hps to default or random augmentation
        

        if uniform_configspace is not None and prior_sampling_fraction is None:
            prior_sampling_fraction = 0.5
        if uniform_configspace is None and prior_sampling_fraction is not None:
            raise ValueError("If `prior_sampling_fraction` is given, `uniform_configspace` must be defined.")
        if uniform_configspace is not None and prior_sampling_fraction is not None:
            self._prior_random_search = RandomSearch(
                acquisition_function=acquisition_function,
                configspace=configspace,
                seed=seed,
            )

            self._uniform_random_search = RandomSearch(
                acquisition_function=acquisition_function,
                configspace=uniform_configspace,
                seed=seed,
            )
        else:
            self._random_search = RandomSearch(
                configspace=configspace,
                acquisition_function=acquisition_function,
                seed=seed,
            )

        self._local_search = LocalSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed
        )

        self._local_search_iterations = local_search_iterations
        self._prior_sampling_fraction = prior_sampling_fraction
        self._uniform_configspace = uniform_configspace
        

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:  # noqa: D102
        """Returns the used acquisition function."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function
        if self._uniform_configspace is not None:
            self._prior_random_search._acquisition_function = acquisition_function
            self._uniform_random_search._acquisition_function = acquisition_function
        else:
            self._random_search._acquisition_function = acquisition_function
        self._local_search._acquisition_function = acquisition_function

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        if self._uniform_configspace is None:
            meta.update(
                {
                    "random_search": self._random_search.meta,
                    "local_search": self._local_search.meta,
                }
            )
        else:
            meta.update(
                {
                    "prior_random_search": self._prior_random_search.meta,
                    "uniform_random_search": self._uniform_random_search.meta,
                    "local_search": self._local_search.meta,
                }
            )

        return meta
    
    def _calculate_hpi_fanova(self, configs):
        """Calcuulates the HPI based on fANOVA and return the top 50% quantile of important hyperparameters.

        Args:
            configs (_type_): _description_

        Returns:
            _type_: list of important hps
        """
        weighting = self._acquisition_function._theta
        X = convert_configurations_to_array(configs)
        Y = self._acquisition_function._model.predict(X)

        print('Path', self.path_to_run)
        run = SMAC3v2Run.from_path(self.path_to_run)
        fanova = fANOVAWeighted(run)
        fanova.train_model(X, Y, weighting)
        df_res = pd.DataFrame(fanova.get_importances(hp_names=None)).loc[0:1].T.reset_index()
        important_hps = df_res[df_res[0] > df_res[0].quantile(0.5)]['index'].to_list()  # select hps over the 50% quantile of importance
        # hps = df_res.sort_values(by=0, ascending=False).head(df_res.shape[0]//2)['index'].to_list()  # select better half of hps
        return important_hps
    
    def _calculate_hpi_hypershap(self, previous_configs):
        """Calcuulates the HPI based on fANOVA and return the top 50% quantile of important hyperparameters.

        Args:
            configs (_type_): _description_

        Returns:
            _type_: list of important hps
        """
        hpo_game = HPIGame(self._configspace, previous_configs, model=self._acquisition_function._model, weighting=self._acquisition_function._theta)
        # set up the computer
        if hpo_game.n_players < 15:
            computer = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
            mi_values = computer(index="Moebius", order=hpo_game.n_players)  # compute Moebius values
        else:
            approximator = shapiq.KernelSHAPIQ(n=hpo_game.n_players, max_order=2, index="k-SII")
            mi_values = approximator.approximate(budget=100*hpo_game.n_players, game=hpo_game)
        thresh = np.quantile(mi_values.values, 0.5)
        coas = [(co, len(co[0])) for co in (mi_values.get_top_k(10).dict_values.items()) if co[1]>=thresh]
        if len(coas)==0:
            return []
        min_coa = list(min(coas, key=lambda x: x[1])[0][0])
        important_hps = [self._configspace.get_hyperparameter_names()[i] for i in min_coa]
        return important_hps
       
    
    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        
        # TODO calculates the most important hps and sets the rest to be very unlikely in the configspaces.
        logger.info(self.hpi)
        if self.hpi=='fanova':
            important_hps = self._calculate_hpi_fanova(previous_configs)
        else:
            important_hps = self._calculate_hpi_hypershap(previous_configs)
        if len(important_hps) > 0:
            if self.adjust_cs:
                self.adjust_configspace(important_hps)
            if self.adjust_previous_cfgs:
                previous_configs = self.adjust_previous_configs(previous_configs, important_hps)

        if self._uniform_configspace is not None and self._prior_sampling_fraction is not None:
            # Get configurations sorted by acquisition function value
            next_configs_by_prior_random_search_sorted = self._prior_random_search._maximize(
                previous_configs,
                round(n_points * self._prior_sampling_fraction),
                _sorted=True,
            )

            # Get configurations sorted by acquisition function value
            next_configs_by_uniform_random_search_sorted = self._uniform_random_search._maximize(
                previous_configs,
                round(n_points * (1 - self._prior_sampling_fraction)),
                _sorted=True,
            )
            next_configs_by_random_search_sorted = (
                next_configs_by_uniform_random_search_sorted + next_configs_by_prior_random_search_sorted
            )
            next_configs_by_random_search_sorted.sort(reverse=True, key=lambda x: x[0])
        else:
            # Get configurations sorted by acquisition function value
            next_configs_by_random_search_sorted = self._random_search._maximize(
                previous_configs=previous_configs,
                n_points=n_points,
                _sorted=True,
            )

        # Choose the best self._local_search_iterations random configs to start the local search, and choose only
        # incumbent from previous configs
        random_starting_points = next_configs_by_random_search_sorted[: self._local_search_iterations]
        next_configs_by_local_search = self._local_search._maximize(
            previous_configs=previous_configs,
            n_points=self._local_search_iterations,
            additional_start_points=random_starting_points,
        )

        next_configs_by_acq_value = next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        first_five = [f"{_[0]} ({_[1].origin})" for _ in next_configs_by_acq_value[:5]]

        logger.debug(f"First 5 acquisition function values of selected configurations: \n{', '.join(first_five)}")

        return next_configs_by_acq_value
    
       
    def adjust_configspace(self, important_hps):
        """Sets all unimportant hyperparameters in all configspaces of local and random search to be very unlikely.

        Args:
            important_hps (_type_): list of important hyperpamaters
        """
        print('adjust configspace')
        cs = copy.deepcopy(self._original_cs)
        random_state = cs.random.get_state()
        new_cs = ConfigurationSpace()
        new_cs.random.set_state(random_state)

        for hp in cs.values():
            if hp.name in important_hps:
                try:
                    new_cs.add(hp)
                except:
                    new_cs.add_hyperparameter(hp)
            else:
                if self.constant:
                    new_hp = Constant(
                        name=hp.name,
                        value=hp.default_value
                    )
                else: #distribution
                    if isinstance(hp, CategoricalHyperparameter):
                        new_weights = [1 if choice == hp.default_value else 0 for choice in hp.choices]
                        new_hp = CategoricalHyperparameter(
                            name=hp.name,
                            choices=hp.choices,
                            default_value=hp.default_value,
                            weights=new_weights,
                        )
                    elif isinstance(hp, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
                        new_hp = NormalFloatHyperparameter(
                            name=hp.name,
                            lower=hp.lower,
                            upper=hp.upper,
                            mu=hp.default_value,
                            sigma=(hp.upper - hp.lower)/10000000000,
                            log=hp.log,
                        )
                    elif isinstance(hp, UniformIntegerHyperparameter):                
                        new_hp = NormalIntegerHyperparameter(
                            name=hp.name,
                            lower=hp.lower,
                            upper=hp.upper,
                            mu=hp.default_value,
                            sigma=(hp.upper - hp.lower)/10000000000,
                            log=hp.log,
                        )
                        
                    else:
                        new_hp = hp
                        print(f"Hyperparameter {hp} not supported. Using old hp values.")

                try:
                    new_cs.add(new_hp)
                except:
                    new_cs.add_hyperparameter(new_hp)
        if not self.constant:
            new_cs.add_conditions(cs.get_conditions())
        
        self._configspace = new_cs
        self._local_search._configspace = new_cs
        if self._uniform_configspace is not None and self._prior_sampling_fraction is not None:
            self._prior_random_search._configspace = new_cs
            self._uniform_random_search._configspace = new_cs
        else:  
            self._random_search._configspace = new_cs
    
    def update(self, cfg, hp, value):
        cfg_change =  copy.copy(cfg)
        try:
            cfg_change[hp] = value
        except Exception as e:
            print(e)
            print(f'Could not set {hp} with value {value}.')
        return cfg_change
    
    def adjust_previous_configs(self, previous_configs, important_hps):
        hps_unimportant = list(set(self._original_cs.get_hyperparameter_names())-set(important_hps))
        converted_configs = copy.copy(previous_configs)
        if self.set_to_default:
            default = self._original_cs.get_default_configuration()
            for hp in hps_unimportant:
                if hp in default:
                    converted_configs = [self.update(cfg, hp, default[hp]) for cfg in converted_configs if hp in cfg]
        else: # random augmentation
            for _ in range(5):
                random_cfgs = self._original_cs.sample_configuration(len(previous_configs))
                for hp in important_hps:
                    random_cfgs = [self.update(new_cfg, hp, old_cfg[hp]) for new_cfg, old_cfg in zip(random_cfgs, previous_configs)]
                converted_configs += random_cfgs
        return converted_configs