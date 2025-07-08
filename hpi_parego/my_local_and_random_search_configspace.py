from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, NormalFloatHyperparameter, UniformIntegerHyperparameter, NormalIntegerHyperparameter, Constant

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
)
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
import random
from itertools import combinations
from collections import defaultdict
import pickle as pckl
from pathlib import Path
import json
import ast
from smac.utils.multi_objective import normalize_costs
from sklearn.metrics import f1_score

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
        configspace: ConfigurationSpace=ConfigurationSpace(),
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
        seed: int = 0,
        uniform_configspace: ConfigurationSpace | None = None,
        prior_sampling_fraction: float | None = None,
        adjust_cs='default',
        hpi_method='fanova',
        adjust_previous_cfgs='no',
        rnd_aug_pc=False,
        thresh=0.5,
        n_trials=100,
        path_to_run=None,
        cs_proba_hpi=False,
        gt_hpi=False
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )
        print('PATH', path_to_run)

        self.path_to_run = Path(path_to_run) if isinstance(path_to_run,str) else path_to_run
        self._original_cs = copy.deepcopy(configspace)
        self.adjust_cs = adjust_cs # whether to adjust the configspace for sampling ('default', 'random', 'incumbent', 'no')
        self.hpi=hpi_method # fanova or hypershap
        self.adjust_previous_cfgs = adjust_previous_cfgs # whther to adjust the previous configs for search ('true_no_retrain', 'no', 'true_retrain')
        self.rnd_aug_pc = rnd_aug_pc # whether to adjust the previous configs by setting the unimportant hps to default or random augmentation
        self.thresh = thresh # threshold or threshlist for quantile cut off for hpi
        self.n_trials = n_trials
        self.important_hps = []
        self.cs_proba_hpi = cs_proba_hpi
        self.incumbent = self._original_cs.sample_configuration()
        self.thresh_list = self.thresh if not isinstance(self.thresh, float) else None
        self.gt_hpi = gt_hpi # wether to use the ground truth hpi from the random search
        self.hps_guess = []

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
            seed=seed,
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
    
    def to_cfg(self, cfg_str):
        # return Configuration(
        #     configuration_space=self._original_cs,
        #     values={x: ast.literal_eval(cfg_str)[i] for i, x in enumerate(self._original_cs)},
        #     )
        cfg_dict = dict(zip(self._original_cs.get_hyperparameter_names(), ast.literal_eval(cfg_str)))
        return Configuration(
        configuration_space=self._original_cs,
        values=cfg_dict,
        allow_inactive_with_values=True
        )

            
    
    def get_objective_bounds(self, costs):
        min_values = np.min(costs, axis=0)
        max_values = np.max(costs, axis=0)

        objective_bounds = []
        for min_v, max_v in zip(min_values, max_values):
            objective_bounds += [(min_v, max_v)]
        return objective_bounds
    
    def convert_Y(self, y, objective_bounds):
        y = list(ast.literal_eval(y))
        y = normalize_costs(y, objective_bounds)
        #ParEGO
        theta_f = self.acquisition_function._theta * y
        return float(np.max(theta_f, axis=0) + self._acquisition_function._rho * np.sum(theta_f, axis=0))
    
    def _calculate_hpi_fanova(self, configs):
        """Calculates the HPI based on fANOVA and return the top 50% quantile of important hyperparameters.

        Args:
            configs (_type_): _description_

        Returns:
            _type_: list of important hps
        """
        if not self.gt_hpi:
            X = convert_configurations_to_array(configs)
            Y = self._acquisition_function.model.predict_marginalized(X)[0]
            return self._cal_fanova(X, Y)
        
        df_rnd = pd.read_parquet('results_random_search/logs.parquet')
        task = f"multi-objective/50/dev/{'/'.join(str(self.path_to_run).split('/')[8:-3])}"
        print('task', task)
        df_rnd = df_rnd[(df_rnd['task_id']==task) & (df_rnd['seed']==self._seed)]
        print('shape of df_rnd', df_rnd.shape)
        rnd_cfgs = df_rnd['trial_info__config'].apply(self.to_cfg).to_list()
        X = convert_configurations_to_array(rnd_cfgs)
        objective_bounds = self.get_objective_bounds(df_rnd['trial_value__cost_raw'].apply(lambda y: ast.literal_eval(y)).tolist())
        Y = df_rnd['trial_value__cost_raw'].apply(lambda y: self.convert_Y(y, objective_bounds)).to_numpy()
        hps_gt, hpis = self._cal_fanova(X, Y)
          
        X = convert_configurations_to_array(configs)
        Y = self._acquisition_function.model.predict_marginalized(X)[0]
        hps_guess, _ = self._cal_fanova(X, Y)
        self.hps_guess.append(hps_guess)
        pckl.dump(self.hps_guess, open(self.path_to_run / 'hps_guess.pckl', 'wb'))
        self.fscores = f1_score(hps_gt, hps_guess)
        pckl.dump(self.fscores, open(self.path_to_run / 'fscores_hps.pckl', 'wb'))
        return hps_gt, hpis
            

    def _cal_fanova(self, X, Y):
        run = SMAC3v2Run.from_path(self.path_to_run)
        fanova = fANOVAWeighted(run)
        fanova.train_model(X, Y)
        df_res = pd.DataFrame(fanova.get_importances(hp_names=None)).loc[0:1].T.reset_index()
        if self.thresh > 0:
            important_hps = df_res[df_res[0] > df_res[0].quantile(self.thresh)]['index'].to_list()  # select hps over the 50% quantile of importance
        else:
            important_hps = df_res['index'].to_list() 
        # hps = df_res.sort_values(by=0, ascending=False).head(df_res.shape[0]//2)['index'].to_list()  # select better half of hps
        hpis = df_res.set_index('index')[0].to_dict()
        del run, df_res, fanova
        return important_hps, hpis
    
    def _random_selection(self):
        hps = self._configspace.get_hyperparameter_names()
        n_hps =  int(np.round((1 - self.thresh) * len(hps)))
        hpis = [random.random() for i in range(len(hps))]
        hpis = np.array(hpis)/sum(hpis)
        hpis = dict(zip(hps, hpis))
        return random.sample(hps, n_hps), hpis

    
    def sum_mi_values_higher(self, values):
        frozen_values = {frozenset(k): v for k, v in values.items()}
        result = defaultdict(float, frozen_values)

        indices = sorted({i for k in frozen_values for i in k})
        size_to_keys = defaultdict(list)
        for k in frozen_values:
            size_to_keys[len(k)].append(k)

        for size in range(2, len(indices) + 1):
            for comb in combinations(indices, size):
                cset = frozenset(comb)
                result[cset] = sum(
                    frozen_values[k] for l in range(1, size)
                    for k in size_to_keys[l] if k.issubset(cset)
                )

        return {tuple(sorted(k)): v for k, v in result.items()}
    
    
    
    
    def _calculate_hpi_hypershap(self, previous_configs):
        """Calcuulates the HPI based on fANOVA and return the top 50% quantile of important hyperparameters.

        Args:
            configs (_type_): _description_

        Returns:
            _type_: list of important hps
        """
            
        hpo_game = HPIGame(self._configspace, previous_configs, model=self._acquisition_function._model)
        print('n hps:', hpo_game.n_players)
        if hpo_game.n_players <= 10:
            computer = shapiq.ExactComputer(n_players=hpo_game.n_players, game=hpo_game)
            # mi_values = computer(index="Moebius", order=hpo_game.n_players)  # compute Moebius values
            mi_values = computer.shapley_interaction(index="FSII", order=2)                     
        elif hpo_game.n_players <= 25:
            # approximator = shapiq.KernelSHAPIQ(n=hpo_game.n_players, max_order=2, index="k-SII")
            # mi_values = approximator.approximate(budget=10*hpo_game.n_players, game=hpo_game)
            approximator = shapiq.PermutationSamplingSII(n=hpo_game.n_players, max_order=2)
            mi_values = approximator.approximate(budget=10 * hpo_game.n_players, game=hpo_game)
        else:
            approximator = shapiq.PermutationSamplingSII(n=hpo_game.n_players, max_order=2)
            mi_values = approximator.approximate(budget=hpo_game.n_players, game=hpo_game)
            
        mi_values = dict(zip(mi_values.interaction_lookup, mi_values.values))
        coas = self.sum_mi_values_higher(mi_values)
        thresh = np.quantile(list(coas.values()), self.thresh) 
        thresh = max(thresh, 0)
        if thresh>0:
            coas = [(co, len(co[0])) for co in (coas.items()) if co[1]>thresh]
        if len(coas)==0:
            return []
        min_coa = list(min(coas, key=lambda x: x[1])[0][0])
        important_hps = [self._configspace.get_hyperparameter_names()[i] for i in min_coa]
        del hpo_game
        return important_hps
       
    # @profile(stream=open("memory_profile.log", "w"))
    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        
        actual_previous_configs = previous_configs.copy()
        
        # TODO calculates the most important hps and sets the rest to be very unlikely in the configspaces.
        print(self.hpi, self.adjust_cs)
        if self.adjust_cs=='incumbent' or self.adjust_cs=='rnd_inc' or self.adjust_previous_cfgs=='true_retrain':
            X = convert_configurations_to_array(previous_configs)
            # Y,_ = self._acquisition_function._model.predict(X)
            Y = self._acquisition_function.model.predict_marginalized(X)[0]
            self.incumbent = previous_configs[np.argmin(Y)]

        if self.thresh_list:
            run = SMAC3v2Run.from_path(self.path_to_run)
            current_trial = len(run.trial_keys)
            pos = min(current_trial // (self.n_trials // len(self.thresh_list)), len(self.thresh_list) - 1)
            self.thresh = self.thresh_list[pos]
            del run
        
        if self.thresh>0:
        
            hpis = []
            if self.hpi=='fanova':
                if random.random() < 0.1:
                    important_hps, hpis = self._random_selection()
                else:
                    important_hps, hpis = self._calculate_hpi_fanova(previous_configs)
            elif self.hpi=='random':
                important_hps, hpis = self._random_selection()
            else:
                if random.random() < 0.1:
                    important_hps, _ = self._random_selection()
                else:
                    important_hps = self._calculate_hpi_hypershap(previous_configs)
            self.important_hps.append(important_hps)
            pckl.dump(self.important_hps, open(self.path_to_run / 'important_hps.pckl', 'wb'))
            
            if len(important_hps) > 0:
                if self.adjust_cs!='no':
                    if self.cs_proba_hpi:
                        self.adjust_cs_hpi(hpis)
                    else:
                        self.adjust_configspace(important_hps)
                if self.adjust_previous_cfgs!='no':
                    previous_configs = self.adjust_previous_configs(previous_configs, important_hps)
                    if self.adjust_previous_cfgs=='true_retrain' or self.adjust_previous_cfgs=='true_retrain_pc':
                        if self.adjust_previous_cfgs=='true_retrain_pc':
                            previous_configs.extend(actual_previous_configs)
                        X = convert_configurations_to_array(previous_configs)
                        with (self.path_to_run / "runhistory.json").open() as json_file:
                            all_data = json.load(json_file)
                            costs = {data['config_id']: data['cost'] for data in all_data["data"]}
                        if len(costs.values())!=len(actual_previous_configs):
                            cfg_id_missing = set(costs.keys()) - set([cfg.config_id for cfg in actual_previous_configs])
                            costs = {k:v for k,v in costs.items() if k not in cfg_id_missing}
                            print(cfg_id_missing, 'cfgs missing in previous_cfgs')
                        Y = list(costs.values())
                        if self.rnd_aug_pc: #random augmentation will be 6 times longer
                            Y = 6*Y
                        if self.adjust_previous_cfgs=='true_retrain_pc':
                            Y = 2*Y
                        
                        Y = np.array([self._multi_objective_algorithm(y) for y in Y])
                        self._acquisition_function.model.train(X, Y)
                        
                        Y = self._acquisition_function.model.predict_marginalized(X)[0]
                        self._acquisition_function.update(model=self._acquisition_function.model, eta=np.min(Y), theta=self._acquisition_function._theta)

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
        
        # Check if configs are new
        new_cfgs = all([cfg not in actual_previous_configs for i, (acq_value, cfg) in enumerate(next_configs_by_local_search)])

        if not new_cfgs:
            self._configspace = self._original_cs
            logger.info("No new configurations found. Falling back to original search space.")
            next_configs_by_local_search = self._local_search._maximize(
                previous_configs=actual_previous_configs,
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
        random_state = self._original_cs.random.get_state()
        new_cs = ConfigurationSpace()
        new_cs.random.set_state(random_state)

        for hp in self._original_cs.values():
            if hp.name in important_hps:
                try:
                    new_cs.add(hp)
                except:
                    new_cs.add_hyperparameter(hp)
            else:
                if self.adjust_cs=='default':
                    hp_value = hp.default_value  
                elif self.adjust_cs=='incumbent':
                    hp_value = self.incumbent[hp.name] if hp.name in self.incumbent else hp.default_value
                else:
                    hp_value = hp.sample_value()
                if isinstance(hp, CategoricalHyperparameter):
                    new_weights = [1 if choice == hp_value else 0 for choice in hp.choices]
                    new_hp = CategoricalHyperparameter(
                        name=hp.name,
                        choices=hp.choices,
                        default_value=hp_value,
                        weights=new_weights,
                    )
                elif isinstance(hp, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
                    new_hp = NormalFloatHyperparameter(
                        name=hp.name,
                        lower=hp.lower,
                        upper=hp.upper,
                        mu=hp_value,
                        sigma=(hp.upper - hp.lower)/10000000000,
                        log=hp.log,
                    )
                elif isinstance(hp, UniformIntegerHyperparameter):                
                    new_hp = NormalIntegerHyperparameter(
                        name=hp.name,
                        lower=hp.lower,
                        upper=hp.upper,
                        mu=hp_value,
                        sigma=(hp.upper - hp.lower)/10000000000,
                        log=hp.log,
                    )
                    
                else:
                    new_hp = hp
                    print(f"Hyperparameter {hp} of type {type(hp)} not supported. Using old hp values.")

                try:
                    new_cs.add(new_hp)
                except:
                    new_cs.add_hyperparameter(new_hp)
        new_cs.add_conditions(self._original_cs.get_conditions())
        
        self._configspace = new_cs
        self._local_search._configspace = new_cs
        if self._uniform_configspace is not None and self._prior_sampling_fraction is not None:
            self._prior_random_search._configspace = new_cs
            self._uniform_random_search._configspace = new_cs
        else:  
            self._random_search._configspace = new_cs
    
    def adjust_cs_hpi(self, hpis):
        """Sets all unimportant hyperparameters in all configspaces of local and random search to be very unlikely.

        Args:
            important_hps (_type_): list of important hyperpamaters
        """
        print('adjust configspace')
        random_state = self._original_cs.random.get_state()
        new_cs = ConfigurationSpace()
        new_cs.random.set_state(random_state)

        for hp in self._original_cs.values():                
            if self.adjust_cs=='default':
                hp_value = hp.default_value  
            elif self.adjust_cs=='incumbent':
                hp_value = self.incumbent[hp.name] if hp.name in self.incumbent else hp.default_value
            else:
                hp_value = hp.sample_value()
            
            if isinstance(hp, CategoricalHyperparameter):
                new_weights = [1 if choice == hp_value else 0 for choice in hp.choices]
                new_hp = CategoricalHyperparameter(
                    name=hp.name,
                    choices=hp.choices,
                    default_value=hp_value,
                    weights=new_weights,
                )
            elif isinstance(hp, (UniformFloatHyperparameter, NormalFloatHyperparameter)):
                new_hp = NormalFloatHyperparameter(
                    name=hp.name,
                    lower=hp.lower,
                    upper=hp.upper,
                    mu=hp_value,
                    sigma=hpis[hp.name],
                    log=hp.log,
                )
            elif isinstance(hp, UniformIntegerHyperparameter):                
                new_hp = NormalIntegerHyperparameter(
                    name=hp.name,
                    lower=hp.lower,
                    upper=hp.upper,
                    mu=hp_value,
                    sigma=hpis[hp.name],
                    log=hp.log,
                )
                
            else:
                new_hp = hp
                print(f"Hyperparameter {hp} not supported. Using old hp values.")

            try:
                new_cs.add(new_hp)
            except:
                new_cs.add_hyperparameter(new_hp)
        new_cs.add_conditions(self._original_cs.get_conditions())
        
        self._configspace = new_cs
        self._local_search._configspace = new_cs
        if self._uniform_configspace is not None and self._prior_sampling_fraction is not None:
            self._prior_random_search._configspace = new_cs
            self._uniform_random_search._configspace = new_cs
        else:  
            self._random_search._configspace = new_cs
    

    def update(self, cfg, hp, old_cfg):
        cfg_change =  copy.copy(cfg)
        try:
            cfg_change[hp] = old_cfg[hp]
        except Exception as e:
            print(e)
            print(f'Could not set {hp}.')
        return cfg_change
    
    def adjust_previous_configs(self, previous_configs, important_hps):
        hps_unimportant = list(set(self._configspace.get_hyperparameter_names())-set(important_hps))
        converted_configs = list(previous_configs)
        target_cfg = self._configspace.get_default_configuration() 
        for hp in hps_unimportant:
            if hp in target_cfg:                    
                for i, cfg in enumerate(converted_configs):
                    if hp in cfg:
                        converted_configs[i] = self.update(cfg, hp, target_cfg) 
        
        for cfg in converted_configs:
            cfg.configspace = self._configspace
            cfg.origin = 'adjusted'
        
        if self.rnd_aug_pc: # random augmentation
            for _ in range(5):
                random_cfgs = self._configspace.sample_configuration(len(converted_configs))
                for hp in important_hps:
                    for i in range(len(random_cfgs)):
                        random_cfgs[i] = self.update(random_cfgs[i], hp, converted_configs[i])
                converted_configs.extend(random_cfgs)
        return converted_configs