from __future__ import annotations

import time
import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.multi_objective.parego import ParEGO

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

from hpi_parego.my_config_selector import MyConfigSelector
from hpi_parego.my_ei_hpi import MyEI
from hpi_parego.my_local_and_random_search_configspace import MyLocalAndSortedRandomSearchConfigSpace
import random 

digits = load_digits()


class MLP:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical("activation", ["logistic", "tanh", "relu"], default="tanh")
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

        cs.add_hyperparameters([n_layer, n_neurons, activation, solver, batch_size, learning_rate, learning_rate_init])

        # use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        # use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
        # use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

        # We can also add multiple conditions on hyperparameters at once:
        # cs.add_conditions([use_lr, use_batch_size, use_lr_init])

        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 10) -> dict[str, float]:
        # lr = config["learning_rate"] if config["learning_rate"] else  "constant"
        # lr_init = config["learning_rate_init"] if config["learning_rate_init"] else  0.001
        # batch_size = config["batch_size"] if config["batch_size"] else  200

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = MLPClassifier(
                hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                solver=config["solver"],
                batch_size=config["batch_size"],
                activation=config["activation"],
                learning_rate=config["learning_rate"],
                learning_rate_init=config["learning_rate_init"],
                max_iter=int(np.ceil(budget)),
                random_state=seed,
            )

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
            score = cross_val_score(classifier, digits.data, digits.target, cv=cv, error_score="raise")

        return {
            "1 - accuracy": 1 - np.mean(score),
            "time": time.time() - start_time,
        }


if __name__ == "__main__":
    mlp = MLP()
    objectives = ["1 - accuracy", "time"]

    # Define our environment variables
    scenario = Scenario(
        mlp.configspace,
        objectives=objectives,
        walltime_limit=300,  # After 30 seconds, we stop the hyperparameter optimization
        n_trials=100,  # Evaluate max 200 different trials
        n_workers=1,
        name='smac3output',
        output_directory='test',
        seed=0
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    multi_objective_algorithm = ParEGO(scenario)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)

    my_acquisition_function = MyEI()
    my_config_selector = MyConfigSelector(scenario)
    my_maximizer = MyLocalAndSortedRandomSearchConfigSpace(mlp.configspace, my_acquisition_function, path_to_run=scenario.output_directory, hpi_method='fanova',
        adjust_previous_cfgs='no', rnd_aug_pc=True, adjust_cs='random', cs_proba_hpi=True, gt_hpi=False, multi_objective_algorithm=multi_objective_algorithm, thresh=[0.75])
    

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        overwrite=True,
        acquisition_function=my_acquisition_function,
        acquisition_maximizer=my_maximizer,
        config_selector=my_config_selector
    )

    # Let's optimize
    incumbents = smac.optimize()
    # Get cost of default configuration
    default_cost = smac.validate(mlp.configspace.get_default_configuration())
    print(f"Validated costs from default config: \n--- {default_cost}\n")

    print("Validated costs from the Pareto front (incumbents):")
    for incumbent in incumbents:
        cost = smac.validate(incumbent)
        print("---", cost)
    

        
    # baseline

    # Define our environment variables
    # scenario = Scenario(
    #     mlp.configspace,
    #     objectives=objectives,
    #     walltime_limit=300,  # After 30 seconds, we stop the hyperparameter optimization
    #     n_trials=200,  # Evaluate max 200 different trials
    #     n_workers=1,
    #     name='baseline',
    #     output_directory='test',
    #     seed=0
    # )

    # We want to run five random configurations before starting the optimization.
    # initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    # multi_objective_algorithm = ParEGO(scenario)
    # intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)

    # Create our SMAC object and pass the scenario and the train method
    # smac = HPOFacade(
    #     scenario,
    #     mlp.train,
    #     initial_design=initial_design,
    #     multi_objective_algorithm=multi_objective_algorithm,
    #     intensifier=intensifier,
    #     overwrite=True,
    # )

    # Let's optimize
    # incumbents = smac.optimize()

    # Get cost of default configuration
    # default_cost = smac.validate(mlp.configspace.get_default_configuration())
    # print(f"Validated costs from default config: \n--- {default_cost}\n")

    # print("Baseline: Validated costs from the Pareto front (incumbents):")
    # for incumbent in incumbents:
    #     cost = smac.validate(incumbent)
    #     print("---", cost)
