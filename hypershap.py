import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from shapiq import Game
import ConfigSpace
from smac.utils.configspace import convert_configurations_to_array

class HPIGame(Game, ABC):
    def __init__(
        self,
        cs,
        cfgs,
        model,
        aggregator=lambda x: np.mean(np.array(x)),
        random_state=0,
        verbose = False,
    ) -> None:
        self.random_state = random_state
        self.cs = cs
        self.n_configs = len(cfgs)
        self.cfgs = cfgs
        self.aggregator = aggregator
        self._model = model
        # determine empty coalition value for normalization
        super().__init__(
            n_players=len(cs.get_hyperparameters()),
            normalization_value=self.get_default_config_performance(),
            verbose=verbose,
            normalize=True,
            player_names=[hp.name for hp in self.cs.get_hyperparameters()]
        )

    def get_default_config_performance(self) -> float:
        X = convert_configurations_to_array([self.cs.get_default_configuration()])
        Y, _ = self._model.predict_marginalized(X)
        return self.aggregator(Y)
        
    def _before_first_value_function_hook(self):
        pass

    def get_n_players(self):
        return len(self.cs.get_hyperparameters())

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        value_list = []
        for i in range(len(coalitions)):
            value_list += [self.evaluate_single_coalition(coalitions[i])]
        return np.array(value_list)

    def prepare_configs_for_coalition(self, coalition, cfgs):
        pass

    def evaluate_single_coalition(self, coalition: np.ndarray):
        if coalition.sum() == 0:
            return self.get_default_config_performance()

        cfgs = self.blind_parameters_according_to_coalition(self.cfgs, coalition)
        
        X = convert_configurations_to_array(cfgs)
        Y,_ = self._model.predict_marginalized(X)
        return self.aggregator(Y)

    def blind_parameters_according_to_coalition(self, cfgs, coalition):
        cfgs = copy.deepcopy(cfgs)
        list_of_hyperparams_to_blind = np.array(
            self.cs.get_hyperparameters()
        )[(1 - coalition).astype(bool)]

        default = self.cs.get_default_configuration()

        for cfg in cfgs:
            for key in cfg.keys():
                if key in list_of_hyperparams_to_blind and key in default.keys():
                    cfg[key] = default[key]
        return cfgs