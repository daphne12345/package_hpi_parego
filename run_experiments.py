from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.multi_objective.parego import ParEGO
from my_config_selector import MyConfigSelector
from my_ei_hpi import MyEI
from my_local_and_random_search import MyLocalAndSortedRandomSearch

if __name__ == "__main__":
    smac_tuner = None

    # Define the smac scenario with the respective configspace, 10000 trials, and 50 epochs
    scenario = Scenario(
        smac_tuner.configspace,
        objectives=['1-accuracy','energy'],
        walltime_limit=1000000,
        n_trials=20,
        n_workers=1,
        max_budget=50,
        seed=0,
        deterministic=True #only one seed
    )

    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    multi_objective_algorithm = ParEGO(scenario, seed=scenario.seed)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=2)
    my_acquisition_maximizer = MyLocalAndSortedRandomSearch(smac_tuner.configspace, path_to_run=scenario.output_directory, seed=scenario.seed)
    my_acquisition_function = MyEI()
    my_config_selector =  MyConfigSelector(scenario)

    smac = HPOFacade(
        scenario,
        smac_tuner.train,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        acquisition_maximizer = my_acquisition_maximizer,
        acquisition_function=my_acquisition_function,
        config_selector=my_config_selector,
        overwrite=False
    )

    incumbents = smac.optimize()



