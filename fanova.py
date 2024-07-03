from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.evaluators.fanova import fANOVA
from deepcave.runs import AbstractRun


class fANOVAWeighted(fANOVA):
    """
    Calculate and provide midpoints and sizes from the forest's split values in order to get the marginals.
    Overriden to train the random forest with an arbitrary weighting of the objectives.
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        super().__init__(run)
        self.n_trees = 100

    def train_model(
            self,
            X, Y, weighting
    ) -> None:
        """
        Train a FANOVA Forest model where the objectives are weighted by the input weighting.
        :param group: the runs as group
        :param df: dataframe containing the encoded data
        :param objectives_normed: the normalized objective names as a list of strings
        :param weighting: the weighting as list
        """
        # X = df[group.configspace.get_hyperparameter_names()].to_numpy()
        Y = sum(obj * weighting for obj, weighting in zip(Y, weighting))

        self._model = FanovaForest(self.cs, n_trees=self.n_trees, seed=0)
        self._model.train(X, Y)
