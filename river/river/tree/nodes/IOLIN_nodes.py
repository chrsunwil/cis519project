from river.utils.skmultiflow_utils import normalize_values_in_dict, round_sig_fig

from ..splitter import IOLINSplitter
from ..splitter.iolin_nominal_splitter import IOLINNominalSplitter
from ..splitter.nominal_splitter_classif import NominalSplitterClassif
from ..utils import do_naive_bayes_prediction
from .leaf import IOLINLeaf


class IOLINLeafMajorityClass(IOLINLeaf):
    """Leaf that always predicts the majority class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, parents, **kwargs):
        super().__init__(stats, depth, splitter, parents, **kwargs)

    @staticmethod
    def new_nominal_splitter():
        return IOLINNominalSplitter()

    @staticmethod
    def new_numeric_splitter():
        return IOLINSplitter()

    def update_stats(self, y, sample_weight):
        try:
            self.stats[y] += sample_weight
        except KeyError:
            self.stats[y] = sample_weight

    def prediction(self, x, *, tree=None):
        return normalize_values_in_dict(self.stats, inplace=False)

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
            Total weight seen.

        """
        return sum(self.stats.values()) if self.stats else 0

    def calculate_promise(self):
        """Calculate how likely a node is going to be split.

        A node with a (close to) pure class distribution will less likely be split.

        Returns
        -------
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self.stats.values())
        if total_seen > 0:
            return total_seen - max(self.stats.values())
        else:
            return 0

    def observed_class_distribution_is_pure(self):
        """Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
            True if observed number of classes is less than 2, False otherwise.
        """
        count = 0
        for weight in self.stats.values():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2

    def __repr__(self):
        if not self.stats:
            return ""

        text = f"Class {max(self.stats, key=self.stats.get)}:"
        for label, proba in sorted(
            normalize_values_in_dict(self.stats, inplace=False).items()
        ):
            # print(self.total_weight)
            text += f"\n\tP({label}) = {round_sig_fig(proba, significant_digits=3)}"

        return text
