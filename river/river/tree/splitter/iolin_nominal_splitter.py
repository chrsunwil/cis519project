import collections
import functools
import math

import scipy

from ..utils import BranchFactory
from .base import Splitter


class IOLINNominalSplitter(Splitter):
    """Splitter utilized to monitor nominal features in classification tasks.

    As the monitored feature is nominal, it already has well-defined partitions. Hence,
    this splitter uses dictionary structures to keep class counts for each incoming category.
    """

    def __init__(self):
        super().__init__()
        self._total_weight_observed = 0.0
        self._missing_weight_observed = 0.0
        self._att_dist_per_class = collections.defaultdict(
            functools.partial(collections.defaultdict, float)
        )
        self._att_values = set()
        self.tree_weight = 0.0
        self.alpha = 0.001

    @property
    def is_numeric(self):
        return False

    def set_tree_weight_and_alpha(self, tree_weight, alpha):
        self.tree_weight = tree_weight
        self.alpha = alpha

    def update(self, att_val, target_val, sample_weight):
        if att_val is None:
            self._missing_weight_observed += sample_weight
        else:
            self._att_values.add(att_val)
            self._att_dist_per_class[target_val][att_val] += sample_weight

        self._total_weight_observed += sample_weight

    def cond_proba(self, att_val, target_val):
        class_dist = self._att_dist_per_class[target_val]
        value = class_dist[att_val]
        try:
            return value / sum(class_dist.values())
        except ZeroDivisionError:
            return 0.0

    def best_evaluated_split_suggestion(
        self, criterion, pre_split_dist, att_idx, binary_only
    ):
        best_suggestion = BranchFactory()

        if not binary_only:
            post_split_dist = self._class_dist_from_multiway_split()
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

            best_suggestion = BranchFactory(
                merit,
                att_idx,
                sorted(self._att_values),
                post_split_dist,
                numerical_feature=False,
                multiway_split=True,
            )

        for att_val in self._att_values:
            post_split_dist = self._class_dist_from_binary_split(att_val)
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                best_suggestion = BranchFactory(
                    merit,
                    att_idx,
                    att_val,
                    post_split_dist,
                    numerical_feature=False,
                    multiway_split=False,
                )

        current_best_option = best_suggestion

        mutual_info = current_best_option.merit

        p_node = self._total_weight_observed / self.tree_weight

        mutual_info *= p_node

        if mutual_info > 0:
            likelihood_ratio = (
                2 * math.log(2) * int(self._total_weight_observed) * mutual_info
            )
            num_attr_values = len(current_best_option.children_stats)
            deg_freedom = num_attr_values - 1
            critical_value = scipy.stats.chi2.ppf(1 - self.alpha, deg_freedom)
            if likelihood_ratio < critical_value:
                current_best_option.merit = -math.inf

        return current_best_option

    def _class_dist_from_multiway_split(self):
        resulting_dist = {}
        for class_val, att_dist in self._att_dist_per_class.items():
            for att_val, weight in att_dist.items():
                if att_val not in resulting_dist:
                    resulting_dist[att_val] = {}
                if class_val not in resulting_dist[att_val]:
                    resulting_dist[att_val][class_val] = 0.0
                resulting_dist[att_val][class_val] += weight

        sorted_keys = sorted(resulting_dist.keys())
        distributions = [dict(sorted(resulting_dist[k].items())) for k in sorted_keys]
        return distributions

    def _class_dist_from_binary_split(self, val_idx):
        equal_dist = {}
        not_equal_dist = {}
        for class_val, att_dist in self._att_dist_per_class.items():
            for att_val, weight in att_dist.items():
                if att_val == val_idx:
                    if class_val not in equal_dist:
                        equal_dist[class_val] = 0.0
                    equal_dist[class_val] += weight
                else:
                    if class_val not in not_equal_dist:
                        not_equal_dist[class_val] = 0.0
                    not_equal_dist[class_val] += weight
        return [equal_dist, not_equal_dist]
