import math
from collections import Counter, defaultdict

import scipy

from .base import Splitter
from ..utils import BranchFactory


class IOLINSplitter(Splitter):
    """Numeric attribute observer for classification tasks that is based on
    a Binary Search Tree.

    This algorithm[^1] is also referred to as exhaustive attribute observer,
    since it ends up storing all the observations between split attempts[^2].

    This splitter cannot perform probability density estimations, so it does not work well
    when coupled with tree leaves using naive bayes models.

    References
    ----------
    [^1]: Domingos, P. and Hulten, G., 2000, August. Mining high-speed data streams.
    In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery
    and data mining (pp. 71-80).
    [^2]: Pfahringer, B., Holmes, G. and Kirkby, R., 2008, May. Handling numeric attributes in
    hoeffding trees. In Pacific-Asia Conference on Knowledge Discovery and Data Mining
    (pp. 296-307). Springer, Berlin, Heidelberg.
    """

    def __init__(self):
        super().__init__()
        self._root = None
        self.total_weight = 0.0
        self.tree_weight = 0.0
        self.alpha = 0.001

    def update(self, att_val, target_val, sample_weight):
        if att_val is None:
            return
        else:
            self.total_weight += sample_weight
            if self._root is None:
                self._root = ExhaustiveNode(att_val, target_val, sample_weight)
            else:
                self._root.insert_value(att_val, target_val, sample_weight)

    def set_tree_weight_and_alpha(self, tree_weight, alpha):
        self.tree_weight = tree_weight
        self.alpha = alpha

    def cond_proba(self, att_val, target_val):
        """The underlying data structure used to monitor the input does not allow probability
        density estimations. Hence, it always returns zero for any given input."""
        return 0.0

    def best_evaluated_split_suggestion(
            self,
            criterion,
            pre_split_dist,
            att_idx,
            binary_only,
    ):
        current_best_option = BranchFactory()

        return self._search_for_best_split_options(
            current_node=self._root,
            current_best_option=current_best_option,
            actual_parent_left=None,
            parent_left=None,
            parent_right=None,
            left_child=False,
            criterion=criterion,
            pre_split_dist=pre_split_dist,
            att_idx=att_idx,
        )

    def _search_for_best_split_option(
        self,
        current_node,
        current_best_option,
        actual_parent_left,
        parent_left,
        parent_right,
        left_child,
        criterion,
        pre_split_dist,
        att_idx,
        best_node,
    ):
        if current_node is None:
            return current_best_option, best_node

        left_dist = {}
        right_dist = {}

        if parent_left is None:
            left_dist.update(
                dict(Counter(left_dist) + Counter(current_node.class_count_left))
            )
            right_dist.update(
                dict(Counter(right_dist) + Counter(current_node.class_count_right))
            )
        else:
            left_dist.update(dict(Counter(left_dist) + Counter(parent_left)))
            right_dist.update(dict(Counter(right_dist) + Counter(parent_right)))

            if left_child:
                # get the exact statistics of the parent value
                exact_parent_dist = {}
                exact_parent_dist.update(
                    dict(Counter(exact_parent_dist) + Counter(actual_parent_left))
                )
                exact_parent_dist.update(
                    dict(
                        Counter(exact_parent_dist)
                        - Counter(current_node.class_count_left)
                    )
                )
                exact_parent_dist.update(
                    dict(
                        Counter(exact_parent_dist)
                        - Counter(current_node.class_count_right)
                    )
                )

                # move the subtrees
                left_dist.update(
                    dict(Counter(left_dist) - Counter(current_node.class_count_right))
                )
                right_dist.update(
                    dict(Counter(right_dist) + Counter(current_node.class_count_right))
                )

                # move the exact value from the parent
                right_dist.update(
                    dict(Counter(right_dist) + Counter(exact_parent_dist))
                )
                left_dist.update(dict(Counter(left_dist) - Counter(exact_parent_dist)))
            else:
                left_dist.update(
                    dict(Counter(left_dist) + Counter(current_node.class_count_left))
                )
                right_dist.update(
                    dict(Counter(right_dist) - Counter(current_node.class_count_left))
                )

        post_split_dists = [left_dist, right_dist]
        merit = criterion.merit_of_split(pre_split_dist, post_split_dists)

        if merit > current_best_option.merit:
            current_best_option = BranchFactory(
                merit,
                att_idx,
                current_node.cut_point,
                post_split_dists,
                multiway_split=True,
            )
            best_node = current_node

        current_best_option, best_node = self._search_for_best_split_option(
            current_node=current_node._left,  # noqa
            current_best_option=current_best_option,
            actual_parent_left=current_node.class_count_left,
            parent_left=post_split_dists[0],
            parent_right=post_split_dists[1],
            left_child=True,
            criterion=criterion,
            pre_split_dist=pre_split_dist,
            att_idx=att_idx,
            best_node=best_node,
        )

        current_best_option, best_node = self._search_for_best_split_option(
            current_node=current_node._right,  # noqa
            current_best_option=current_best_option,
            actual_parent_left=current_node.class_count_left,
            parent_left=post_split_dists[0],
            parent_right=post_split_dists[1],
            left_child=False,
            criterion=criterion,
            pre_split_dist=pre_split_dist,
            att_idx=att_idx,
            best_node=best_node,
        )

        return current_best_option, best_node

    def _search_for_best_split_options(
            self,
            current_node,
            current_best_option,
            actual_parent_left,
            parent_left,
            parent_right,
            left_child,
            criterion,
            pre_split_dist,
            att_idx,
    ):
        current_best_option = BranchFactory()

        if current_node is None:
            return current_best_option

        current_best_option, best_node = self._search_for_best_split_option(
            current_node=current_node,
            current_best_option=current_best_option,
            actual_parent_left=None,
            parent_left=None,
            parent_right=None,
            left_child=False,
            criterion=criterion,
            pre_split_dist=pre_split_dist,
            att_idx=att_idx,
            best_node=None,
        )
        if not isinstance(current_best_option.split_info, list):
            current_best_option.split_info = [current_best_option.split_info]

        mutual_info = current_best_option.merit
        if mutual_info > 0:
            pass
            p_node = self.total_weight / self.tree_weight

            mutual_info *= p_node

            if mutual_info > 0:
                likelihood_ratio = (
                        2 * math.log(2) * int(self.total_weight) * mutual_info
                )
                num_attr_values = len(current_best_option.children_stats)
                deg_freedom = num_attr_values - 1
                critical_value = scipy.stats.chi2.ppf(1 - self.alpha, deg_freedom)
                if likelihood_ratio < critical_value:
                    current_best_option.merit = -math.inf
        else:
            return current_best_option

        if len(current_best_option.children_stats[0]) > 1:
            left_split = self._search_for_best_split_options(
                current_node=best_node._left,
                current_best_option=None,
                actual_parent_left=None,
                parent_left=None,
                parent_right=None,
                left_child=False,
                criterion=criterion,
                pre_split_dist=current_best_option.children_stats[0],
                att_idx=att_idx,
            )
            if left_split.merit > 0:
                # current_best_option.merit += left_split.merit
                current_best_option.children_stats = (
                        left_split.children_stats + current_best_option.children_stats[
                                                    1:]
                )
                current_best_option.split_info = (
                        left_split.split_info + current_best_option.split_info
                )
        if len(current_best_option.children_stats[1]) > 1:
            right_split = self._search_for_best_split_options(
                current_node=best_node._right,
                current_best_option=None,
                actual_parent_left=None,
                parent_left=None,
                parent_right=None,
                left_child=False,
                criterion=criterion,
                pre_split_dist=current_best_option.children_stats[1],
                att_idx=att_idx,
            )
            if right_split.merit > 0:
                # current_best_option.merit += right_split.merit
                current_best_option.children_stats = (
                        current_best_option.children_stats[
                        :-1] + right_split.children_stats
                )
                current_best_option.split_info = (
                        current_best_option.split_info + right_split.split_info
                )
        return current_best_option


class ExhaustiveNode:
    def __init__(self, att_val, target_val, sample_weight):
        self.class_count_left = defaultdict(float)
        self.class_count_right = defaultdict(float)
        self._left = None
        self._right = None

        self.cut_point = att_val
        self.class_count_left[target_val] += sample_weight

    def insert_value(self, val, label, sample_weight):
        if val == self.cut_point:
            self.class_count_left[label] += sample_weight
        elif val < self.cut_point:
            self.class_count_left[label] += sample_weight
            if self._left is None:
                self._left = ExhaustiveNode(val, label, sample_weight)
            else:
                self._left.insert_value(val, label, sample_weight)
        else:
            self.class_count_right[label] += sample_weight
            if self._right is None:
                self._right = ExhaustiveNode(val, label, sample_weight)
            else:
                self._right.insert_value(val, label, sample_weight)
