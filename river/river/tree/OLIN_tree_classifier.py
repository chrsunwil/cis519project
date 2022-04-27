import math

import numpy as np
import scipy

from river import base, metrics

from .IOLIN_tree import IOLINTree
from .nodes.branch import DTBranch
from .nodes.IOLIN_nodes import IOLINLeafMajorityClass
from .split_criterion import IOLINInfoGainSplitCriterion
from .splitter import IOLINSplitter, Splitter


class OLINTreeClassifier(IOLINTree, base.Classifier):
    """OLIN information network classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_confidence
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold
        Threshold below which a split will be forced to break ties.
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric
        attributes should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.
    max_window
        The largest that a given window show be
    min_add_count
        The smallest increment allowed to the training period
    max_add_count
        The largest increment allowed to the training period
    inc_add_count
        How much to increase the training period when drift is not detected
    red_add_count
        How much to decrease the training period when drift is detected
    alpha
        Controls the statistical significance tests
    max_err
        Controls the max error allowed for detecting drift

    Notes
    -----
    An OLIN classifier [^1] is an information network which has been adapted for
    online learning. It is well suited to dealing with concept drift, because a
    new network can be created. The method behind OLIN is simple at it's core.
    There is a training window and a verification window. An IN is built with the
    training window. It is then verified with the verification window. If drift
    is detected in the verification window, then there will be changes in the size
    of the next verfication window. Otherwise, the verification window will increase.
    This reeduces the number of times you create a new IN. It can generally
    learn a data set more quickly than alternative classifiers, but can be
    computationally expensive because the model is recreated relatively often.
    See IOLIN for an improvement in this regard.

    References
    ----------

    [^1]: Last, M. (2002) Online classification of nonstationary data streams, Intell. Data Anal. 6(2) 129–147.

    Examples
    --------
    >>> from river import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.Agrawal(seed=42, perturbation = .01, classification_function = 2)
    >>> # Take 4000 instances from the infinite data generator
    >>> dataset = iter(gen.take(4000))

    >>> model = tree.OLINTreeClassifier(
    ...     nominal_attributes=['elevel', 'car', 'zipcode']
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 93.21%
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05,
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        binary_split: bool = False,
        max_size: int = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        max_window: int = 1000,
        min_add_count: int = 10,
        max_add_count: int = 100,
        inc_add_count: int = 10,
        red_add_count: int = 10,
        alpha: float = 0.001,
        max_err: float = 0.6,
    ):

        super().__init__(
            max_depth=max_depth,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.grace_period = grace_period
        self.split_criterion = IOLINInfoGainSplitCriterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.leaf_prediction = IOLINLeafMajorityClass
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        if splitter is None:
            # self.splitter = GaussianSplitter()
            self.splitter = IOLINSplitter()
        else:
            if not splitter.is_target_class:
                raise ValueError(
                    "The chosen splitter cannot be used in classification tasks."
                )
            self.splitter = splitter

        # To keep track of the observed classes
        self.classes: set = set()

        self.cur_depth = 0

        self.max_err = max_err

        # EQ. 7 [^1]
        numerator = scipy.stats.chi2.ppf(alpha, 1)
        H_p_err = -max_err * np.log2(max_err) - (1 - max_err) * np.log2(1 - max_err)

        denomenator = 2 * np.log(2 * (np.log2(2) - H_p_err - max_err * np.log2(1)))
        self.window = max(int(numerator / denomenator), min_add_count)

        self.add_count = 4

        self.max_window = max_window

        self.min_add_count = min_add_count
        self.max_add_count = max_add_count

        self.inc_add_count = inc_add_count
        self.red_add_count = red_add_count

        self.batch = []
        self.used_split_features = set()

        self.alpha = alpha

        self.first_attr = None
        self.first_attr_vals = set()

    @IOLINTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        self._split_criterion = IOLINInfoGainSplitCriterion

    @IOLINTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        self._leaf_prediction = IOLINLeafMajorityClass

    def _new_leaf(self, initial_stats=None, parent=None, parents=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        return IOLINLeafMajorityClass(initial_stats, depth, self.splitter, parents)

    def _new_split_criterion(self):
        split_criterion = IOLINInfoGainSplitCriterion(alpha=self.alpha)

        return split_criterion

    def _attempt_to_split(self, leaves, **kwargs):
        """Attempt to split a leaf.
        Parameters
        ----------
        leaves
            The leaves to evaluate.
        kwargs
            Other parameters passed to the new branch.
        """
        leaves_split_suggestions = []

        #Get the best split suggestions for each of the leaves to attempt to split
        for leaf in leaves:
            split_criterion = self._new_split_criterion()
            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            leaves_split_suggestions.append(best_split_suggestions)

        attributes = []

        #Find the different attributes under consideration
        for split_suggestion in leaves_split_suggestions[0]:
            attributes.append(split_suggestion.feature)

        #Remove the attributes which have already been split on
        temp_attributes = []
        for attribute in attributes:
            if attribute not in self.used_split_features:
                temp_attributes.append(attribute)

        attributes = temp_attributes

        attributes_merits = [0] * len(attributes)

        #Sum the Merits across all of the leaves in consideration
        for counter in range(len(attributes)):
            for leaf_split_suggestion in leaves_split_suggestions:
                for attribute_split in leaf_split_suggestion:
                    if attribute_split.feature == attributes[counter]:
                        attributes_merits[counter] += max(attribute_split.merit, 0)

        #Find the attribute with the highest merit
        sorted_attributes_merits = np.argsort(attributes_merits)[::-1]

        best_attr_idx = sorted_attributes_merits[0]

        return_leaves = []

        #For each of the leaves, split on the best attribute if the merit is positive
        for counter in range(len(leaves_split_suggestions)):
            for suggestion in leaves_split_suggestions[counter]:

                if suggestion.merit > 0 and (
                    suggestion.feature == attributes[best_attr_idx]
                ):

                    best_suggestion = suggestion
                    split_decision = best_suggestion

                    if split_decision.feature is None:
                        # Pre-pruning - null wins
                        leaves[counter].deactivate()
                        self._n_inactive_leaves += 1
                        self._n_active_leaves -= 1
                    else:
                        branch = self._branch_selector(
                            split_decision.numerical_feature,
                            split_decision.multiway_split,
                        )

                        #Create the new leaves from the current split node
                        new_leaves = tuple(
                            self._new_leaf(
                                {},
                                parent=leaves[counter],
                                parents=leaves[counter].parents,
                            )
                            for initial_stats in split_decision.children_stats
                        )

                        new_split = split_decision.assemble(
                            branch,
                            leaves[counter].stats,
                            leaves[counter].depth,
                            *new_leaves,
                            **kwargs
                        )

                        #Set the parent pointers of the current new nodes
                        inc = 0
                        for new_leaf in new_leaves:
                            new_leaf.parents = [(new_split, inc)]
                            inc += 1

                        self._n_active_leaves -= 1
                        self._n_active_leaves += len(new_leaves)

                        #Update the child points of the parents of the new nodes
                        parents = leaves[counter].parents

                        if parents is None:
                            self._root = new_split
                        else:
                            for parent, parent_branch in parents:
                                parent.children[parent_branch] = new_split

                        self.used_split_features.add(best_suggestion.feature)

                        if self.first_attr is None:
                            self.first_attr = best_suggestion.feature

                        #Add the new nodes to the return list so we can attempt split on them
                        return_leaves += new_leaves

        return return_leaves

    def build_IN(self, train_batch, *, sample_weight=1.0):

        #Reset the information known to the classifier
        self.classes = set()

        self._root = self._new_leaf()
        self.cur_depth = 0
        self._n_active_leaves = 1

        self.used_split_features = set()

        self.first_attr = None
        self.first_attr_vals = set()

        nodes_to_attempt_split = [
            self._root
        ]

        while (
            len(nodes_to_attempt_split) > 0
        ):  # While there is a node in the last layer which got created

            self._train_weight_seen_by_model = 0

            #Teach the new nodes the relevant data by walking through the network
            for cur_node in nodes_to_attempt_split:
                self._train_weight_seen_by_model = 0
                for (
                    x,
                    y,
                ) in train_batch:
                    self.classes.add(y)
                    if self.first_attr is not None:
                        self.first_attr_vals.add(x[self.first_attr])
                    self._train_weight_seen_by_model += sample_weight
                    if cur_node in iter(self._root.walk(x, until_leaf=False)):
                        cur_node.learn_one(x, y, sample_weight=sample_weight, tree=self)

            # attempt to split the current node. Get back the new children
            new_leaves = self._attempt_to_split(nodes_to_attempt_split)

            #Now attempte to split the new children
            nodes_to_attempt_split = new_leaves

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Train the model on instance x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Class label for sample x.
        sample_weight
            Sample weight.

        Returns
        -------
        self

        Notes
        -----
        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.
        """

        # Add to the batch
        self.batch.append((x, y))

        if len(self.batch) >= self.window + self.add_count:

            # Build new IN model
            self.build_IN(
                train_batch=self.batch[: self.window], sample_weight=sample_weight
            )

            # Get our training and validation errors
            err = metrics.Accuracy()
            for x, y in self.batch[: self.window]:
                err.update(y, self.predict_one(x))
            train_err = err.get()

            err = metrics.Accuracy()
            for x, y in self.batch[self.window :]:
                err.update(y, self.predict_one(x))
            validation_err = err.get()

            # EQ. 9 [^1]
            var_diff = (train_err * (1 - train_err)) / self.window + (
                validation_err * (1 - validation_err)
            ) / self.add_count

            # EQ. 10 [^1]
            max_diff = 2.326 * math.sqrt(var_diff)

            # detect concept drift
            if (validation_err - train_err) <= max_diff:  # No concept drift
                # New Validation and Train windows
                self.add_count = int(
                    min(
                        self.add_count * (1 + (self.inc_add_count / 100.0)),
                        self.max_add_count,
                    )
                )
                self.window = min(self.window + self.add_count, self.max_window)
                self.batch = self.batch[-self.window :]

            else:  # Yes, concept drift
                # recalculate size of window
                # EQ. 8 [^1]
                if isinstance(self._root, DTBranch):
                    NT_ip = len(self._root.children)
                else:
                    NT_ip = 0

                # added a case for concept drift when the tree is a stump
                if NT_ip == 0:
                    numerator = scipy.stats.chi2.ppf(1 - self.alpha, 1)
                    H_p_err = -self.max_err * np.log2(self.max_err) - (
                        1 - self.max_err
                    ) * np.log2(1 - self.max_err)

                    denomenator = 2 * np.log(
                        2 * (np.log2(2) - H_p_err - self.max_err * np.log2(1))
                    )
                    self.window = max(int(numerator / denomenator), self.min_add_count)
                else:

                    # math or np might be more performant
                    H_train_err = -(
                        scipy.special.xlogy(train_err, train_err)
                        + scipy.special.xlog1py(1 - train_err, -train_err)
                    ) / np.log(2)

                    ys = dict.fromkeys(self.classes, 0)

                    for _, y in self.batch[: self.window]:
                        ys[y] += 1

                    probs = np.array(list(ys.values())) / sum(ys.values())
                    #                 print(probs)
                    H_A_i = scipy.stats.entropy(probs)

                    numerator = scipy.stats.chi2.ppf(
                        1 - self.alpha, (NT_ip - 1) * (len(self.classes) - 1)
                    )

                    denomenator = 2 * np.log(
                        max(
                            2
                            * (
                                H_A_i
                                - H_train_err
                                - scipy.special.xlogy(train_err, len(self.classes) - 1)
                                / np.log(2)
                            ),
                            1.001,
                        )
                    )

                    self.window = max(
                        self.min_add_count,
                        min(int(numerator / denomenator), self.max_window),
                    )

                # Find the new validation window
                self.add_count = int(
                    max(
                        self.add_count * (1 - (self.red_add_count / 100.0)),
                        self.min_add_count,
                    )
                )

                # Update the batch data to drop the training data
                self.batch = self.batch[-self.window :]

        return self

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in self.classes}
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            proba.update(leaf.prediction(x, tree=self))
        return proba

    @property
    def _multiclass(self):
        return True
