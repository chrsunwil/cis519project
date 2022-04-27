import math

import numpy as np
import scipy

from river import base, metrics

from .IOLIN_tree import IOLINTree
from .nodes.branch import DTBranch
from .nodes.IOLIN_nodes import IOLINLeafMajorityClass
from .split_criterion import IOLINInfoGainSplitCriterion
from .splitter import IOLINSplitter, Splitter


class IOLINTreeClassifier(IOLINTree, base.Classifier):
    """IOLIN information network classifier.

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
    An IOLIN classifier [^1] is an adaptation of an OLIN classifer for
    online-learning. It is well suited to dealing with concept drift,
    because a new network can be created. However, the improvement upon
    OLIN is that IOLIN does not rebuild the network at each iteration
    when given a new training window, instead extending the existing network
    in order to save computation cost and improve speed.

    References
    ----------

    [^1]: Cohen, L., Avrahami, G., Last, M., Kandel, A. (2008) Info-fuzzy algorithms for mining dynamic data streams.
            Applied soft computing. 8 1283-1294.

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
    Accuracy: 80.92%
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
        max_window: int = 2000,
        min_add_count: int = 50,
        max_add_count: int = 400,
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

        # EQ. 7
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

        # To keep track of last hidden layer
        self.last_hidden_layer = self._root
        self.last_leaf_layer = self._root

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

        # Get the best split suggestions for each of the leaves to attempt to split
        for leaf in leaves:
            split_criterion = self._new_split_criterion()
            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            leaves_split_suggestions.append(best_split_suggestions)

        attributes = []

        # Find the different attributes under consideration
        for split_suggestion in leaves_split_suggestions[0]:
            attributes.append(split_suggestion.feature)

        # Remove the attributes which have already been split on
        temp_attributes = []
        for attribute in attributes:
            if attribute not in self.used_split_features:
                temp_attributes.append(attribute)

        attributes = temp_attributes

        attributes_merits = [0] * len(attributes)

        # Sum the Merits across all of the leaves in consideration
        for counter in range(len(attributes)):
            for leaf_split_suggestion in leaves_split_suggestions:
                for attribute_split in leaf_split_suggestion:
                    if attribute_split.feature == attributes[counter]:
                        attributes_merits[counter] += max(attribute_split.merit, 0)

        sorted_attributes_merits = np.argsort(attributes_merits)[::-1]

        # Find the attribute with the highest merit
        best_attr_idx = sorted_attributes_merits[0]

        # Find the attribute with the second-highest merit
        # For each of the leaves, store the second-best attribute if the merit is positive
        second_suggestion = None
        if len(sorted_attributes_merits) > 1:

            sec_best_attr_idx = sorted_attributes_merits[1]

            for counter in range(len(leaves_split_suggestions)):
                for suggestion in leaves_split_suggestions[counter]:

                    if suggestion.merit > 0 and (
                            suggestion.feature == attributes[sec_best_attr_idx]
                    ):
                        second_suggestion = suggestion
                        self.most_recent_second = second_suggestion.feature

        return_leaves = []

        # For each of the leaves, split on the best attribute if the merit is positive
        for counter in range(len(leaves_split_suggestions)):
            for suggestion in leaves_split_suggestions[counter]:

                if suggestion.merit > 0 and (
                        suggestion.feature == attributes[best_attr_idx]
                ):
                    best_suggestion = suggestion

                    split_decision = best_suggestion

                    self.most_recent_best = split_decision.feature

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

                        # Create the new leaves from the current split node
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

                        # Set the parent pointers of the current new nodes
                        inc = 0
                        for new_leaf in new_leaves:
                            new_leaf.parents = [(new_split, inc)]
                            inc += 1

                        self._n_active_leaves -= 1
                        self._n_active_leaves += len(new_leaves)

                        # Update the child points of the parents of the new nodes
                        parents = leaves[counter].parents

                        if parents is None:
                            self._root = new_split
                        else:
                            for parent, parent_branch in parents:
                                parent.children[parent_branch] = new_split

                        # Track the features that have been split on already
                        self.used_split_features.add(best_suggestion.feature)

                        if self.first_attr is None:
                            self.first_attr = best_suggestion.feature

                        # Add the new nodes to the return list so we can attempt split on them
                        return_leaves += new_leaves

        return return_leaves

    def build_IN(self, train_batch, *, sample_weight=1.0):
        """
        build_IN learns a new Info-Fuzzy Network on a given training window
        :param train_batch: current training window
        :param sample_weight: weight of sample
        """

        # Reset the information known to the classifier
        self.classes = set()

        self._root = self._new_leaf()
        self.cur_depth = 0
        self._n_active_leaves = 1

        self.used_split_features = set()

        self.first_attr = None
        self.first_attr_vals = set()

        nodes_to_attempt_split = [self._root]
        self.last_leaf_layer = [self._root]

        while (
                len(nodes_to_attempt_split) > 0
        ):  # While there is a node in the last layer which got created

            self._train_weight_seen_by_model = 0

            # Teach the new nodes the relevant data by walking through the network
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

            # Now attempt to split the new children
            nodes_to_attempt_split = new_leaves

            # Store the last layer added to the tree
            if len(new_leaves) > 0:
                self.last_leaf_layer = new_leaves

    def update_IN(self, train_batch, *, sample_weight=1.0):
        """
        update_IN updates an Info-Fuzzy Network on a given training window by spliltting the nodes on the final layer of
        the current network by splitting on attributes not included in the network
        :param train_batch: current training window
        :param sample_weight: weight of sample
        """

        if self.last_leaf_layer[0] == self._root:
            self.build_IN(train_batch, sample_weight=sample_weight)
            return

        self.replace_last_layer(train_batch)

        nodes_to_attempt_split = self.last_leaf_layer

        nodes_to_attempt_split = [
            self._new_leaf(
                {},
                parent=node.parents[0][0],
                parents=node.parents,
            )
            for node in nodes_to_attempt_split
        ]

        for node in nodes_to_attempt_split:
            for parent, parent_branch in node.parents:
                parent.children[parent_branch] = node

        while len(nodes_to_attempt_split) > 0:

            self._train_weight_seen_by_model = 0

            # Teach the new nodes the relevant data by walking through the network
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

            new_leaves = self._attempt_to_split(nodes_to_attempt_split)
            nodes_to_attempt_split = new_leaves

            # Store the last layer added to the network
            if len(new_leaves) > 0:
                self.last_leaf_layer = new_leaves

    def check_split_validity(self, branch, branch_parent):
        """
        Check split validity traverses the current network, pruning the tree if the previous splits are no longer
        statistically significant given the new data
        :param branch: a given branch in the network
        :param branch_parent: the parent node of branch
        """

        # calculate the conditional mutual info of the branch
        criterion = self._new_split_criterion()
        # likelihood ratio test
        children_stats = []
        for child in branch.children:
            children_stats.append(child.stats)

        mutual_info = criterion.merit_of_split(branch.stats, children_stats,
                                               node_weight=branch.total_weight,
                                               tree_weight=self._train_weight_seen_by_model)

        p_node = branch.total_weight / self._train_weight_seen_by_model
        mutual_info *= p_node

        # perform likelihood ratio-test
        if mutual_info > 0:
            likelihood_ratio = (
                    2 * math.log(2) * int(branch.total_weight) * mutual_info
            )
            num_attr_values = len(branch.children)
            deg_freedom = num_attr_values - 1
            critical_value = scipy.stats.chi2.ppf(1 - self.alpha, deg_freedom)

            if likelihood_ratio < critical_value:

                # remove split and make branch a leaf
                if branch_parent is None:
                    # if branch has no parents, make root new leaf
                    self._root = self._new_leaf(branch.stats, None)
                    self.last_leaf_layer = [self._root]
                    self._n_active_leaves = 1
                else:
                    # initialize new leaf and replace the correct child of branch_parent
                    new_leaf = self._new_leaf(branch.stats, branch_parent)

                    new_children = branch_parent.children
                    for idx, child in enumerate(branch_parent.children):
                        if isinstance(child, DTBranch) and child.feature == branch.feature:
                            new_children[idx] = new_leaf
                    self._n_active_leaves -= 1

            else:
                # traverse tree
                for child in branch.children:
                    if isinstance(child, DTBranch):
                        self.check_split_validity(child, branch)

    def replace_last_layer(self, train_batch):
        """
        replace_last_layer compares the conditional mutual entropy of the current last hidden layer to the second-best
        attribute of the last hidden layer and conditionally replaces the layer
        :param train_batch: current training window
        """

        last_hidden_depth = self.last_leaf_layer[0].depth - 1

        if last_hidden_depth < 2:
            return

        map = {}

        for x,y in train_batch:
            path = list(self._root.walk(x, until_leaf=False))
            if path[-1] not in map.keys() and (path[-1].depth == last_hidden_depth):
                map[path[-1]] = self._new_leaf({}, None, [(path[-2], path[-2].branch_no(x))])
                map[path[-1]].depth = last_hidden_depth
                map[path[-1]].learn_one(x, y)
            elif path[-1].depth == last_hidden_depth:
                map[path[-1]].learn_one(x, y)
            if path[-2] not in map.keys() and (path[-1].depth == last_hidden_depth + 1):
                map[path[-2]] = self._new_leaf({}, None, [(path[-3], path[-3].branch_no(x))])
                map[path[-2]].depth = last_hidden_depth
                map[path[-2]].learn_one(x, y)
            elif path[-1].depth == last_hidden_depth + 1:
                map[path[-2]].learn_one(x, y)

        criterion = self._new_split_criterion()

        best_MI = np.sum([node.splitters[self.most_recent_best].best_evaluated_split_suggestion(criterion,
                                                                                                node.stats,
                                                                                                self.most_recent_best,
                                                                                                self.binary_split).merit for node in map.values()])

        second_best_MI = np.sum([node.splitters[self.most_recent_second].best_evaluated_split_suggestion(criterion,
                                                                                                node.stats,
                                                                                                self.most_recent_best,
                                                                                                self.binary_split).merit for node in map.values()])

        if best_MI < second_best_MI:
            # replace layer
            for old_node, new_node in map.items():
                parent, branch_no = new_node.parents[0]
                parent.children[branch_no] = new_node

            self.last_leaf_layer = map.values()

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

        * If the tree is empty, build an IFN
        * If concept drift is not detected, perform Update_Current_Network
        * If concept drift is detected, rebuild IFN
        """

        # Add to the batch
        self.batch.append((x, y))

        if len(self.batch) >= self.window + self.add_count:

            if self._root is None:
                # Build new IN model if not initialized
                self.build_IN(
                    train_batch=self.batch[: self.window], sample_weight=sample_weight
                )

            # Get our training and validation errors
            err = metrics.Accuracy()
            for x, y in self.batch[: self.window]:
                err.update(y, self.predict_one(x))
            train_err = err.get()

            err = metrics.Accuracy()
            for x, y in self.batch[self.window:]:
                err.update(y, self.predict_one(x))
            validation_err = err.get()

            # EQ. 9
            var_diff = (train_err * (1 - train_err)) / self.window + (
                    validation_err * (1 - validation_err)
            ) / self.add_count

            # EQ. 10
            max_diff = 2.326 * math.sqrt(var_diff)

            # detect concept drift
            if (validation_err - train_err) < max_diff:  # No concept drift

                # Check split validity on all branches in network
                if isinstance(self._root, DTBranch):
                    self.check_split_validity(self._root, None)

                # conditionally replace last layer and perform New_Split_Process
                self.update_IN(train_batch=self.batch[: self.window], sample_weight=sample_weight)

                # New Validation and Train windows
                self.add_count = int(
                    min(
                        self.add_count * (1 + (self.inc_add_count / 100.0)),
                        self.max_add_count,
                    )
                )
                self.window = min(self.window + self.add_count, self.max_window)
                self.batch = self.batch[-self.window:]

            else:  # Yes, concept drift

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
                    # # log2(0) handling
                    # if train_err == 0:
                    # math or np might be more performant
                    H_train_err = -(
                            scipy.special.xlogy(train_err, train_err)
                            + scipy.special.xlog1py(1 - train_err, -train_err)
                    ) / np.log(2)
                    #                 H_train_err= scipy.stats.entropy([train_err])

                    ys = dict.fromkeys(self.classes, 0)

                    for _, y in self.batch[: self.window]:
                        ys[y] += 1

                    probs = np.array(list(ys.values())) / sum(ys.values())

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

                #
                # Find the new validation window
                self.add_count = int(
                    max(
                        self.add_count * (1 - (self.red_add_count / 100.0)),
                        self.min_add_count,
                    )
                )

                # Update the batch data to drop the training data
                self.batch = self.batch[-self.window:]

                # build new IN
                self.build_IN(
                    train_batch=self.batch[: self.window], sample_weight=sample_weight
                )

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
