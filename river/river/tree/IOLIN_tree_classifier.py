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
    """Hoeffding Tree or Very Fast Decision Tree classifier.

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

    Notes
    -----
    A Hoeffding Tree [^1] is an incremental, anytime decision tree induction algorithm that is
    capable of learning from massive data streams, assuming that the distribution generating
    examples does not change over time. Hoeffding trees exploit the fact that a small sample can
    often be enough to choose an optimal splitting attribute. This idea is supported mathematically
    by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of
    an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision
    tree learners is that it has sound guarantees of performance. Using the Hoeffding bound one
    can show that its output is asymptotically nearly identical to that of a non-incremental
    learner using infinitely many examples. Implementation based on MOA [^2].

    References
    ----------

    [^1]: G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    [^2]: Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------
    >>> from river import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.Agrawal(classification_function=0, seed=42)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.HoeffdingTreeClassifier(
    ...     grace_period=100,
    ...     split_confidence=1e-5,
    ...     nominal_attributes=['elevel', 'car', 'zipcode']
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 83.78%
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

    def _attempt_to_split(self, leaves, update=False, **kwargs):
        """Attempt to split a leaf.
        Parameters
        ----------
        leaf
            The leaf to evaluate.
        parent
            The leaf's parent.
        parent_branch
            Parent leaf's branch index.
        kwargs
            Other parameters passed to the new branch.
        """
        print("attempt_to_split()")
        leaves_split_suggestions = []  ## added from OLIN_tree_classifier

        for leaf in leaves:
            split_criterion = self._new_split_criterion()
            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            # if len(leaves) == 3 and len(leaf.stats) > 1:
            #     print(leaf)
            leaves_split_suggestions.append(best_split_suggestions)

        attributes = []

        for split_suggestion in leaves_split_suggestions[0]:
            attributes.append(split_suggestion.feature)

        temp_attributes = []
        for attribute in attributes:
            if attribute not in self.used_split_features:
                temp_attributes.append(attribute)

        attributes = temp_attributes

        attributes_merits = [0] * len(attributes)

        for counter in range(len(attributes)):
            for leaf_split_suggestion in leaves_split_suggestions:
                for attribute_split in leaf_split_suggestion:
                    if attribute_split.feature == attributes[counter]:
                        # TODO remove any negative merit
                        attributes_merits[counter] += max(attribute_split.merit, 0)

        sorted_attributes_merits = np.argsort(attributes_merits)[::-1]

        best_attr_idx = sorted_attributes_merits[0]

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

                        # if split_decision.feature is None:
                        #     # Pre-pruning - null wins
                        #     leaves[counter].deactivate()
                        #     self._n_inactive_leaves += 1
                        #     self._n_active_leaves -= 1
                        # else:
                        #     branch = self._branch_selector(
                        #         split_decision.numerical_feature,
                        #         split_decision.multiway_split,
                        #     )
                        #
                        #     new_leaves = tuple(
                        #         self._new_leaf(
                        #             {},
                        #             parent=leaves[counter],
                        #             parents=leaves[counter].parents,
                        #         )
                        #         for initial_stats in split_decision.children_stats
                        #     )
                        #     self.alternate_last_layer = new_leaves

        return_leaves = []
        # print(len(leaves_split_suggestions[2]))
        # print(attributes)
        # print(attributes_merits)

        for counter in range(len(leaves_split_suggestions)):
            for suggestion in leaves_split_suggestions[counter]:

                if suggestion.merit > 0 and (
                        suggestion.feature == attributes[best_attr_idx]
                ):  # NEGATIVE MERIT SHOULDN'T SPLIT?

                    #                 print(best_suggestion.feature)
                    best_suggestion = suggestion

                    print("tree weight", self._train_weight_seen_by_model)
                    print("MI in att_to_spl", best_suggestion.merit)
                    print("children stats in att_to_spl", best_suggestion.children_stats)
                    print("thresholds in att_to_slp:", best_suggestion.split_info)

                    # if this is update current network, compare second suggestion to best suggestion
                    # if (update is True) and (second_suggestion.merit > best_suggestion.merit):
                    #     # remove previous best feature from set of used features
                    #     self.used_split_features.remove(best_suggestion)
                    #     best_suggestion = second_suggestion
                    #     # TODO: remove leaves?

                    # print("best suggestion:", best_suggestion.feature)
                    # print("second suggestion:", second_suggestion.feature)

                    split_decision = best_suggestion

                    self.most_recent_best = split_decision.feature

                    # print(split_decision)
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

                        new_leaves = tuple(
                            self._new_leaf(
                                {},
                                parent=leaves[counter],
                                parents=leaves[counter].parents,
                            )
                            for initial_stats in split_decision.children_stats
                        )

                        # Don't think this matters because I build whole network at one time
                        #                 self.cur_depth += 1 #THIS IS ASSUMING THAT ONLY LAST LAYERS CAN SPLIT

                        #                     print(self.cur_depth)

                        new_split = split_decision.assemble(
                            branch,
                            leaves[counter].stats,
                            leaves[counter].depth,
                            *new_leaves,
                            **kwargs
                        )

                        #                         print(new_leaves[1].__repr__())

                        #                         print("Add ", len(new_leaves), "new leaves for ", best_suggestion.feature)

                        #                         print(new_split.repr_split)

                        inc = 0
                        for new_leaf in new_leaves:
                            new_leaf.parents = [(new_split, inc)]
                            inc += 1

                        self._n_active_leaves -= 1
                        self._n_active_leaves += len(new_leaves)

                        parents = leaves[counter].parents  # leaves[counter] -- former leaf
                        # print("leaves[counter]=",leaves[counter])

                        #                         if parents is not None:
                        #                             print(leaves[counter].parents[0][0].repr_split)
                        #                         print(split_decision.feature, split_decision.children_stats, parents[0][0].repr_split if parents is not None else None)
                        if parents is None:
                            self._root = new_split
                        else:
                            for parent, parent_branch in parents:
                                parent.children[parent_branch] = new_split

                        self.used_split_features.add(best_suggestion.feature)

                        if self.first_attr is None:
                            self.first_attr = best_suggestion.feature
                        return_leaves += new_leaves

                        # print(len(return_leaves))
        # print(f'for real: {len(return_leaves)}')
        return return_leaves

    def build_IN(self, train_batch, *, sample_weight=1.0): #split_last_layer=False

        # if split_last_layer is False:
        self.classes = set()

        self._root = self._new_leaf()
        self.cur_depth = 0
        self._n_active_leaves = 1

        self.used_split_features = set()

        self.first_attr = None
        self.first_attr_vals = set()

        nodes_to_attempt_split = [self._root]  # Reset so that we can learn the new layers
        self.last_leaf_layer = [self._root]

        while (
                len(nodes_to_attempt_split) > 0
        ):  # While there is a node in the last layer which got created

            self._train_weight_seen_by_model = 0

            # Teach the current node all of the data

            for cur_node in nodes_to_attempt_split:
                # print("cur_node", cur_node)
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
            # print(len(nodes_to_attempt_split))

            new_leaves = self._attempt_to_split(nodes_to_attempt_split)

            print("Children stats after att_to_spl")
            if isinstance(self._root, DTBranch):
                for child in self._root.children:
                    print(child.stats)
            # print("new_leaves1: ", new_leaves)
            # print("used_split_features: ", self.used_split_features)

            # self.last_hidden_layer = nodes_to_attempt_split
            nodes_to_attempt_split = new_leaves
            # print(nodes_to_attempt_split)

            ### tracking last hidden layer
            # print("nodestosplit:",len(nodes_to_attempt_split))

            if len(new_leaves) > 0:
            # if nodes_to_attempt_split is not None:
            #     last_hidden_layer = []
            #     for node in nodes_to_attempt_split:
            #         if node.parents not in last_hidden_layer:
            #             last_hidden_layer.append(node.parents[0][0])
            #             # print("PARENTS:")
            #             # print(node.parents[0][0])
            #     self.last_hidden_layer = last_hidden_layer  # TODO: this has duplicates of same branch from each child

                self.last_leaf_layer = new_leaves
                # print("new_leaves2: ", new_leaves)
            # print("last layer in build_IN:", self.last_hidden_layer)

            # print("last layer attsplt:", self.last_hidden_layer)

            # print(self.last_hidden_layer)
            #
        print("Added new Layer")
            #
            # print("Built an IN")
        # print("Children stats after buildIN")
        # if isinstance(self._root, DTBranch):
        #     for child in self._root.children:
        #         print(child.stats)

    def update_IN(self, train_batch, *, sample_weight=1.0):

        if self.last_leaf_layer[0] == self._root:
            print("calling buildIN on updateIN")
            self.build_IN(train_batch, sample_weight=sample_weight)
            print("out")
            return

        self.replace_last_layer(train_batch)

        nodes_to_attempt_split = self.last_leaf_layer

        print(nodes_to_attempt_split[0])

        # copy nodes, get rid of data, update references
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
            print("in", len(self.last_leaf_layer))

            # print("nodes to split updateIN:", nodes_to_attempt_split)

            # from build_IN
            self._train_weight_seen_by_model = 0

            # Teach the current node all of the data

            for cur_node in nodes_to_attempt_split:

                # print("cur_node: ", cur_node)
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

            if len(new_leaves) > 0:
                self.last_leaf_layer = new_leaves

            if len(self.last_leaf_layer) > 0:
                print("Added a new layer")
            else:
                print("No splits?")

    def check_split_validity(self, branch, branch_parent):  # branch
        print("check split val")

        criterion = self._new_split_criterion()
        # likelihood ratio test
        children_stats = []
        for child in branch.children:
            children_stats.append(child.stats)

        # weight of current node
        # branch_weight = sum(branch.stats.values())  # does this make sense?
        # tree_weight = sum(self._root.stats.values())
        # print(tree_weight, self._train_weight_seen_by_model)

        mutual_info = criterion.merit_of_split(branch.stats, children_stats,
                                               node_weight=branch.total_weight,
                                               tree_weight=self._train_weight_seen_by_model)
                                               # tree_weight=tree_weight)
        # p_node = branch_weight / self._train_weight_seen_by_model
        p_node = branch.total_weight / self._train_weight_seen_by_model
        mutual_info *= p_node
        if mutual_info > 0:
            likelihood_ratio = (
                    2 * math.log(2) * int(branch.total_weight) * mutual_info
            )
            num_attr_values = len(branch.children)  # len(branch.children
            deg_freedom = num_attr_values - 1
            critical_value = scipy.stats.chi2.ppf(1 - self.alpha, deg_freedom)

            if likelihood_ratio < critical_value:  # TODO: am i doing likelihood statistic right?
                print("MI:", mutual_info)
                print("branch stats:", branch.stats)
                print("children stats:", children_stats)
                print("critical val:", critical_value)
                print("branch weight:", branch.total_weight)
                print("tree weight:", self._train_weight_seen_by_model)

                # turn branch into leaf
                if branch_parent is None:
                    # if branch has no parents, make root new leaf
                    self._root = self._new_leaf(branch.stats, None)
                    self.last_leaf_layer = [self._root]
                    print("unsplit branch (root)")
                    self._n_active_leaves = 1
                    # print("feature:", branch.feature)
                    # print(branch.stats)
                    # print(mutual_info)
                    # print(p_node)
                    # print(likelihood_ratio)
                    # print(critical_value)
                else:
                    # initialize new leaf and replace the correct child of branch_parent
                    new_leaf = self._new_leaf(branch.stats, branch_parent)

                    new_children = branch_parent.children
                    for idx, child in enumerate(branch_parent.children):
                        if isinstance(child, DTBranch) and child.feature == branch.feature:
                            new_children[idx] = new_leaf
                    print("unsplit branch")

                    # TODO: update active/inactive leaves?
                    self._n_active_leaves -= 1

            else:
                for child in branch.children:  # root.iter_branches()
                    if isinstance(child, DTBranch):
                        self.check_split_validity(child, branch)

    def replace_last_layer(self, train_batch):

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

            if self._root is None:  # or isinstance(self._root, IOLINLeafMajorityClass):
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
                print("no drift")

                print(self._root)
                if isinstance(self._root, DTBranch):
                    print("Root is branch")
                    # print(self._root.stats)
                    self.check_split_validity(self._root, None)

                # TODO: compare current last layer to second best attribute
                # self.last_hidden_layer[0].stats

                # print("Last layer exists: ", self.last_leaf_layer)
                # for node in self.last_hidden_layer:
                #     print(node)

                print("calling update_IN()")
                # print(self._root.stats, self.last_leaf_layer.stats)
                self.update_IN(train_batch=self.batch[: self.window], sample_weight=sample_weight)  # TODO: Sample weight or cur weight

                # print("calling attempt to split on")#, self.last_hidden_layer[0].feature)
                # self._attempt_to_split([self.last_hidden_layer[0]])  # does a class variable work here?

                # TODO: What to do with windows
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
                print("drift detected")
                # recalculate size of window
                # EQ. 8
                # print(train_err)
                # print(validation_err)
                # print(max_diff)
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
                    #                 print(probs)
                    H_A_i = scipy.stats.entropy(probs)

                    numerator = scipy.stats.chi2.ppf(
                        1 - self.alpha, (NT_ip - 1) * (len(self.classes) - 1)
                    )
                    # print(self.alpha, self.summary, self.classes)
                    # print(numerator)

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
                    #                 print(H_A_i)
                    #                 print(H_train_err)
                    #                 print(train_err*np.log2(len(self.classes)-1))
                    # #                 print((NT_ip-1)*(len(self.classes)-1))
                    #                 print(numerator)
                    #                 print(denomenator)
                    # print(denomenator)
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
                # self.build_IN(
                #     train_batch=self.batch[: self.window], sample_weight=sample_weight, split_last_layer=False
                # )

        # print(self.window)

        # return self (down below)
        """
        #### Existing code from river #####
        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None

        node = None
        if isinstance(self._root, DTBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, IOLINLeaf):
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = (
                            p_node.branch_no(x)
                            if isinstance(p_node, DTBranch)
                            else None
                        )
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
        else:
            while True:
                # Split node encountered a previously unseen categorical value (in a multi-way
                #  test), so there is no branch to sort the instance to
                if node.max_branches() == -1 and node.feature in x:
                    # Create a new branch to the new categorical value
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    self._n_active_leaves += 1
                    node = leaf
                # The split feature is missing in the instance. Hence, we pass the new example
                # to the most traversed path in the current subtree
                else:
                    _, node = node.most_common_path()
                    # And we keep trying to reach a leaf
                    if isinstance(node, DTBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, IOLINLeaf):
                    break
            # Learn from the sample
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()
        """

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
