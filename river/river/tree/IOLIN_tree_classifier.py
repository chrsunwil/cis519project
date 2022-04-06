from river import base

from .IOLIN_tree import IOLINTree
from .nodes.branch import DTBranch
from .nodes.IOLIN_nodes import LeafMajorityClass
from .nodes.leaf import IOLINLeaf
from .split_criterion import (
    IOLINInfoGainSplitCriterion
)
from .splitter import GaussianSplitter, Splitter


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
        self.leaf_prediction = LeafMajorityClass
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        if splitter is None:
            self.splitter = GaussianSplitter()
        else:
            if not splitter.is_target_class:
                raise ValueError(
                    "The chosen splitter cannot be used in classification tasks."
                )
            self.splitter = splitter

        # To keep track of the observed classes
        self.classes: set = set()

    @IOLINTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        self._split_criterion = IOLINInfoGainSplitCriterion

    @IOLINTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        self._leaf_prediction = LeafMajorityClass

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1
           
        return LeafMajorityClass(initial_stats, depth, self.splitter)

    def _new_split_criterion(self):
        split_criterion = IOLINInfoGainSplitCriterion()

        return split_criterion

    def _attempt_to_split(
        self, leaf: IOLINLeaf, parent: DTBranch, parent_branch: int, **kwargs
    ):
        """Attempt to split a leaf.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node (branch node).
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attributes. Depends on the tree's configuration.

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
        if not leaf.observed_class_distribution_is_pure():  # noqa
            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.split_confidence,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit
                    > hoeffding_bound
                    or hoeffding_bound < self.tie_threshold
                ):
                    should_split = True
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        if (
                            suggestion.feature
                            and best_suggestion.merit - suggestion.merit
                            > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.feature is None:
                    # Pre-pruning - null wins
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                # Manage memory
                self._enforce_size_limit()

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
