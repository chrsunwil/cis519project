from river import base
from river import metrics
from .IOLIN_tree import IOLINTree
from .nodes.branch import DTBranch
from .nodes.IOLIN_nodes import LeafMajorityClass
from .nodes.leaf import IOLINLeaf
from .split_criterion import (
    IOLINInfoGainSplitCriterion
)
from .splitter import GaussianSplitter, Splitter

import math

class OLINTreeClassifier(IOLINTree, base.Classifier):
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
        max_window : int = 60,
        min_add_count : int = 4,
        max_add_count : int = 50,
        inc_add_count : int = 10,
        red_add_count : int = 10,
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
        
        self.cur_depth = 0
        
        #EQ. 7 HOW DO I IMPLEMENT THIS?
        self.window = 3 
        self.add_count = 4
        
        self.max_window = max_window
        
        self.min_add_count = min_add_count
        self.max_add_count = max_add_count
        
        self.inc_add_count = inc_add_count
        self.red_add_count = red_add_count
        
        self.batch = []
        self.used_split_features = set()

    @IOLINTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        self._split_criterion = IOLINInfoGainSplitCriterion

    @IOLINTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        self._leaf_prediction = LeafMajorityClass

    def _new_leaf(self, initial_stats=None, parent=None, parents=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1
           
        return LeafMajorityClass(initial_stats, depth, self.splitter, parents)

    def _new_split_criterion(self):
        split_criterion = IOLINInfoGainSplitCriterion()

        return split_criterion

    def _attempt_to_split(
        self, leaf: IOLINLeaf, **kwargs
    ):
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
        split_criterion = self._new_split_criterion()
        best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
        best_split_suggestions.sort()
        for best_suggestion in best_split_suggestions:
            if (best_suggestion.merit > 0 
                and (best_suggestion.feature not in self.used_split_features)): #NEGATIVE MERIT SHOULDN'T SPLIT?

#                 print(best_suggestion.feature)
                
                split_decision = best_suggestion
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
                        self._new_leaf(initial_stats, parent=leaf, parents=leaf.parents)
                        for initial_stats in split_decision.children_stats
                    )

                    #Don't think this matters because I build whole network at one time
    #                 self.cur_depth += 1 #THIS IS ASSUMING THAT ONLY LAST LAYERS CAN SPLIT

        #                     print(self.cur_depth)

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    for new_leaf in leaves:
                        new_leaf.parents = [(new_split, 0)]

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)

                    parents = leaf.parents

                    if parents is None:
                        self._root = new_split
                    else:
                        for parent, parent_branch in parents:
                            parent.children[parent_branch] = new_split

                    self.used_split_features.add(best_suggestion.feature)

                    return leaves
                return []
        return []

    
    def build_IN(self, train_batch,*, sample_weight=1.0):
        
        self.classes = set()
        
        self._root = self._new_leaf()
        self.cur_depth = 0
        self._n_active_leaves = 1 
        
        self.used_split_features = set()
        
        nodes_to_attempt_split = [self._root] #Reset so that we can learn the new layers
        
        
        while len(nodes_to_attempt_split) > 0: #While there is a node in the last layer (or prev)
            
            cur_node = nodes_to_attempt_split.pop(0)
            
            self._train_weight_seen_by_model = 0
            
            #Teach the current node all of the data
            ##CREATE A BATCH UPDATE?
            for x,y, in train_batch:
                self.classes.add(y)
                self._train_weight_seen_by_model += sample_weight
                cur_node.learn_one(x,y, sample_weight=sample_weight, tree=self)
                
            #attempt to split the current node. Get back the new children
            new_leaves = self._attempt_to_split(cur_node)
#             print(self.used_split_features)
            nodes_to_attempt_split += new_leaves
                
#         print("Built an IN")
                

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
        
        #Add to the batch
        self.batch.append((x,y))
        
        if len(self.batch) >= self.window + self.add_count:
            
            #Build new IN model
            self.build_IN(train_batch = self.batch[:self.window],sample_weight=sample_weight)
            
            
            err = metrics.Accuracy()
            for x,y in self.batch[:self.window]:
                err.update(y, self.predict_one(x))
            train_err = err.get()
            err = metrics.Accuracy()
            for x,y in self.batch[self.window:]:
                err.update(y, self.predict_one(x))
            validation_err = err.get()
            
            #IS THIS THE SAME AS EXPECTED VALUE OF ERROR? EQ. 9
            var_diff = ((train_err * (1-train_err))/self.window 
                        + (validation_err * (1-validation_err))/self.add_count)
            
            max_diff = 2.326 * math.sqrt(var_diff)
            
            #detect concept drift
            if (validation_err - train_err) < max_diff:
                self.add_count = int(min(self.add_count * (1 + (self.inc_add_count/100.0)), 
                                      self.max_add_count))
                self.window = min(self.window+self.add_count,self.max_window)
                self.batch = self.batch[self.add_count:]
            else:
                #recalculate size of window
                #USE EQ. 8 -- I have literally no clue how to do this
                

                self.add_count = int(max(self.add_count * (1 - (self.red_add_count/100.0)), 
                                      self.min_add_count))      
        
 
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
