import copy
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Generic, List, Tuple, TypeVar

from treequest.algos.base import Algorithm
from treequest.algos.tree import Node, Tree
from treequest.types import GenerateFnType, StateScoreType

# Type variable for state
StateT = TypeVar("StateT")


@dataclass(order=True)
class BFSHeapItem(Generic[StateT]):
    """
    Heap item for Best First Search algorithm.

    Implements ordering to prioritize:
    1. Higher scores (reversed for min heap)
    2. Shallower nodes when scores are equal
    """

    # Use sort_index for comparison but keep it hidden from __init__
    sort_index: Tuple[float, int] = dataclasses.field(init=False, repr=False)
    node: Node[StateT] = dataclasses.field(compare=False)
    score: float = dataclasses.field(compare=False)

    def __post_init__(self):
        # Negative score for min heap (higher scores prioritized)
        # Node depth as secondary sort key (shallower nodes preferred)
        self.sort_index = (-self.score, self.node.depth)


@dataclass
class BFSState(Generic[StateT]):
    """State for Best First Search algorithm."""

    tree: Tree[StateT]
    next_nodes: List[Tuple[Node[StateT], str]] = dataclasses.field(default_factory=list)
    leaves: List[BFSHeapItem[StateT]] = dataclasses.field(default_factory=list)


class BestFirstSearchAlgo(Algorithm[StateT, BFSState[StateT]]):
    """
    Best First Search algorithm implementation.

    This algorithm:
    1. Prioritizes node expansion based on scores (higher scores first)
    2. When scores are equal, prefers shallower nodes
    3. For each selected node, generates `num_samples` samples for each available action,
       for a total of `num_samples` * number of actions new nodes.

    The expansion process:
    - Select the highest-scoring node from the priority queue
    - For that node, generate num_samples samples for each available action
    - Add all new nodes to the priority queue
    - Repeat
    """

    def __init__(self, *, num_samples: int = 2):
        """
        Initialize the Best First Search algorithm.

        Args:
            num_samples: Number of samples to generate for each action.
                        For each selected node, a total of (num_samples * number of actions)
                        new nodes will be generated.
        """
        self.num_samples = num_samples

    def step(
        self,
        state: BFSState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> BFSState:
        """
        Generate one additional node and add that to a given state.

        For each selected node, this algorithm will generate:
        - num_samples samples for each available action
        - Total of (num_samples * number of actions) new nodes

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        # If no nodes are queued for expansion, select a node to expand
        if not state.next_nodes:
            if not state.leaves:
                # If no leaves exist, use the root node
                parent = state.tree.root
            else:
                # Otherwise, pop the highest priority node from the heap
                parent = heappop(state.leaves).node

            # Queue up nodes for expansion
            # For each selected node, we generate num_samples * len(actions) new nodes
            state.next_nodes = [
                (parent, action)
                for _ in range(self.num_samples)
                for action in generate_fn
            ]

        # Get the next node to expand
        node, action = state.next_nodes.pop(0)

        # Generate a new state and add it to the tree
        new_state, new_score = generate_fn[action](node.state)
        new_node = state.tree.add_node((new_state, new_score), node)

        # Add the new node to the priority queue
        heappush(state.leaves, BFSHeapItem(node=new_node, score=new_score))

        return state

    def init_tree(self) -> BFSState:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return BFSState(tree)

    def get_state_score_pairs(self, state: BFSState) -> List[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree, excluding the root node.

        The root node is excluded because it typically represents an initial
        empty state with no score.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs for all non-root nodes
        """
        return state.tree.get_state_score_pairs()
