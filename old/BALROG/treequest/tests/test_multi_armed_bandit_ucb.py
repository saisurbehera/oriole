import random
from typing import Optional, Tuple

from treequest.algos.multi_armed_bandit_ucb import MultiArmedBanditUCBAlgo
from treequest.visualization import visualize_tree_graphviz


def test_multi_armed_bandit_ucb():
    """Test the MultiArmedBanditUCBAlgo implementation."""

    # Define a simple generate function that returns random scores
    def generate_fn_a(state: Optional[str]) -> Tuple[str, float]:
        # Action A tends to give higher scores on average
        return f"State from A: {random.random()}", 0.7 + 0.3 * random.random()

    def generate_fn_b(state: Optional[str]) -> Tuple[str, float]:
        # Action B tends to give lower scores on average
        return f"State from B: {random.random()}", 0.3 + 0.3 * random.random()

    # Create the algorithm and initial state
    algo = MultiArmedBanditUCBAlgo(exploration_weight=1.0)
    state = algo.init_tree()

    # Create a mapping of actions to generate functions
    generate_fns = {"A": generate_fn_a, "B": generate_fn_b}

    # Run several steps
    n_steps = 20
    for _ in range(n_steps):
        state = algo.step(state, generate_fns)

    # Check that we have the expected number of nodes
    state_score_pairs = algo.get_state_score_pairs(state)
    assert len(state_score_pairs) == n_steps

    # Check that action A was selected more often than action B (since it gives higher scores)
    a_count = len(state.scores_by_action["A"])
    b_count = len(state.scores_by_action["B"])

    # Print statistics for debugging
    print(
        f"Action A selected {a_count} times with avg score: {sum(state.scores_by_action['A']) / a_count if a_count else 0}"
    )
    print(
        f"Action B selected {b_count} times with avg score: {sum(state.scores_by_action['B']) / b_count if b_count else 0}"
    )

    # After enough steps, A should be selected more often than B
    # Note: This might not always be true for small n_steps due to exploration
    if n_steps > 15:
        assert a_count > b_count, (
            "Action A should be selected more often than B after sufficient exploration"
        )

    # After running the algorithm steps
    # Visualize the tree
    visualize_tree_graphviz(
        state.tree,
        save_path="tests/multiarm_ucb",
        title="Multi Arm Bandit UCB",
        format="png",
    )
