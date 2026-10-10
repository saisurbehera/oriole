import random
from typing import Optional, Tuple

from treequest.algos.ab_mcts_a.algo import ABMCTSA
from treequest.algos.ab_mcts_a.prob_state import PriorConfig


def test_basic_initialization():
    """Test basic initialization of ABMCTSA."""
    # Test with default parameters
    algo = ABMCTSA()
    assert algo.prior_config is not None
    assert algo.reward_average_priors is None
    assert (
        algo.model_selection_strategy == "multiarm_bandit_thompson"
    )  # Default strategy

    # Test with custom parameters
    prior_config = PriorConfig(dist_type="beta")
    algo = ABMCTSA(prior_config=prior_config, reward_average_priors=0.7)
    assert algo.prior_config == prior_config
    assert algo.reward_average_priors == 0.7
    assert (
        algo.model_selection_strategy == "multiarm_bandit_thompson"
    )  # Default strategy

    # Test with custom model selection strategy
    algo = ABMCTSA(model_selection_strategy="multiarm_bandit_ucb")
    assert algo.model_selection_strategy == "multiarm_bandit_ucb"

    # Test initial state
    state = algo.init_tree()
    assert state.tree is not None
    assert len(state.thompson_states) == 0
    assert state.tree.root is not None
    assert state.tree.root.is_root()


def test_single_step():
    """Test a single step of the ABMCTSA."""
    # Set a fixed seed for reproducibility
    random.seed(42)

    # Create a simple generate function
    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        return f"Generated from test: {random.random()}", random.random()

    # Initialize algorithm and state
    algo = ABMCTSA(model_selection_strategy="stack")
    state = algo.init_tree()

    # Create a mapping with a single action
    generate_fns = {"test_action": generate_fn}

    # Run a single step
    new_state = algo.step(state, generate_fns)

    # Check that a new node was created
    assert len(new_state.tree.get_state_score_pairs()) == 1
    assert len(new_state.thompson_states) == 1

    # Check that root node has one child
    assert len(new_state.tree.root.children) == 1

    # The first child should be expanded from the root
    first_child = new_state.tree.root.children[0]
    assert first_child.parent == new_state.tree.root
    assert isinstance(first_child.state, str)
    assert "Generated from test" in first_child.state


def test_thompson_sampling_algo_with_mock():
    """Test the ABMCTSA using a mocked version to always select action_a."""

    # Create a custom version of ABMCTSA
    class MockABMCTSA(ABMCTSA):
        def _select_child(
            self,
            state,
            node,
            generate_fn,
        ):
            """
            Simplified selection that always creates a new node
            """
            # Get or create thompson state for this node
            thompson_state = state.thompson_states.get_or_create(
                node,
                list(generate_fn.keys()),
            )

            # Ask for next node or action using Thompson Sampling
            selection = thompson_state.select_next(state.all_rewards_store)

            # If string returned, we need to generate a new node with that action
            if isinstance(selection, str):
                # We always select action_a
                selection = "action_a"
                new_node = self._generate_new_child(state, node, generate_fn, selection)
                return new_node, selection
            else:
                # Otherwise, we return the existing child with that index
                if selection >= len(node.children):
                    raise RuntimeError(
                        f"Something went wrong in ABMCTSA algorithm: selected index {selection} is out of bounds."
                    )
                return node.children[selection], None

        def _expand_node(self, state, node, generate_fn):
            """
            Override to always use action_a for expansion.
            """
            # Create thompson state for this node if it doesn't exist
            state.thompson_states.get_or_create(
                node,
                list(generate_fn.keys()),
                prior_config=self.prior_config,
                reward_average_priors=self.reward_average_priors,
                model_selection_strategy=self.model_selection_strategy,
            )

            # Always use action_a for generating child
            action = "action_a"

            new_node = self._generate_new_child(state, node, generate_fn, action)
            return new_node, action

    # Create the mock algorithm
    mock_algo = MockABMCTSA()
    state = mock_algo.init_tree()

    # Create generate functions
    def generate_fn_a(state):
        return f"State A: {random.random()}", 0.8

    def generate_fn_b(state):
        return f"State B: {random.random()}", 0.4

    generate_fns = {"action_a": generate_fn_a, "action_b": generate_fn_b}

    # Run multiple steps
    n_steps = 3
    for _ in range(n_steps):
        state = mock_algo.step(state, generate_fns)

    # Check we have the expected number of nodes
    assert len(state.tree.get_state_score_pairs()) == n_steps

    # Check that all nodes were created with action_a
    for node in state.tree.get_nodes():
        if not node.is_root() and node.state is not None:
            assert "State A:" in str(node.state)


def test_model_selection_strategies():
    """Test different model selection strategies."""
    # Test stack strategy (default)
    algo_stack = ABMCTSA(model_selection_strategy="stack")
    assert algo_stack.model_selection_strategy == "stack"
    state_stack = algo_stack.init_tree()
    assert state_stack.thompson_states.default_model_selection_strategy == "stack"

    # Test multiarm_bandit_thompson strategy
    algo_thompson = ABMCTSA(model_selection_strategy="multiarm_bandit_thompson")
    assert algo_thompson.model_selection_strategy == "multiarm_bandit_thompson"
    state_thompson = algo_thompson.init_tree()
    assert (
        state_thompson.thompson_states.default_model_selection_strategy
        == "multiarm_bandit_thompson"
    )

    # Test multiarm_bandit_ucb strategy
    algo_ucb = ABMCTSA(model_selection_strategy="multiarm_bandit_ucb")
    assert algo_ucb.model_selection_strategy == "multiarm_bandit_ucb"
    state_ucb = algo_ucb.init_tree()
    assert (
        state_ucb.thompson_states.default_model_selection_strategy
        == "multiarm_bandit_ucb"
    )

    # Test invalid strategy
    try:
        ABMCTSA(model_selection_strategy="invalid_strategy")
        assert False, "Should have raised ValueError for invalid strategy"
    except ValueError:
        pass  # Expected behavior
