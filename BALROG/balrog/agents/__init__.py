from balrog.client import create_llm_client

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .custom import CustomAgent
from .dreamcoder import DreamCoderAgent
from .dreamcoder_v2 import DreamCoderAgentV2
from .hierarchical_dreamcoder_v2 import HierarchicalDreamCoderAgentV2
from .llm_prompted_agent import LLMPromptedAgent, PromptStrategy
from .bootstrapped_agent import BootstrappedAgent
from .micro_agent import MicroAgent
from .simple_heuristic_agent import SimpleHeuristicAgent
from .advanced_heuristic_agent import AdvancedHeuristicAgent
from .hierarchical_rl_agent import HierarchicalRLAgent
from .expert_guided_rl_agent import ExpertGuidedRLAgent
from .planning_first_agent import PlanningFirstAgent
from .simple_mcts_agent import SimpleMCTSAgent
from .feedback_mcts_agent import FeedbackMCTSAgent
from .feedback_mcts_agent_v2 import FeedbackMCTSAgentV2
from .stairs_focused_agent import StairsFocusedAgent
from .feedback_mcts_fixed import FeedbackMCTSFixed
from .feedback_mcts_working import FeedbackMCTSWorking
from .mcts_final import MCTSFinal
from .llm_mcts_agent import LLMMCTSAgent
from .real_llm_mcts import RealLLMMCTSAgent
# Temporarily disable AB-MCTS due to TreeQuest issues
# try:
#     from .ab_mcts_agent import ABMCTSAgent
# except ImportError:
ABMCTSAgent = None
try:
    from .mcts_planning_agent import MCTSPlanningAgent
except ImportError:
    MCTSPlanningAgent = None
from .dummy import DummyAgent
from .few_shot import FewShotAgent
from .naive import NaiveAgent
from .robust_naive import RobustNaiveAgent
from .robust_cot import RobustCoTAgent


class AgentFactory:
    """Factory class for creating agents based on configuration.

    The `AgentFactory` class is responsible for initializing the appropriate agent type
    based on the provided configuration, which includes setting up the LLM client and
    prompt builder.
    """

    def __init__(self, config):
        """Initialize the AgentFactory with configuration settings.

        Args:
            config (omegaconf.DictConfig): Configuration object containing settings for the agent and client.
        """
        self.config = config

    def create_agent(self):
        """Create an agent instance based on the agent type specified in the configuration.

        The function uses the `config.agent.type` attribute to determine which agent to create.
        It supports several agent types, including Naive, Chain-of-Thought, Self-Refine, Dummy,
        and Custom agents.

        Returns:
            Agent: An instance of the selected agent type, configured with the client and prompt builder.

        Raises:
            ValueError: If an unknown agent type is specified in the configuration.
        """
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        if self.config.agent.type == "naive":
            return NaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "cot":
            return ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "dummy":
            return DummyAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "custom":
            return CustomAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "few_shot":
            return FewShotAgent(client_factory, prompt_builder, self.config.agent.max_icl_history)
        elif self.config.agent.type == "robust_naive":
            return RobustNaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "robust_cot":
            return RobustCoTAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "dreamcoder":
            return DreamCoderAgentV2(  # Use improved version
                client_factory, 
                prompt_builder,
                memory_size=self.config.agent.get("memory_size", 1000),
                repetition_penalty_window=self.config.agent.get("repetition_penalty_window", 20),
                library_save_path=self.config.agent.get("library_save_path", "dreamcoder_library.json")
            )
        elif self.config.agent.type == "hierarchical":
            return HierarchicalDreamCoderAgentV2(client_factory, prompt_builder)
        elif self.config.agent.type == "llm_prompted":
            # Get prompt strategy from config
            strategy_name = self.config.agent.get("prompt_strategy", "multiple_choice")
            strategy = PromptStrategy[strategy_name.upper()]
            return LLMPromptedAgent(client_factory, prompt_builder, prompt_strategy=strategy)
        elif self.config.agent.type == "bootstrapped":
            return BootstrappedAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "micro":
            return MicroAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "simple_heuristic":
            return SimpleHeuristicAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "advanced_heuristic":
            return AdvancedHeuristicAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "hierarchical_rl":
            return HierarchicalRLAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "expert_guided_rl":
            return ExpertGuidedRLAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "planning_first":
            return PlanningFirstAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "mcts_planning":
            if MCTSPlanningAgent:
                return MCTSPlanningAgent(client_factory, prompt_builder)
            else:
                raise ValueError("MCTSPlanningAgent not available - TreeQuest dependencies missing")
        elif self.config.agent.type == "simple_mcts":
            return SimpleMCTSAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "feedback_mcts":
            return FeedbackMCTSAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "feedback_mcts_v2":
            return FeedbackMCTSAgentV2(client_factory, prompt_builder)
        elif self.config.agent.type == "feedback_mcts_fixed":
            return FeedbackMCTSFixed(client_factory, prompt_builder)
        elif self.config.agent.type == "mcts_working":
            return FeedbackMCTSWorking(client_factory, prompt_builder)
        elif self.config.agent.type == "mcts_final":
            return MCTSFinal(client_factory, prompt_builder)
        elif self.config.agent.type == "llm_mcts":
            return LLMMCTSAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "real_llm_mcts":
            return RealLLMMCTSAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "ab_mcts":
            if ABMCTSAgent is None:
                raise ImportError("ABMCTSAgent not available. Please install treequest dependencies.")
            return ABMCTSAgent(client_factory, prompt_builder)

        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")
