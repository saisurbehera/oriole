# Add these expert abstractions to DreamCoderAgentV2._initialize_primitives()

# Expert patterns from successful NetHack players
expert_abstractions = [
    ProgramAbstraction(
        name="expert_early_exploration",
        pattern=("search", "open", "north", "search", "east"),
        context_conditions=(("game_stage", "early"), ("location_type", "room")),
        success_rate=0.8
    ),
    ProgramAbstraction(
        name="expert_door_breaking",
        pattern=("kick", "kick", "open", "north"),
        context_conditions=(("door_present", "true"), ("door_locked", "true")),
        success_rate=0.75
    ),
    ProgramAbstraction(
        name="expert_stair_finding",
        pattern=("search", "search", "search", "east", "search"),
        context_conditions=(("explored_percent", "high"), ("no_stairs", "true")),
        success_rate=0.7
    ),
    ProgramAbstraction(
        name="expert_combat_kiting",
        pattern=("wait", "kick", "south", "wait"),
        context_conditions=(("enemy_adjacent", "true"), ("hp_percent", "low")),
        success_rate=0.65
    ),
    ProgramAbstraction(
        name="expert_diagonal_scout",
        pattern=("northeast", "search", "southwest", "search"),
        context_conditions=(("location_type", "room"), ("unexplored_corners", "true")),
        success_rate=0.7
    )
]

# Add to library
for abstr in expert_abstractions:
    self.library.append(abstr)
