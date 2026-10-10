#!/usr/bin/env python3
"""
Show how DreamCoder learning works in BALROG
"""

print("🧠 HOW DREAMCODER LEARNING WORKS IN BALROG")
print("=" * 50)

print("\n1️⃣ PATTERN RECOGNITION")
print("-" * 30)
print("During gameplay, the agent tracks:")
print("- Every action taken")
print("- The context when action was taken") 
print("- Whether it led to progress (score/depth increase)")

print("\nExample trace:")
print("  Context: {'stairs_visible': 'true', 'position': 'room'}")
print("  Actions: ['search', 'east', 'east', 'north', 'down']")
print("  Result: Depth increased! ✅")

print("\n2️⃣ ABSTRACTION CREATION")
print("-" * 30)
print("From successful episodes, it creates reusable 'programs':")
print("""
ProgramAbstraction(
    name="navigate_to_stairs_in_room",
    pattern=["east", "east", "north", "down"],
    context_conditions=["stairs_visible", "in_room"],
    success_rate=0.8
)
""")

print("\n3️⃣ PATTERN LIBRARY")
print("-" * 30)
print("The agent builds a library of successful patterns:")
print("- 'door_opening_sequence': ['kick', 'kick', 'open']")
print("- 'stairs_navigation': ['search', 'move_to_stairs', 'down']")
print("- 'combat_in_corridor': ['wait', 'kick', 'wait']")
print("- 'item_pickup': ['move_to_item', 'pickup']")

print("\n4️⃣ PATTERN MATCHING & REUSE")
print("-" * 30)
print("In new situations, the agent:")
print("1. Checks current context")
print("2. Finds matching patterns with high success rates")
print("3. Executes the learned sequence")
print("4. Falls back to LLM if no pattern matches")

print("\n5️⃣ LEARNING FEEDBACK LOOP")
print("-" * 30)
print("After each episode:")
print("- Successful patterns → Increase success rate")
print("- Failed patterns → Decrease success rate")
print("- New successful sequences → Add to library")
print("- Repetitive failures → Avoid that pattern")

print("\n📊 EXAMPLE LEARNING PROGRESSION:")
print("-" * 30)
print("Episode 1: Random exploration, finds stairs by luck (10% success)")
print("Episode 2: Remembers 'search near walls', finds stairs faster (30% success)")
print("Episode 3: Learns 'systematic room search', consistent stairs finding (60% success)")
print("Episode 4: Masters 'quick stairs navigation', goes deeper (80% success)")
print("Episode 5: Combines patterns for optimal play (90% success)")

print("\n🔍 IN OUR NETHACK TEST:")
print("-" * 30)
print("The agent SAW the stairs ('<' symbol) but didn't act on them because:")
print("1. The LLM was giving verbose responses instead of actions")
print("2. The agent got stuck in a 'search' loop") 
print("3. No successful patterns were learned yet")

print("\n💡 HOW TO IMPROVE:")
print("-" * 30)
print("1. Run more episodes to build pattern library")
print("2. Use better prompts for the small LLM")
print("3. Add hardcoded 'bootstrap' patterns to start with")
print("4. Increase exploration when stuck")

# Show actual patterns from the expert trajectories
print("\n🏆 EXPERT PATTERNS (from trajectory files):")
print("-" * 30)

expert_patterns = [
    {
        "name": "door_kick_sequence",
        "pattern": ["kick", "kick", "kick", "kick", "open"],
        "context": "When facing a locked door",
        "frequency": "65.4% of actions in trajectories"
    },
    {
        "name": "stairs_search", 
        "pattern": ["search", "search", "move", "search"],
        "context": "When looking for stairs",
        "frequency": "Common exploration pattern"
    },
    {
        "name": "quick_descent",
        "pattern": ["navigate_to_stairs", "down"],
        "context": "When stairs are visible", 
        "frequency": "Goal-oriented behavior"
    }
]

for p in expert_patterns:
    print(f"\n'{p['name']}':")
    print(f"  Pattern: {p['pattern']}")
    print(f"  Context: {p['context']}")
    print(f"  Frequency: {p['frequency']}")