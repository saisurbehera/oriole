#!/usr/bin/env python3
"""Analyze the action space in NLE based on the code"""

# Based on the code in balrog/environments/nle/base.py:
# Lines 107-117 show how actions are constructed for non-minihack environments

print("Analysis of NLE Action Space based on BALROG code:")
print("=" * 60)

# From the code:
# 1. available_actions from USEFUL_ACTIONS (mapped through all_nle_action_map)
# 2. single_chars: a-z and A-Z
# 3. single_digits: 0-9  
# 4. double_digits: 00-99

# Calculate the extra actions added
single_chars_lowercase = 26  # a-z
single_chars_uppercase = 26  # A-Z
single_digits = 10           # 0-9
double_digits = 100          # 00-99

total_extra = single_chars_lowercase + single_chars_uppercase + single_digits + double_digits

print(f"\nExtra actions added to USEFUL_ACTIONS:")
print(f"- Single lowercase chars (a-z): {single_chars_lowercase}")
print(f"- Single uppercase chars (A-Z): {single_chars_uppercase}")
print(f"- Single digits (0-9): {single_digits}")
print(f"- Double digits (00-99): {double_digits}")
print(f"\nTotal extra actions: {total_extra}")

print(f"\nThe claim states: '52 letters + 100 numbers = 152 extra actions'")
print(f"This appears to be correct: {single_chars_lowercase + single_chars_uppercase} letters + {single_digits + double_digits} numbers = {total_extra} extra actions")

print("\n" + "=" * 60)
print("Why these are added:")
print("- Single chars (a-z, A-Z): Used for inventory items, directions, etc.")
print("- Single digits (0-9): Used for counts, menu selections")
print("- Double digits (00-99): Used for larger counts or indices")

print("\nThese allow the agent to interact with inventory items and provide numeric inputs,")
print("which are essential for playing NetHack but may not be part of the core USEFUL_ACTIONS.")