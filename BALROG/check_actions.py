#!/usr/bin/env python3
"""Check NetHack USEFUL_ACTIONS"""

try:
    from nle.nethack import USEFUL_ACTIONS
    from nle.language_wrapper.wrappers import nle_language_wrapper as language_wrapper
    
    print(f'Total USEFUL_ACTIONS: {len(USEFUL_ACTIONS)}')
    print('\nUSEFUL_ACTIONS list:')
    for i, act in enumerate(USEFUL_ACTIONS):
        print(f'{act.name:<20}', end='')
        if (i + 1) % 3 == 0:
            print()
    print('\n')
    
    # Check the language action map
    print(f"Total actions in NLELanguageWrapper.all_nle_action_map: {len(language_wrapper.NLELanguageWrapper.all_nle_action_map)}")
    
    # Show first 20 mapped actions
    print("\nFirst 20 action mappings:")
    for i, (action, action_strs) in enumerate(language_wrapper.NLELanguageWrapper.all_nle_action_map.items()):
        if i < 20:
            print(f"{action.name:<20} -> {action_strs}")
            
except ImportError as e:
    print(f"Error importing NLE: {e}")
    print("Make sure NLE is installed in the BALROG environment")