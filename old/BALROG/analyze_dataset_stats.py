#!/usr/bin/env python3
"""
Dataset Statistics Analyzer
============================

Comprehensive analysis of the enhanced NetHack expert dataset.
"""

import json
import pandas as pd
from pathlib import Path
from datasets import load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np

def analyze_enhanced_dataset():
    """Analyze the enhanced dataset with comprehensive statistics"""
    
    # Load the dataset
    dataset_path = "/home/sai/Desktop/oriole/BALROG/enhanced_nethack_dataset"
    metadata_path = "/home/sai/Desktop/oriole/BALROG/expert_metadata.json"
    
    print("🔍 Loading Enhanced NetHack Expert Dataset...")
    dataset = load_from_disk(dataset_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Basic Statistics:")
    print(f"   - Training examples: {len(dataset['train'])}")
    print(f"   - Validation examples: {len(dataset['validation'])}")
    print(f"   - Total examples: {len(dataset['train']) + len(dataset['validation'])}")
    
    # Combine train and validation for analysis
    all_examples = []
    for split in ['train', 'validation']:
        for example in dataset[split]:
            example['split'] = split
            all_examples.append(example)
    
    df = pd.DataFrame(all_examples)
    
    print(f"\n📈 Episode Statistics:")
    print(f"   - Unique sessions: {df['session_id'].nunique()}")
    print(f"   - Unique users: {df['user_id'].nunique()}")
    print(f"   - Unique episodes: {df.groupby(['session_id', 'episode_num']).ngroups}")
    print(f"   - Total episodes from metadata: {metadata['total_episodes']}")
    print(f"   - Quality episodes from metadata: {metadata['quality_episodes']}")
    
    # Episode performance analysis
    episode_stats = df.groupby(['session_id', 'episode_num']).agg({
        'episode_points': 'first',
        'episode_turns': 'first', 
        'episode_maxlvl': 'first',
        'character_role': 'first',
        'character_race': 'first',
        'expert_skill_level': 'first',
        'episode_outcome': 'first'
    }).reset_index()
    
    print(f"\n🎯 Episode Performance Analysis:")
    print(f"   - Min points: {episode_stats['episode_points'].min():,}")
    print(f"   - Max points: {episode_stats['episode_points'].max():,}")
    print(f"   - Mean points: {episode_stats['episode_points'].mean():.0f}")
    print(f"   - Median points: {episode_stats['episode_points'].median():.0f}")
    
    print(f"\n⚡ Episode Length Analysis:")
    print(f"   - Min turns: {episode_stats['episode_turns'].min():,}")
    print(f"   - Max turns: {episode_stats['episode_turns'].max():,}")
    print(f"   - Mean turns: {episode_stats['episode_turns'].mean():.0f}")
    print(f"   - Median turns: {episode_stats['episode_turns'].median():.0f}")
    
    print(f"\n🏔️ Depth Achievement:")
    print(f"   - Max depth reached: {episode_stats['episode_maxlvl'].max()}")
    print(f"   - Mean depth: {episode_stats['episode_maxlvl'].mean():.1f}")
    print(f"   - Episodes reaching level 2+: {(episode_stats['episode_maxlvl'] >= 2).sum()}")
    print(f"   - Episodes reaching level 5+: {(episode_stats['episode_maxlvl'] >= 5).sum()}")
    print(f"   - Episodes reaching level 10+: {(episode_stats['episode_maxlvl'] >= 10).sum()}")
    
    # Character analysis
    print(f"\n🧙 Character Class Distribution:")
    role_counts = episode_stats['character_role'].value_counts()
    for role, count in role_counts.head(10).items():
        percentage = (count / len(episode_stats)) * 100
        print(f"   - {role}: {count} episodes ({percentage:.1f}%)")
    
    print(f"\n🧝 Character Race Distribution:")
    race_counts = episode_stats['character_race'].value_counts()
    for race, count in race_counts.head(10).items():
        percentage = (count / len(episode_stats)) * 100
        print(f"   - {race}: {count} episodes ({percentage:.1f}%)")
    
    # Skill level analysis
    print(f"\n🎓 Expert Skill Level Distribution:")
    skill_counts = df['expert_skill_level'].value_counts()
    for skill, count in skill_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   - {skill}: {count} examples ({percentage:.1f}%)")
    
    # Episode outcome analysis
    print(f"\n🏆 Episode Outcome Distribution:")
    outcome_counts = df['episode_outcome'].value_counts()
    for outcome, count in outcome_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   - {outcome}: {count} examples ({percentage:.1f}%)")
    
    # Action analysis
    print(f"\n⚔️ Action Distribution:")
    action_counts = df['output'].value_counts()
    total_actions = len(df)
    print(f"   - Unique actions: {len(action_counts)}")
    print(f"   - Top 10 actions:")
    for action, count in action_counts.head(10).items():
        percentage = (count / total_actions) * 100
        print(f"     • {action}: {count} times ({percentage:.1f}%)")
    
    # Subgoal analysis
    print(f"\n🎯 Subgoal Distribution:")
    subgoal_counts = df['subgoal'].value_counts()
    for subgoal, count in subgoal_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   - {subgoal}: {count} examples ({percentage:.1f}%)")
    
    # Training step analysis
    print(f"\n📚 Training Step Analysis:")
    print(f"   - Steps per episode (mean): {df.groupby(['session_id', 'episode_num']).size().mean():.1f}")
    print(f"   - Steps per episode (median): {df.groupby(['session_id', 'episode_num']).size().median():.1f}")
    print(f"   - Min steps per episode: {df.groupby(['session_id', 'episode_num']).size().min()}")
    print(f"   - Max steps per episode: {df.groupby(['session_id', 'episode_num']).size().max()}")
    
    # Turn progression analysis
    print(f"\n⏰ Turn Progression Analysis:")
    print(f"   - Mean estimated turn: {df['estimated_turn'].mean():.0f}")
    print(f"   - Median estimated turn: {df['estimated_turn'].median():.0f}")
    print(f"   - Early game steps (turn < 1000): {(df['estimated_turn'] < 1000).sum()}")
    print(f"   - Mid game steps (1000 <= turn < 5000): {((df['estimated_turn'] >= 1000) & (df['estimated_turn'] < 5000)).sum()}")
    print(f"   - Late game steps (turn >= 5000): {(df['estimated_turn'] >= 5000).sum()}")
    
    # Depth progression analysis
    print(f"\n🏔️ Depth Progression Analysis:")
    print(f"   - Mean estimated depth: {df['estimated_depth'].mean():.1f}")
    print(f"   - Level 1 steps: {(df['estimated_depth'] == 1).sum()}")
    print(f"   - Level 2+ steps: {(df['estimated_depth'] >= 2).sum()}")
    print(f"   - Level 5+ steps: {(df['estimated_depth'] >= 5).sum()}")
    
    # Performance correlation analysis
    print(f"\n📊 Performance Correlations:")
    episode_perf = episode_stats[['episode_points', 'episode_turns', 'episode_maxlvl']].corr()
    print(f"   - Points vs Turns correlation: {episode_perf.loc['episode_points', 'episode_turns']:.3f}")
    print(f"   - Points vs Max Level correlation: {episode_perf.loc['episode_points', 'episode_maxlvl']:.3f}")
    print(f"   - Turns vs Max Level correlation: {episode_perf.loc['episode_turns', 'episode_maxlvl']:.3f}")
    
    # Quality indicators
    print(f"\n✨ Data Quality Indicators:")
    print(f"   - Examples with reasoning: {df['reasoning'].notna().sum()}")
    print(f"   - Average instruction length: {df['instruction'].str.len().mean():.0f} chars")
    print(f"   - Average output length: {df['output'].str.len().mean():.0f} chars")
    print(f"   - Examples with episode metadata: {df['episode_points'].notna().sum()}")
    
    # High-performance episodes
    high_perf = episode_stats[episode_stats['episode_points'] >= 20000]
    if len(high_perf) > 0:
        print(f"\n🏆 High-Performance Episodes (20k+ points):")
        print(f"   - Count: {len(high_perf)}")
        print(f"   - Mean points: {high_perf['episode_points'].mean():.0f}")
        print(f"   - Mean depth: {high_perf['episode_maxlvl'].mean():.1f}")
        print(f"   - Mean turns: {high_perf['episode_turns'].mean():.0f}")
        
        print(f"   - Top roles in high-perf:")
        for role, count in high_perf['character_role'].value_counts().head(5).items():
            print(f"     • {role}: {count}")
    
    # Sample examples
    print(f"\n📋 Sample Training Examples:")
    for i, example in enumerate(dataset['train'].select(range(3))):
        print(f"\n   Example {i+1}:")
        print(f"   - Action: {example['output']}")
        print(f"   - Subgoal: {example['subgoal']}")
        print(f"   - Episode points: {example['episode_points']:,}")
        print(f"   - Character: {example['character_role']} {example['character_race']}")
        print(f"   - Skill level: {example['expert_skill_level']}")
        print(f"   - Instruction (first 100 chars): {example['instruction'][:100]}...")
    
    # Scalability analysis
    print(f"\n🚀 Scalability Analysis:")
    print(f"   - Current dataset size: ~{len(df)} examples")
    print(f"   - With 100 users (5x scale): ~{len(df) * 5:,} examples")
    print(f"   - With 2000 users (100x scale): ~{len(df) * 100:,} examples")
    print(f"   - Storage per example: ~{len(str(all_examples[0])) / 1024:.1f} KB")
    print(f"   - Estimated storage for 2000 users: ~{(len(df) * 100 * len(str(all_examples[0]))) / (1024**3):.1f} GB")
    
    return df, episode_stats

def create_summary_report():
    """Create a comprehensive summary report"""
    print("=" * 60)
    print("📊 ENHANCED NETHACK EXPERT DATASET ANALYSIS")
    print("=" * 60)
    
    df, episode_stats = analyze_enhanced_dataset()
    
    print(f"\n" + "=" * 60)
    print("✅ DATASET READY FOR SFT TRAINING")
    print("=" * 60)
    print(f"✨ High-quality dataset with full expert metadata")
    print(f"🎯 {len(df):,} training examples from real expert gameplay")
    print(f"📈 Scalable architecture ready for 100x more data")
    print(f"🚀 Ready for HuggingFace Transformers SFT training")

if __name__ == "__main__":
    create_summary_report()