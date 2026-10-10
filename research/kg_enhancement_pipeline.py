#!/usr/bin/env python3
"""
KG Enhancement Pipeline - Post-process to achieve A+ quality
"""

import pandas as pd
import re
from collections import defaultdict
import json
from typing import Dict, List, Tuple

class KGEnhancer:
    def __init__(self):
        # Define precise relation mappings
        self.relation_refinements = {
            'special_attack': {
                # Map generic actions to proper relations
                'patterns': [
                    (r'(take|steal|pick up)', 'can_obtain'),
                    (r'(hit|attack|strike)', 'combat_action'),
                    (r'(use|wield|wear)', 'can_use'),
                    (r'(cure|heal|restore)', 'healing_effect'),
                    (r'(turn to stone|petrif)', 'causes_stoning'),
                    (r'(teleport|warp)', 'causes_teleport'),
                    (r'(reflect|bounce)', 'reflects_attack'),
                    (r'(resist|immune)', 'provides_resistance'),
                    (r'(grant|confer|give)', 'grants_ability'),
                    (r'(damage|hurt|harm)', 'deals_damage'),
                    (r'(summon|create|generate)', 'summons_creature'),
                    (r'(transform|polymorph|change)', 'causes_polymorph'),
                    (r'(instakill|instant death)', 'instant_death'),
                    (r'(paralyze|freeze|stun)', 'causes_paralysis'),
                    (r'(drain|sap|reduce)', 'drains_attribute'),
                ],
                'default': 'special_ability'
            }
        }
        
        # Relations that should be normalized
        self.value_normalizations = {
            'alignment': {
                '−15': 'chaotic (-15)',
                '0': 'neutral (0)',
                '+': 'lawful',
                '-': 'chaotic',
                'neutral': 'neutral (0)'
            },
            'has_resistance': {
                'resistance resistance': 'poison resistance',  # Common error
                'fire': 'fire resistance',
                'cold': 'cold resistance',
                'shock': 'shock resistance',
                'sleep': 'sleep resistance',
                'poison': 'poison resistance',
                'acid': 'acid resistance',
                'stone': 'petrification resistance',
                'magic': 'magic resistance'
            }
        }
        
        # Relations that need validation
        self.relation_validators = {
            'attack_damage': self._validate_dice_notation,
            'armor_class': self._validate_number,
            'speed': self._validate_number,
            'difficulty': self._validate_number,
            'base_level': self._validate_number,
            'cost': self._validate_cost,
            'weight': self._validate_number,
            'nutrition': self._validate_number,
        }
        
        # Type cleanup patterns
        self.type_cleanup = {
            r'"prettytable.*"': None,  # Remove HTML artifacts
            r'"wikitable.*"': None,
            r'chance$': 'random_generation',
            r'^monster$': 'monster_class',
            r'^item$': 'item_class',
            r'^weapon$': 'weapon_class',
            r'^armor$': 'armor_class',
        }
        
    def enhance_kg(self, input_path: str, output_path: str):
        """Main enhancement pipeline"""
        print(f"Loading KG from {input_path}")
        df = pd.read_csv(input_path, sep='\t')
        original_count = len(df)
        
        print(f"Original triples: {original_count}")
        
        # Step 1: Clean and refine special_attack relations
        print("\n1. Refining special_attack relations...")
        df = self._refine_special_attacks(df)
        
        # Step 2: Normalize values
        print("\n2. Normalizing values...")
        df = self._normalize_values(df)
        
        # Step 3: Validate and fix data types
        print("\n3. Validating data types...")
        df = self._validate_datatypes(df)
        
        # Step 4: Remove low-quality triples
        print("\n4. Removing low-quality triples...")
        df = self._remove_low_quality(df)
        
        # Step 5: Fix type_of relations
        print("\n5. Fixing type_of relations...")
        df = self._fix_type_relations(df)
        
        # Step 6: Add missing relations
        print("\n6. Adding inferred relations...")
        df = self._add_inferred_relations(df)
        
        # Step 7: Deduplicate semantically
        print("\n7. Semantic deduplication...")
        df = self._semantic_dedup(df)
        
        # Step 8: Final quality check
        print("\n8. Final quality check...")
        df = self._final_quality_check(df)
        
        # Save enhanced KG
        df.to_csv(output_path, sep='\t', index=False)
        
        print(f"\nEnhanced KG saved to {output_path}")
        print(f"Final triples: {len(df)} (removed {original_count - len(df)} low-quality triples)")
        
        # Generate quality report
        self._generate_quality_report(df, output_path.replace('.tsv', '_quality_report.json'))
        
        return df
    
    def _refine_special_attacks(self, df):
        """Refine special_attack relations to more specific types"""
        refined_rows = []
        
        for idx, row in df.iterrows():
            if row['type'] == 'special_attack':
                tail_lower = str(row['tail']).lower()
                new_relation = 'special_ability'  # default
                
                # Check patterns
                for pattern, new_rel in self.relation_refinements['special_attack']['patterns']:
                    if re.search(pattern, tail_lower):
                        new_relation = new_rel
                        break
                
                row['type'] = new_relation
            
            refined_rows.append(row)
        
        return pd.DataFrame(refined_rows)
    
    def _normalize_values(self, df):
        """Normalize values based on relation type"""
        for idx, row in df.iterrows():
            rel_type = row['type']
            
            if rel_type in self.value_normalizations:
                normalizations = self.value_normalizations[rel_type]
                tail_str = str(row['tail']).strip()
                
                # Direct mapping
                if tail_str in normalizations:
                    df.at[idx, 'tail'] = normalizations[tail_str]
                # Partial matching for resistances
                elif rel_type == 'has_resistance':
                    for key, normalized in normalizations.items():
                        if key in tail_str.lower() and 'resistance' not in tail_str:
                            df.at[idx, 'tail'] = normalized
                            break
        
        return df
    
    def _validate_dice_notation(self, value):
        """Validate dice notation (e.g., 2d4)"""
        return bool(re.match(r'^\d+d\d+$', str(value)))
    
    def _validate_number(self, value):
        """Validate numeric value"""
        try:
            float(str(value).replace('+', '').replace('-', ''))
            return True
        except:
            return False
    
    def _validate_cost(self, value):
        """Validate cost values"""
        value_str = str(value)
        return bool(re.match(r'^\d+\s*(zm|zorkmids?)?$', value_str, re.IGNORECASE))
    
    def _validate_datatypes(self, df):
        """Validate data types for specific relations"""
        valid_indices = []
        
        for idx, row in df.iterrows():
            rel_type = row['type']
            
            if rel_type in self.relation_validators:
                validator = self.relation_validators[rel_type]
                if validator(row['tail']):
                    valid_indices.append(idx)
                else:
                    print(f"  Invalid {rel_type}: {row['head']} -> {row['tail']}")
            else:
                valid_indices.append(idx)
        
        return df.loc[valid_indices]
    
    def _remove_low_quality(self, df):
        """Remove low-quality triples"""
        # Criteria for removal
        keep_indices = []
        
        for idx, row in df.iterrows():
            head = str(row['head']).strip()
            tail = str(row['tail']).strip()
            
            # Skip if too short
            if len(head) < 2 or len(tail) < 2:
                continue
            
            # Skip if contains wiki markup
            if any(marker in head + tail for marker in ['{{', '}}', '[[', ']]', '<', '>', '&mdash;']):
                continue
            
            # Skip if tail is truncated
            if tail.endswith('...') or tail.endswith('&mdash'):
                continue
            
            # Skip nonsensical type_of relations
            if row['type'] == 'type_of' and tail in ['chance', '"prettytable"', '"wikitable"']:
                continue
            
            # Skip if self-loop (unless it's an alias)
            if head.lower() == tail.lower() and row['type'] != 'also_known_as':
                continue
            
            keep_indices.append(idx)
        
        return df.loc[keep_indices]
    
    def _fix_type_relations(self, df):
        """Fix type_of relations"""
        for idx, row in df.iterrows():
            if row['type'] == 'type_of':
                tail = str(row['tail'])
                
                # Apply cleanup patterns
                for pattern, replacement in self.type_cleanup.items():
                    if re.search(pattern, tail):
                        if replacement:
                            df.at[idx, 'tail'] = replacement
                        else:
                            # Mark for removal
                            df.at[idx, 'type'] = 'REMOVE'
                        break
        
        # Remove marked rows
        return df[df['type'] != 'REMOVE']
    
    def _add_inferred_relations(self, df):
        """Add inferred relations based on existing data"""
        new_triples = []
        
        # Group by entity
        entity_facts = defaultdict(dict)
        for _, row in df.iterrows():
            entity_facts[row['head']][row['type']] = row['tail']
        
        # Infer new relations
        for entity, facts in entity_facts.items():
            # If has attack_damage, infer it's a creature
            if 'attack_damage' in facts and 'type_of' not in facts:
                new_triples.append({
                    'head': entity,
                    'type': 'type_of',
                    'tail': 'creature',
                    'source': 'inferred'
                })
            
            # If has armor_class but no type, it's likely a monster
            if 'armor_class' in facts and 'type_of' not in facts:
                new_triples.append({
                    'head': entity,
                    'type': 'type_of',
                    'tail': 'monster',
                    'source': 'inferred'
                })
            
            # If grants resistance, add grants_resistance relation
            if 'grants' in facts and 'resistance' in str(facts['grants']):
                resistance_match = re.search(r'(\w+)\s+resistance', str(facts['grants']))
                if resistance_match:
                    new_triples.append({
                        'head': entity,
                        'type': 'grants_resistance',
                        'tail': f"{resistance_match.group(1)} resistance",
                        'source': 'inferred'
                    })
        
        # Add new triples
        if new_triples:
            new_df = pd.DataFrame(new_triples)
            df = pd.concat([df, new_df], ignore_index=True)
        
        return df
    
    def _semantic_dedup(self, df):
        """Remove semantic duplicates"""
        # Group by head, type and check for similar tails
        seen = set()
        keep_indices = []
        
        for idx, row in df.iterrows():
            # Create normalized key
            key = (
                str(row['head']).lower().strip(),
                row['type'],
                str(row['tail']).lower().strip()
            )
            
            if key not in seen:
                seen.add(key)
                keep_indices.append(idx)
        
        return df.loc[keep_indices]
    
    def _final_quality_check(self, df):
        """Final quality check and fixes"""
        # Remove any remaining problematic entries
        quality_indices = []
        
        for idx, row in df.iterrows():
            # Final checks
            if len(str(row['tail'])) > 200:  # Too long
                continue
            
            if row['type'] == 'type_of' and str(row['tail']) == str(row['head']):
                continue
            
            quality_indices.append(idx)
        
        return df.loc[quality_indices]
    
    def _generate_quality_report(self, df, report_path):
        """Generate quality report"""
        report = {
            'total_triples': len(df),
            'unique_entities': len(set(df['head'].tolist() + df['tail'].tolist())),
            'unique_relations': len(df['type'].unique()),
            'relation_distribution': df['type'].value_counts().to_dict(),
            'quality_metrics': {
                'avg_facts_per_entity': len(df) / df['head'].nunique(),
                'game_specific_relations': len([r for r in df['type'].unique() 
                                               if any(term in r for term in 
                                                     ['damage', 'resistance', 'armor', 'attack'])]),
                'validated_numeric_facts': len(df[df['type'].isin(self.relation_validators.keys())]),
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nQuality report saved to {report_path}")


def main():
    enhancer = KGEnhancer()
    
    # Enhance the partial KG
    enhanced_df = enhancer.enhance_kg(
        'data/nethack_kg_partial.tsv',
        'data/nethack_kg_enhanced.tsv'
    )
    
    print("\nTop relations in enhanced KG:")
    print(enhanced_df['type'].value_counts().head(20))


if __name__ == "__main__":
    main()