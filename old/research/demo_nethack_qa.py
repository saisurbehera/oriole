"""
Test and demonstrate the NetHack KGQA system capabilities
"""

from nethack_kgqa import NetHackKGQA
import time

def test_comprehensive_qa():
    """Test all types of queries with detailed output"""
    
    print("="*80)
    print("NetHack Knowledge Graph Question Answering System")
    print("Demonstrating Advanced Features")
    print("="*80)
    
    # Initialize
    print("\nLoading system...")
    start = time.time()
    qa = NetHackKGQA('data/nethack_kg_full/nethack_kg_intermediate.tsv')
    print(f"Loaded in {time.time() - start:.2f} seconds")
    print(f"Knowledge base: {len(qa.entities)} entities, {len(qa.relations)} relations")
    
    # Test cases covering all features
    test_cases = [
        # 1. Semantic matching with typos/variations
        {
            'category': 'Semantic Entity Matching',
            'questions': [
                "What is the Amulet of Yender?",  # Typo
                "Tell me about ammulet of yendor",  # Misspelling
                "aoy properties",  # Abbreviation
                "What's a wizrd of yendor?"  # Typo
            ]
        },
        
        # 2. Contextual relevance
        {
            'category': 'Contextual Understanding',
            'questions': [
                "What items are found in the castle?",
                "Which monsters appear in Gehennom?",
                "What's in Sokoban?",
                "Items used by wizards"
            ]
        },
        
        # 3. Multi-hop reasoning
        {
            'category': 'Complex Reasoning',
            'questions': [
                "How is the Amulet of Yendor related to ascension?",
                "What connects NetHack to Rogue?",
                "Path from Dungeons of Doom to Astral Plane",
                "How are variants related to NetHack?"
            ]
        },
        
        # 4. Aggregation and counting
        {
            'category': 'Statistical Queries',
            'questions': [
                "How many NetHack variants exist?",
                "Count all artifacts",
                "How many types of monsters are there?",
                "List all items in NetHack"
            ]
        },
        
        # 5. Comparison queries
        {
            'category': 'Comparisons',
            'questions': [
                "What's the difference between blessed and cursed?",
                "Compare SLASH'EM and UnNetHack",
                "Difference between wand and scroll",
                "NetHack vs Hack'EM"
            ]
        },
        
        # 6. Boolean/verification
        {
            'category': 'Fact Verification',
            'questions': [
                "Is Sokoban part of NetHack?",
                "Can you eat a cockatrice corpse?",
                "Does the Amulet of Yendor grant wishes?",
                "Is Excalibur an artifact?"
            ]
        },
        
        # 7. Natural language variations
        {
            'category': 'Natural Language Understanding',
            'questions': [
                "Where can I find the amulet?",
                "Who guards the Amulet of Yendor?",
                "What do I need for ascension?",
                "How do I get to the astral plane?"
            ]
        },
        
        # 8. Confidence scoring edge cases
        {
            'category': 'Confidence Testing',
            'questions': [
                "What is a flibbertigibbet?",  # Nonsense
                "How is NetHack related to Minecraft?",  # Unrelated
                "What color is the Amulet of Yendor?",  # Missing info
                "Can you polymorph into a grid bug?"  # Specific game mechanics
            ]
        }
    ]
    
    # Run tests
    for test_group in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_group['category']}")
        print('='*60)
        
        for question in test_group['questions']:
            print(f"\nQ: {question}")
            
            # Get answer
            answer = qa.answer(question)
            
            # Display results
            print(f"A: {answer.text}")
            print(f"Confidence: {answer.confidence:.1%}")
            print(f"Type: {answer.answer_type}")
            
            # Show evidence if high confidence
            if answer.confidence > 0.7 and answer.evidence_triples:
                print("Evidence:")
                for h, r, t in answer.evidence_triples[:2]:
                    print(f"  • {h} --[{r}]--> {t}")
            
            # Show evaluation breakdown
            if answer.evaluation_breakdown:
                scores = answer.evaluation_breakdown
                print("Scoring breakdown:")
                for metric, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"  • {metric}: {score:.3f}")
            
            # Show alternatives if low confidence
            if answer.confidence < 0.7 and answer.alternative_answers:
                print("Alternatives considered:")
                for alt, conf in answer.alternative_answers[:2]:
                    print(f"  • {alt} ({conf:.1%})")
            
            print("-" * 40)
    
    # Interactive demo
    print("\n" + "="*60)
    print("INTERACTIVE DEMO - Try your own questions!")
    print("="*60)
    print("\nExamples to try:")
    print("- What monsters drop good loot?")
    print("- How do I become a god?")
    print("- What's the most powerful weapon?")
    print("- Is it safe to pray in Gehennom?")
    print("- What happens if you eat too much?")
    print("\nType 'quit' to exit\n")
    
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        answer = qa.answer(question)
        
        print(f"\nAnswer: {answer.text}")
        print(f"Confidence: {answer.confidence:.1%}")
        
        # Show reasoning path if interesting
        if answer.reasoning_path:
            print(f"Reasoning: {' → '.join(answer.reasoning_path)}")
        
        print()


if __name__ == "__main__":
    test_comprehensive_qa()