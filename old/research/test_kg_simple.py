"""
Simple test script for KG operations without full ULTRA dependencies
"""

from collections import defaultdict

class SimpleKGWrapper:
    """Simplified KG wrapper for testing without ULTRA dependencies"""
    
    def __init__(self):
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.triples = []
        self.graph = defaultdict(lambda: defaultdict(set))
        
    def parse_kg_from_file(self, file_path, delimiter="\t"):
        """Load KG from TSV file"""
        triples = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) == 3:
                    triples.append(tuple(parts))
        self.parse_kg_from_triples(triples)
        
    def parse_kg_from_triples(self, triples):
        """Parse KG from list of triples"""
        entities = set()
        relations = set()
        
        self.triples = triples
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
            # Build graph structure
            self.graph[h][r].add(t)
        
        # Create mappings
        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2relation = {i: r for r, i in self.relation2id.items()}
        
    def get_entity_relations(self, entity):
        """Get all relations for an entity"""
        if entity not in self.graph:
            return []
        return list(self.graph[entity].keys())
    
    def get_neighbors(self, entity, relation):
        """Get neighbors via a specific relation"""
        return list(self.graph[entity].get(relation, []))
    
    def answer_simple_query(self, head, relation):
        """Answer simple link prediction query"""
        return self.get_neighbors(head, relation)
    
    def get_stats(self):
        """Get KG statistics"""
        stats = {
            'num_entities': len(self.entity2id),
            'num_relations': len(self.relation2id),
            'num_triples': len(self.triples)
        }
        
        # Relation distribution
        rel_dist = defaultdict(int)
        for h, r, t in self.triples:
            rel_dist[r] += 1
        stats['relation_distribution'] = dict(rel_dist)
        
        return stats


def test_nethack_kg():
    """Test with NetHack A+ knowledge graph"""
    print("=== Testing NetHack A+ KG ===\n")
    
    # Load the A+ KG with pandas first to handle the header
    import pandas as pd
    df = pd.read_csv("data/nethack_kg_final_a_plus.tsv", sep='\t')
    
    # Convert to triples format
    triples = []
    for _, row in df.iterrows():
        triples.append((str(row['head']), str(row['type']), str(row['tail'])))
    
    kg = SimpleKGWrapper()
    kg.parse_kg_from_triples(triples[:5000])  # Use subset for testing
    
    # Show stats
    stats = kg.get_stats()
    print(f"Loaded KG with {stats['num_entities']} entities and {stats['num_relations']} relations")
    print(f"Total triples: {stats['num_triples']}")
    
    print("\nSample entities:", list(kg.entity2id.keys())[:10])
    print("Relations:", list(kg.relation2id.keys())[:10])
    
    # Test queries
    print("\n--- Query Examples ---")
    
    # Query 1: What attack damage does Medusa have?
    damage = kg.answer_simple_query("Medusa", "attack_damage")
    print(f"\nMedusa attack damage: {damage}")
    
    # Query 2: What resistances does Soldier ant have?
    resistances = kg.answer_simple_query("Soldier ant", "has_resistance")
    print(f"\nSoldier ant resistances: {resistances}")
    
    # Query 3: What creatures are extremely hard?
    hard_creatures = []
    for entity in list(kg.entity2id.keys())[:100]:  # Check first 100 entities
        if "extremely hard" in kg.answer_simple_query(entity, "difficulty_category"):
            hard_creatures.append(entity)
    print(f"\nExtremely hard creatures: {hard_creatures}")
    
    # Test MRR on this subset
    print("\n--- MRR Test ---")
    test_mrr_simple(kg)
    
    # Query 2: Who acted in Inception? (Need to search backwards)
    actors = []
    for entity in kg.entity2id:
        if "Inception" in kg.answer_simple_query(entity, "acted_in"):
            actors.append(entity)
    print(f"Actors in Inception: {actors}")
    
    # Query 3: What genres is Interstellar?
    genres = kg.answer_simple_query("Interstellar", "genre")
    print(f"Interstellar genres: {genres}")
    
    # Show entity relations
    print("\n--- Entity Relations ---")
    for entity in ["Christopher_Nolan", "Inception", "Warner_Bros"]:
        relations = kg.get_entity_relations(entity)
        print(f"{entity} has relations: {relations}")


def test_tech_kg():
    """Test with tech companies knowledge graph"""
    print("\n\n=== Testing Tech KG ===\n")
    
    kg = SimpleKGWrapper()
    kg.parse_kg_from_file("sample_data/tech_kg.tsv")
    
    stats = kg.get_stats()
    print(f"Loaded KG with {stats['num_entities']} entities and {stats['num_relations']} relations")
    
    # Show relation distribution
    print("\nTop relations by frequency:")
    rel_dist = sorted(stats['relation_distribution'].items(), key=lambda x: x[1], reverse=True)
    for rel, count in rel_dist[:5]:
        print(f"  {rel}: {count} triples")
    
    # Test queries
    print("\n--- Query Examples ---")
    
    # What companies does Google own?
    companies = kg.answer_simple_query("Google", "owns")
    print(f"\nGoogle owns: {companies}")
    
    # Who are the CEOs? (Need to search backwards)
    print("\nCompany CEOs:")
    for company in ["Google", "Apple", "Microsoft"]:
        ceos = []
        for person in kg.entity2id:
            if company in kg.answer_simple_query(person, "ceo_of"):
                ceos.append(person)
        if ceos:
            print(f"  {company}: {ceos[0]}")
    
    # Where are companies headquartered?
    print("\nCompany headquarters:")
    for company in ["Google", "Apple", "Microsoft"]:
        hq = kg.answer_simple_query(company, "headquartered_in")
        print(f"  {company}: {hq}")


def test_path_queries(kg):
    """Test multi-hop path queries"""
    print("\n--- Path Query Examples ---")
    
    # Find companies in California (2-hop: company -> city -> state)
    print("\nCompanies in California (2-hop query):")
    cities_in_ca = []
    for city in kg.entity2id:
        if "California" in kg.answer_simple_query(city, "located_in"):
            cities_in_ca.append(city)
    
    companies_in_ca = []
    for company in ["Google", "Apple", "Microsoft"]:
        hq = kg.answer_simple_query(company, "headquartered_in")
        if hq and hq[0] in cities_in_ca:
            companies_in_ca.append(company)
    
    print(f"Companies headquartered in California: {companies_in_ca}")


def test_custom_analysis():
    """Custom analysis on the KG"""
    print("\n\n=== Custom Analysis ===\n")
    
    # Analyze movie KG
    kg = SimpleKGWrapper()
    kg.parse_kg_from_file("sample_data/movies_kg.tsv")
    
    print("Movie KG Analysis:")
    
    # Find all Sci-Fi movies
    scifi_movies = []
    for entity in kg.entity2id:
        if "Sci-Fi" in kg.answer_simple_query(entity, "genre"):
            scifi_movies.append(entity)
    print(f"Sci-Fi movies: {scifi_movies}")
    
    # Find movies by year
    print("\nMovies by release year:")
    for year in ["2008", "2010", "2013", "2014"]:
        movies = []
        for entity in kg.entity2id:
            if year in kg.answer_simple_query(entity, "released_in"):
                movies.append(entity)
        if movies:
            print(f"  {year}: {movies}")
    
    # Analyze tech KG
    kg_tech = SimpleKGWrapper()
    kg_tech.parse_kg_from_file("sample_data/tech_kg.tsv")
    
    print("\n\nTech KG Analysis:")
    
    # Find all co-founders
    print("Company co-founders:")
    for company in ["Google", "Apple", "Microsoft"]:
        founders = []
        for person in kg_tech.entity2id:
            if company in kg_tech.answer_simple_query(person, "co-founded"):
                founders.append(person)
        if founders:
            print(f"  {company}: {founders}")


if __name__ == "__main__":
    print("Simple KG Test Suite")
    print("=" * 50)
    
    # Run tests
    test_movies_kg()
    test_tech_kg()
    
    # Additional analysis
    kg_tech = SimpleKGWrapper()
    kg_tech.parse_kg_from_file("sample_data/tech_kg.tsv")
    test_path_queries(kg_tech)
    
    test_custom_analysis()
    
    print("\n" + "=" * 50)
    print("Tests completed successfully!")
    print("\nThis demonstrates basic KG operations:")
    print("- Loading KGs from TSV files")
    print("- Simple link prediction queries")
    print("- Multi-hop path queries")
    print("- KG statistics and analysis")