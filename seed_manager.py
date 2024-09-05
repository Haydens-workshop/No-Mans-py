# seed_manager.py

import random
import hashlib
import json
import os
from typing import List, Dict, Optional

class SeedManager:
    def __init__(self, seed_file: str = "predefined_seeds.json"):
        self.used_seeds: set = set()
        self.predefined_seeds: List[str] = []
        self.seed_categories: Dict[str, List[str]] = {}
        self.current_seed: Optional[str] = None
        self.seed_file = seed_file
        self.load_predefined_seeds()

    def load_predefined_seeds(self):
        if os.path.exists(self.seed_file):
            with open(self.seed_file, 'r') as f:
                data = json.load(f)
                self.predefined_seeds = data.get('general', [])
                self.seed_categories = {k: v for k, v in data.items() if k != 'general'}
        else:
            print(f"Warning: Seed file '{self.seed_file}' not found. Using empty predefined seed list.")

    def save_predefined_seeds(self):
        data = {'general': self.predefined_seeds, **self.seed_categories}
        with open(self.seed_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_predefined_seed(self, seed: str, category: Optional[str] = None):
        if category:
            if category not in self.seed_categories:
                self.seed_categories[category] = []
            if seed not in self.seed_categories[category]:
                self.seed_categories[category].append(seed)
        else:
            if seed not in self.predefined_seeds:
                self.predefined_seeds.append(seed)
        self.save_predefined_seeds()

    def remove_predefined_seed(self, seed: str, category: Optional[str] = None):
        if category and category in self.seed_categories:
            self.seed_categories[category] = [s for s in self.seed_categories[category] if s != seed]
        else:
            self.predefined_seeds = [s for s in self.predefined_seeds if s != seed]
        self.save_predefined_seeds()

    def get_new_seed(self, category: Optional[str] = None) -> str:
        if category and category in self.seed_categories and self.seed_categories[category]:
            seed_pool = self.seed_categories[category]
        elif self.predefined_seeds:
            seed_pool = self.predefined_seeds
        else:
            return self._generate_random_seed()

        available_seeds = [seed for seed in seed_pool if seed not in self.used_seeds]
        if available_seeds:
            self.current_seed = random.choice(available_seeds)
        else:
            self.current_seed = self._generate_random_seed()

        self.used_seeds.add(self.current_seed)
        return self.current_seed

    def _generate_random_seed(self) -> str:
        while True:
            new_seed = hashlib.md5(str(random.getrandbits(128)).encode()).hexdigest()
            if new_seed not in self.used_seeds:
                return new_seed

    def get_current_seed(self) -> Optional[str]:
        return self.current_seed

    def reset_used_seeds(self):
        self.used_seeds.clear()
        self.current_seed = None

    def get_seed_info(self, seed: str) -> Dict[str, any]:
        # This method would ideally interact with the game or a database to get information about a specific seed
        # For now, we'll return a placeholder dictionary
        return {
            "seed": seed,
            "planet_type": random.choice(["Lush", "Toxic", "Radioactive", "Desert", "Frozen", "Barren"]),
            "resources": random.sample(["Carbon", "Ferrite Dust", "Sodium", "Oxygen", "Copper", "Gold"], 3),
            "hazard_level": random.randint(0, 100)
        }

    def generate_seed_sequence(self, length: int, category: Optional[str] = None) -> List[str]:
        return [self.get_new_seed(category) for _ in range(length)]

    def add_seed_category(self, category: str, seeds: List[str] = []):
        if category not in self.seed_categories:
            self.seed_categories[category] = seeds
            self.save_predefined_seeds()

    def remove_seed_category(self, category: str):
        if category in self.seed_categories:
            del self.seed_categories[category]
            self.save_predefined_seeds()

    def get_seed_categories(self) -> List[str]:
        return list(self.seed_categories.keys())

    def get_seeds_in_category(self, category: str) -> List[str]:
        return self.seed_categories.get(category, [])

    def is_seed_used(self, seed: str) -> bool:
        return seed in self.used_seeds

    def mark_seed_as_used(self, seed: str):
        self.used_seeds.add(seed)

    def get_unused_seeds(self, category: Optional[str] = None) -> List[str]:
        if category and category in self.seed_categories:
            seed_pool = self.seed_categories[category]
        else:
            seed_pool = self.predefined_seeds

        return [seed for seed in seed_pool if seed not in self.used_seeds]

    def generate_hybrid_seed(self, seed1: str, seed2: str) -> str:
        # Create a new seed by combining characteristics of two existing seeds
        combined = hashlib.md5((seed1 + seed2).encode()).hexdigest()
        return combined

    def mutate_seed(self, seed: str, mutation_rate: float = 0.1) -> str:
        # Create a slightly modified version of an existing seed
        seed_bytes = bytearray.fromhex(seed)
        for i in range(len(seed_bytes)):
            if random.random() < mutation_rate:
                seed_bytes[i] = random.randint(0, 255)
        return seed_bytes.hex()

# Example usage:
if __name__ == "__main__":
    seed_manager = SeedManager()
    
    # Add some predefined seeds
    seed_manager.add_predefined_seed("0xe876443f6a0a28a6", "frigate")
    seed_manager.add_predefined_seed("0x1a2b3c4d5e6f7890", "planet")
    
    # Get a new seed
    new_seed = seed_manager.get_new_seed()
    print(f"New seed: {new_seed}")
    
    # Get seed info
    seed_info = seed_manager.get_seed_info(new_seed)
    print(f"Seed info: {seed_info}")
    
    # Generate a seed sequence
    seed_sequence = seed_manager.generate_seed_sequence(5)
    print(f"Seed sequence: {seed_sequence}")
    
    # Demonstrate hybrid and mutated seeds
    hybrid_seed = seed_manager.generate_hybrid_seed(seed_sequence[0], seed_sequence[1])
    print(f"Hybrid seed: {hybrid_seed}")
    
    mutated_seed = seed_manager.mutate_seed(hybrid_seed)
    print(f"Mutated seed: {mutated_seed}")