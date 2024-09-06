##  Reward Learning  ##

import numpy as np
from collections import deque

class RewardFunction:
    def __init__(self):
        self.exploration_weight = 0.3
        self.resource_weight = 0.2
        self.survival_weight = 0.2
        self.discovery_weight = 0.15
        self.mission_weight = 0.15

        self.previous_inventory = {}
        self.previous_units = 0
        self.previous_nanites = 0
        self.previous_quicksilver = 0
        self.previous_discovered_species = set()
        
        self.health_history = deque(maxlen=100)
        self.oxygen_history = deque(maxlen=100)
        
        self.mission_progress = 0
        self.base_building_progress = 0

        # New attributes
        self.previous_technology = set()
        self.previous_milestones = set()
        self.previous_standing = {}
        self.previous_discovered_words = set()

    def calculate_reward(self, env):
        total_reward = (
            self.exploration_weight * self._exploration_reward(env) +
            self.resource_weight * self._resource_reward(env) +
            self.survival_weight * self._survival_reward(env) +
            self.discovery_weight * self._discovery_reward(env) +
            self.mission_weight * self._mission_reward(env)
        )
        
        # Update previous states
        self._update_previous_states(env)
        
        return total_reward

    def _exploration_reward(self, env):
        reward = 0
        
        # Reward for exploring new areas
        new_areas = len(env.explored_areas) - len(set(self.previous_inventory.keys()))
        reward += new_areas * 0.1
        
        # Reward for traveling distance
        if hasattr(env, 'previous_position'):
            distance = np.linalg.norm(np.array(env.position) - np.array(env.previous_position))
            reward += distance * 0.01
        
        # Reward for discovering new planets or systems
        if env.current_planet and env.current_planet not in env.discovered_planets:
            reward += 10
        
        # New: Reward for using portals or teleporters
        if hasattr(env, 'used_portal') and env.used_portal:
            reward += 5
        
        # New: Reward for discovering and activating ancient alien structures
        if hasattr(env, 'activated_structures'):
            reward += len(env.activated_structures) * 2
        
        return reward

    def _resource_reward(self, env):
        reward = 0
        
        # Reward for gathering resources
        for item, quantity in env.inventory.items():
            if item not in self.previous_inventory:
                reward += quantity * 0.5  # New item bonus
            else:
                reward += (quantity - self.previous_inventory[item]) * 0.1
        
        # Reward for earning units, nanites, and quicksilver
        reward += (env.units - self.previous_units) * 0.001
        reward += (env.nanites - self.previous_nanites) * 0.01
        reward += (env.quicksilver - self.previous_quicksilver) * 0.1
        
        # Reward for crafting items
        if hasattr(env, 'crafted_items'):
            reward += len(env.crafted_items) * 1
        
        # New: Reward for efficient inventory management
        if hasattr(env, 'inventory_capacity'):
            free_space = env.inventory_capacity - sum(env.inventory.values())
            reward += free_space * 0.01  # Small reward for maintaining free inventory space
        
        # New: Reward for selling items at a profit
        if hasattr(env, 'sold_items'):
            for item, profit in env.sold_items.items():
                reward += profit * 0.05
        
        return reward

    def _survival_reward(self, env):
        reward = 0
        
        # Penalize damage
        health_change = env.health - (self.health_history[-1] if self.health_history else 100)
        oxygen_change = env.oxygen - (self.oxygen_history[-1] if self.oxygen_history else 100)
        
        reward += health_change * 0.1
        reward += oxygen_change * 0.1
        
        # Reward for maintaining high health and oxygen
        reward += (env.health / 100) * 0.1
        reward += (env.oxygen / 100) * 0.1
        
        # Reward for surviving in hazardous environments
        if hasattr(env, 'hazard_level'):
            reward += env.hazard_level * 0.1
        
        # New: Reward for efficient use of life support systems
        if hasattr(env, 'life_support_efficiency'):
            reward += env.life_support_efficiency * 0.2
        
        # New: Reward for surviving extreme weather events
        if hasattr(env, 'survived_extreme_event') and env.survived_extreme_event:
            reward += 5
        
        return reward

    def _discovery_reward(self, env):
        reward = 0
        
        # Reward for discovering new species
        new_species = len(env.discovered_species) - len(self.previous_discovered_species)
        reward += new_species * 5
        
        # Reward for scanning flora, fauna, and minerals
        if hasattr(env, 'scanned_objects'):
            reward += len(env.scanned_objects) * 0.5
        
        # Reward for uploading discoveries
        if hasattr(env, 'uploaded_discoveries'):
            reward += len(env.uploaded_discoveries) * 2
        
        # New: Reward for learning alien languages
        new_words = len(env.discovered_words) - len(self.previous_discovered_words)
        reward += new_words * 0.5
        
        # New: Reward for discovering rare resources or exotic planets
        if hasattr(env, 'discovered_rare_resources'):
            reward += len(env.discovered_rare_resources) * 3
        
        if hasattr(env, 'discovered_exotic_planet') and env.discovered_exotic_planet:
            reward += 10
        
        return reward

    def _mission_reward(self, env):
        reward = 0
        
        # Reward for mission progress
        if hasattr(env, 'mission_progress'):
            progress_change = env.mission_progress - self.mission_progress
            reward += progress_change * 10
        
        # Reward for base building progress
        if hasattr(env, 'base_building_progress'):
            base_progress_change = env.base_building_progress - self.base_building_progress
            reward += base_progress_change * 5
        
        # Reward for completing missions
        if hasattr(env, 'completed_missions'):
            reward += len(env.completed_missions) * 20
        
        # New: Reward for unlocking new technologies
        new_tech = len(env.unlocked_technology) - len(self.previous_technology)
        reward += new_tech * 5
        
        # New: Reward for achieving milestones
        new_milestones = len(env.achieved_milestones) - len(self.previous_milestones)
        reward += new_milestones * 10
        
        # New: Reward for improving standing with alien races
        for race, standing in env.faction_standing.items():
            if race in self.previous_standing:
                standing_change = standing - self.previous_standing[race]
                reward += standing_change * 2
            else:
                reward += standing * 2  # New race discovered
        
        return reward

    def _update_previous_states(self, env):
        self.previous_inventory = env.inventory.copy()
        self.previous_units = env.units
        self.previous_nanites = env.nanites
        self.previous_quicksilver = env.quicksilver
        self.previous_discovered_species = env.discovered_species.copy()
        
        self.health_history.append(env.health)
        self.oxygen_history.append(env.oxygen)
        
        if hasattr(env, 'mission_progress'):
            self.mission_progress = env.mission_progress
        if hasattr(env, 'base_building_progress'):
            self.base_building_progress = env.base_building_progress

        # Update new attributes
        self.previous_technology = env.unlocked_technology.copy()
        self.previous_milestones = env.achieved_milestones.copy()
        self.previous_standing = env.faction_standing.copy()
        self.previous_discovered_words = env.discovered_words.copy()

    def reset(self):
        self.previous_inventory = {}
        self.previous_units = 0
        self.previous_nanites = 0
        self.previous_quicksilver = 0
        self.previous_discovered_species = set()
        
        self.health_history.clear()
        self.oxygen_history.clear()
        
        self.mission_progress = 0
        self.base_building_progress = 0

        # Reset new attributes
        self.previous_technology = set()
        self.previous_milestones = set()
        self.previous_standing = {}
        self.previous_discovered_words = set()
