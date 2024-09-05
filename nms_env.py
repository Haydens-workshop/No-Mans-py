# nms_env.py

import gym
import numpy as np
import pyautogui
import cv2
import time
import logging
from gym import spaces
from seed_manager import SeedManager
from reward_function import RewardFunction
from utils import capture_screen, process_image, detect_text_on_screen, detect_objects_on_screen

class NoMansSkyEnv(gym.Env):
    def __init__(self, seed_manager, reward_function):
        super(NoMansSkyEnv, self).__init__()
        
        self.seed_manager = seed_manager
        self.reward_function = reward_function
        
        # Define action and observation space
        self.action_space = spaces.Discrete(12)  # W, A, S, D, E, F, X, Mouse1, Mouse2, I, Tab, Esc
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        # Game state variables
        self.current_seed = None
        self.inventory = {}
        self.position = (0, 0, 0)
        self.explored_areas = set()
        self.health = 100
        self.shields = 100
        self.oxygen = 100
        self.hazard_protection = 100
        self.units = 0
        self.nanites = 0
        self.quicksilver = 0
        self.discovered_species = set()
        self.current_planet = None
        self.in_space = False
        self.in_ship = False
        self.discovered_planets = set()  # Add this line
        
        # Action mapping
        self.actions = {
            0: self.move_forward,
            1: self.move_left,
            2: self.move_backward,
            3: self.move_right,
            4: self.interact,
            5: self.scan,
            6: self.craft,
            7: self.mine,
            8: self.use_secondary,
            9: self.open_inventory,
            10: self.open_discovery_menu,
            11: self.pause_menu
        }
        
        # Initialize game window location
        self.game_window = None
        self.locate_game_window()
        
        logging.info("NoMansSkyEnv initialized")
    
    def step(self, action):
        # Execute the chosen action
        self.actions[action]()
        
        # Update game state
        self.update_game_state()
        
        # Get new observation
        observation = self.get_observation()
        
        # Calculate reward
        reward = self.reward_function.calculate_reward(self)
        
        # Check if episode is done
        done = self.check_done()
        
        # Compile info dictionary
        info = self.get_info()
        
        return observation, reward, done, info
    
    def reset(self):
        # Generate a new seed or use a predefined one
        self.current_seed = self.seed_manager.get_new_seed()
        
        # Reset game state
        self.reset_game_state()
        
        # Return initial observation
        return self.get_observation()
    
    def render(self, mode='human'):
        if mode == 'human':
            cv2.imshow("No Man's Sky AI", self.get_observation())
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.get_observation()
    
    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        screen = capture_screen(region=self.game_window)
        return process_image(screen)
    
    def update_game_state(self):
        screen = capture_screen(region=self.game_window)
        
        # Update inventory
        self.inventory = self.detect_inventory(screen)
        
        # Update health, shields, oxygen, hazard protection
        self.health, self.shields, self.oxygen, self.hazard_protection = self.detect_status_bars(screen)
        
        # Update units, nanites, quicksilver
        self.units, self.nanites, self.quicksilver = self.detect_currencies(screen)
        
        # Check current location (planet, space, ship)
        self.current_planet, self.in_space, self.in_ship = self.detect_location(screen)
        
        # Update discoveries
        new_discoveries = self.detect_discoveries(screen)
        self.discovered_species.update(new_discoveries)
        
        # Update position (this would require more sophisticated tracking)
        self.update_position()
        
        logging.debug(f"Game state updated: Health={self.health}, Oxygen={self.oxygen}, Planet={self.current_planet}")
    
    def check_done(self):
        return self.health <= 0 or self.oxygen <= 0
    
    def get_info(self):
        return {
            "position": self.position,
            "inventory": self.inventory,
            "explored_areas": len(self.explored_areas),
            "health": self.health,
            "shields": self.shields,
            "oxygen": self.oxygen,
            "hazard_protection": self.hazard_protection,
            "units": self.units,
            "nanites": self.nanites,
            "quicksilver": self.quicksilver,
            "discovered_species": len(self.discovered_species),
            "current_planet": self.current_planet,
            "in_space": self.in_space,
            "in_ship": self.in_ship
        }
    
    def reset_game_state(self):
        # Navigate to the game's main menu
        self.pause_menu()
        time.sleep(1)
        pyautogui.press('down', presses=3)
        pyautogui.press('enter')
        time.sleep(1)
        
        # Start a new game with the current seed
        pyautogui.write(self.current_seed)
        pyautogui.press('enter')
        time.sleep(10)  # Wait for game to load
        
        # Reset all state variables
        self.inventory = {}
        self.position = (0, 0, 0)
        self.explored_areas = set()
        self.health = 100
        self.shields = 100
        self.oxygen = 100
        self.hazard_protection = 100
        self.units = 0
        self.nanites = 0
        self.quicksilver = 0
        self.discovered_species = set()
        self.current_planet = None
        self.in_space = False
        self.in_ship = False
        
        logging.info("Game state reset with new seed")
    
    def locate_game_window(self):
        # Try to find the No Man's Sky window
        windows = pyautogui.getWindowsWithTitle("No Man's Sky")
        if windows:
            window = windows[0]
            self.game_window = (window.left, window.top, window.width, window.height)
        else:
            raise Exception("Could not locate No Man's Sky game window")
    
    # Action methods
    def move_forward(self):
        pyautogui.keyDown('w')
        time.sleep(0.1)
        pyautogui.keyUp('w')
        self.position = (self.position[0], self.position[1] + 1, self.position[2])
        self.explored_areas.add(self.position)
    
    def move_left(self):
        pyautogui.keyDown('a')
        time.sleep(0.1)
        pyautogui.keyUp('a')
        self.position = (self.position[0] - 1, self.position[1], self.position[2])
        self.explored_areas.add(self.position)
    
    def move_backward(self):
        pyautogui.keyDown('s')
        time.sleep(0.1)
        pyautogui.keyUp('s')
        self.position = (self.position[0], self.position[1] - 1, self.position[2])
        self.explored_areas.add(self.position)
    
    def move_right(self):
        pyautogui.keyDown('d')
        time.sleep(0.1)
        pyautogui.keyUp('d')
        self.position = (self.position[0] + 1, self.position[1], self.position[2])
        self.explored_areas.add(self.position)
    
    def interact(self):
        pyautogui.press('e')
        time.sleep(0.1)
    
    def scan(self):
        pyautogui.press('f')
        time.sleep(0.1)
    
    def craft(self):
        pyautogui.press('x')
        time.sleep(0.1)
    
    def mine(self):
        pyautogui.mouseDown(button='left')
        time.sleep(0.5)
        pyautogui.mouseUp(button='left')
    
    def use_secondary(self):
        pyautogui.mouseDown(button='right')
        time.sleep(0.1)
        pyautogui.mouseUp(button='right')
    
    def open_inventory(self):
        pyautogui.press('i')
        time.sleep(0.1)
    
    def open_discovery_menu(self):
        pyautogui.press('tab')
        time.sleep(0.1)
    
    def pause_menu(self):
        pyautogui.press('esc')
        time.sleep(0.1)
    
    # Helper methods for updating game state
    def detect_inventory(self, screen):
        # Use image processing to detect inventory items
        # This is a placeholder and would need to be implemented based on the game's UI
        return {}
    
    def detect_status_bars(self, screen):
        # Use image processing to detect health, shields, oxygen, and hazard protection
        # This is a placeholder and would need to be implemented based on the game's UI
        return 100, 100, 100, 100
    
    def detect_currencies(self, screen):
        # Use OCR to detect units, nanites, and quicksilver
        return detect_text_on_screen(screen, ['units', 'nanites', 'quicksilver'])
    
    def detect_location(self, screen):
        # Use image processing to determine current location
        # This is a placeholder and would need to be implemented based on the game's UI
        return None, False, False
    
    def detect_discoveries(self, screen):
        # Use image processing to detect new discoveries
        # This is a placeholder and would need to be implemented based on the game's UI
        return set()
    
    def update_position(self):
        # This would require more sophisticated tracking, possibly using in-game coordinates
        # For now, we'll use a simple placeholder
        pass