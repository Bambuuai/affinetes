"""Gin Rummy Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class GinRummyAgent(BaseGameAgent):
    """Gin Rummy Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "gin_rummy"
    
    def get_rules(self) -> str:
        return """GIN RUMMY RULES:
Setup: 52-card deck. Each player receives 7-10 cards (variant dependent). Remaining cards form draw pile.
Goal: Form melds (sets of 3+ same rank, or runs of 3+ consecutive cards in same suit) to minimize deadwood (unmelded cards).

Each turn:
1. Draw: Take top card from draw pile or discard pile
2. Discard: Place one card face-up on discard pile

Knocking: When deadwood ≤ knock_card value (8-10), you may knock to end the hand.
Gin: If all cards form melds (0 deadwood), declare "Gin" for bonus points.

Scoring: Winner scores difference in deadwood values. Gin earns 25-point bonus."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Gin Rummy parameter generation
        """
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return {
            "hand_size": 7 + hand_var,  # 7, 8, 9
            "knock_card": 10 - knock_var  # 10, 9, 8
        }
