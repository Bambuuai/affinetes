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

SETUP:
- 52-card deck, each player receives 7-10 cards (variant dependent)
- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)

MELDS (Valid Combinations):
1. SET: 3+ cards of SAME RANK (e.g., 7♠ 7♥ 7♣)
2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5♦ 6♦ 7♦)
Examples:
- Valid runs: A♠-2♠-3♠, 9♥-10♥-J♥-Q♥, 10♣-J♣-Q♣-K♣
- Invalid: K♠-A♠-2♠ (Ace is LOW only, not wraparound)

EACH TURN:
1. DRAW: Pick from stock pile OR discard pile
2. DISCARD: Place ONE card face-up on discard pile

KNOCKING:
- When deadwood ≤ knock_card value (8-10), you MAY knock to end hand
- Gin: ALL cards form melds (0 deadwood) = 25-point bonus

SCORING: Winner scores difference in deadwood point values.
Card Values: A=1, 2-10=face value, J=11, Q=12, K=13"""
    
    def format_state(self, state, player_id: int) -> str:
        """Format Gin Rummy state"""
        return state.observation_string(player_id)
    
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
