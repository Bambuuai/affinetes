"""Blackjack Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class BlackjackAgent(BaseGameAgent):
    """Blackjack Game Agent - Uses default observation_string formatting"""
    
    @property
    def game_name(self) -> str:
        return "blackjack"
    
    def get_rules(self) -> str:
        return """BLACKJACK RULES:
Setup: 52-card deck. Player vs Dealer. Goal: Get card total closer to 21 than dealer without going over.
Card values: 2-10 = face value, J/Q/K = 10, A = 1 or 11 (player's choice).

Player turn:
- Hit: Take another card
- Stand: Keep current total and end turn

Dealer turn: Must hit on 16 or less, stand on 17 or more.

Winning: Beat dealer's total without exceeding 21. If you exceed 21, you bust and lose immediately."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Blackjack parameter generation
        """
        return {}
