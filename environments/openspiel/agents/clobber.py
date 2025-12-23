"""Clobber Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class ClobberAgent(BaseGameAgent):
    """Clobber Game Agent - Enhanced with move format clarity"""
    
    @property
    def game_name(self) -> str:
        return "clobber"
    
    def get_rules(self) -> str:
        return """CLOBBER RULES:
Board: Rectangular grid (5×5, 6×6, or 7×7) filled with alternating black and white pieces.
Goal: Be the last player able to move.

Movement: On your turn, move one of your pieces orthogonally (horizontally or vertically) to capture an adjacent opponent piece. The captured piece is removed and replaced by your piece.
Must capture: Every move must capture an opponent piece. No non-capturing moves allowed.

Move Format: Moves are specified as "row col" (e.g., "2 3" means row 2, column 3).
Important: You can ONLY move to a position occupied by an opponent piece that is directly adjacent (up/down/left/right) to one of your pieces.

Losing: If you have no legal moves (no adjacent opponent pieces to capture), you lose.
Strategy: Force opponent into position with no captures available."""
    
    def format_state(self, state, player_id: int) -> str:
        """Enhanced state formatting with move clarity"""
        # Use fallback logic from base class
        try:
            base_obs = state.observation_string(player_id)
        except:
            try:
                base_obs = state.information_state_string(player_id)
            except:
                base_obs = str(state)
        
        # Add player role and move format reminder
        player_symbol = "Black (b)" if player_id == 0 else "White (w)"
        opponent_symbol = "White (w)" if player_id == 0 else "Black (b)"
        
        return f"""Current State:
{base_obs}

Your Pieces: {player_symbol}
Opponent Pieces: {opponent_symbol}

Remember: You must capture an opponent piece by moving one of your pieces to an adjacent opponent position (orthogonally only)."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Clobber parameter generation
        """
        size_var = config_id % 3
        board_size = 5 + size_var  # 5, 6, 7
        return {
            "rows": board_size,
            "columns": board_size
        }
