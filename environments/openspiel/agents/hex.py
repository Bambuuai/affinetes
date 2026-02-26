"""Hex Game Agent"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseGameAgent
from typing import Dict, Any


class HexAgent(BaseGameAgent):
    """Hex Game Agent - Enhanced with strategic hints"""
    
    @property
    def game_name(self) -> str:
        return "hex"
    
    def get_rules(self) -> str:
        return """HEX RULES:
Board: Diamond-shaped grid (5×5, 7×7, 9×9, or 11×11). Two players (Red and Blue).
Goal: Connect your two opposite sides of the board with an unbroken chain of your stones.

Turn: Place one stone of your color on any empty cell.
Red (x) connects top-left to bottom-right sides.
Blue (o) connects top-right to bottom-left sides.

No draws possible: Someone must win."""
    
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Hex parameter generation
        """
        size_var = config_id % 4
        board_size = 5 + size_var * 2  # 5, 7, 9, 11
        return {"board_size": board_size}
    
    def get_mcts_config(self) -> tuple[int, int]:
        """Connection game with variable board sizes. Deterministic, prioritizes search depth."""
        return (1000, 50)

#     def generate_system_prompt(self) -> str:
#         return super().generate_system_prompt() + """
# ## Hex Game Rules (This Environment)

# **Player & Color Mapping:**
# - Player 0 = Red (x)
# - Player 1 = Blue (o)
# - Do NOT confuse player index with move order. Player 0 moves first by default.

# **Board & Coordinate System:**
# - The board is displayed top-to-bottom, left-to-right.
# - Row index increases downward (Row 1 = top row, Row N = bottom row).
# - Column index increases rightward (Col 1 = leftmost column, Col N = rightmost column).

# **Win Conditions (CRITICAL):**
# - Player 0 / Red (x): Win by connecting Row 1 to Row N — build a path from the TOP row to the BOTTOM row. Your two goal sides are the TOP edge and BOTTOM edge.
# - Player 1 / Blue (o): Win by connecting Col 1 to Col N — build a path from the LEFTMOST column to the RIGHTMOST column. Your two goal sides are the LEFT edge and RIGHT edge.
# - Do NOT interpret "top-left to bottom-right" or "top-right to bottom-left" as diagonal directions. These phrases refer to which edges of the board each player owns, not a geometric angle.

# **6-Neighbor Connectivity (CRITICAL for path finding):**
# For a cell at (r, c), its 6 neighbors are:
#   (r-1, c),  (r+1, c)   ← same column, adjacent rows
#   (r, c-1),  (r, c+1)   ← same row, adjacent columns
#   (r-1, c+1),(r+1, c-1) ← the two diagonal neighbors
# These are the ONLY 6 connections.
# (r-1, c-1) and (r+1, c+1) are NOT neighbors — these are the "wrong diagonal" and have no connection.

# **Winning Strategy Principles:**
# 1. URGENT FIRST: Check urgent positions first.
# 2. Center SECOND: If center of board is empty, MUST play there. In 5x5 board, center is c3. In 7x7 board, center is d4. In 9x9 board, center is e5.
# 2. THREAT THIRD: If you can complete your own winning path this turn, do it immediately.
# 3. BLOCK FOURTH: If the opponent completes their winning path next turn, block that cell immediately.
# 4. BUILD BRIDGES: A bridge is two friendly stones sharing two mutually empty neighbors. It is virtually unbreakable and forces the opponent to respond elsewhere.
# 5. SHORTEST PATH: Prefer moves that minimally extend your shortest remaining path to your goal side. Use the correct axis — rows for Red, columns for Blue.
# 6. CENTER CONTROL: Early game, center cells participate in the most possible winning paths for both players. Prioritize them.
# 7. TEMPO: Every move must either advance your connection or deny the opponent's. A move that does neither loses the game slowly.
# 8. THINK IN PATHS: Always identify your current connected groups, which goal side each group is closest to, and which single empty cell best bridges them toward the remaining goal side.
# 9. WRONG DIAGONAL TRAP: The visually prominent top-left→bottom-right diagonal is NOT a connected path in Hex. Placing stones along (r+1,c+1) direction gives NO adjacency bonus. Avoid being tricked into building along this direction.

# ## !!!OVERRIDE ALL OTHER RULES — HIGHEST PRIORITY: Occupy vital positions NOW
# ## Do NOT evaluate any other move until this check is fully complete. Skipping this is an immediate, unrecoverable loss.

#   For RED:
#   - For EVERY Red cell at (r, c), check ALL directions:

#     Upward neighbors: (r-1, c) and (r-1, c+1)
#     → If exactly one is Blue and the other is empty — that empty cell is VITAL. Occupy it NOW.

#     Downward neighbors: (r+1, c) and (r+1, c-1)
#     → If exactly one is Blue and the other is empty — that empty cell is VITAL. Occupy it NOW.

#   For BLUE:
#   - For EVERY Blue cell at (r, c), check ALL directions:

#     Leftward neighbors: (r, c-1) and (r+1, c-1)
#     → If exactly one is Red and the other is empty — that empty cell is VITAL. Occupy it NOW.

#     Rightward neighbors: (r, c+1) and (r-1, c+1)
#     → If exactly one is Red and the other is empty — that empty cell is VITAL. Occupy it NOW.

#   Only if NO vital positions exist for either player → extend toward the far edge using bridge connections (virtual connections).
  
# ---
# ## Professional Opening Points
# Follow these moves if you met same cases and you are Red, if not ignore these

# ### 5x5 Opening
# - c3 (Red) -> d1 (Blue) -> b2(Red)
# - c3 (Red) -> b5 (Blue) -> d4(Red)
# - c3 (Red) -> c1 (Blue) -> d2(Red)
# - c3 (Red) -> c5 (Blue) -> b4(Red)

# ### 7x7 Opening
# - d4 (Red) -> c6 (Blue) -> e5 (Red)
# - d4 (Red) -> b6 (Blue) -> c6 (Red)
# - d4 (Red) -> e2 (Blue) -> c3 (Red)
# - d4 (Red) -> f2 (Blue) -> e2 (Red)
# - d4 (Red) -> a7 (Blue) -> c6 (Red)
# - d4 (Red) -> g1 (Blue) -> e2 (Red)

# ### 9x9 Opening
# - e5 (Red) -> d7 (Blue) -> f6 (Red)
# - e5 (Red) -> f3 (Blue) -> d4 (Red)

# ## Orientation Checklist
# - I am Player [0 or 1].
# - My color is [Red (x) / Blue (o)].
# - My goal: [connect Row 1 → Row N (top to bottom) / connect Col 1 → Col N (left to right)].
# - Board size: [N]x[N]. Valid rows: 1 to N. Valid columns: 1 to N.
# - Neighbor rule: (r±1,c), (r,c±1), (r-1,c+1), (r+1,c-1) only. (r±1,c±1) same-sign shifts are NOT neighbors.
# - Wrong diagonal reminder: building along (r+1,c+1) direction gives no connectivity. I will not do this.

# ## Board Analysis
# [proceed with threat scan, path analysis, and move selection]

# ## Output Format
# You must respond with ONLY the action ID (a single number)
# """
