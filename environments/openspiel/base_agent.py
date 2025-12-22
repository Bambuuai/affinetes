"""Base Game Agent for OpenSpiel games"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pyspiel


class BaseGameAgent(ABC):
    """
    Base class for game-specific agents.
    
    Each game implements an Agent subclass that encapsulates:
    - State formatting
    - Rule descriptions
    - Prompt generation
    - Parameter configuration
    """
    
    @property
    @abstractmethod
    def game_name(self) -> str:
        """Game name in OpenSpiel"""
        pass
    
    @abstractmethod
    def get_rules(self) -> str:
        """
        Return game rules text
        
        Returns:
            Complete rule description text
        """
        pass
    
    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """
        Convert OpenSpiel state to LLM-friendly description
        
        DEFAULT IMPLEMENTATION: Use observation_string as fallback.
        Most games can use this default. Override only if needed for better formatting.
        
        Args:
            state: OpenSpiel game state
            player_id: Current player ID
            
        Returns:
            Human-readable state description
        """
        return state.observation_string(player_id)
    
    @abstractmethod
    def generate_params(self, config_id: int) -> Dict[str, Any]:
        """
        Generate game parameters based on config_id
        
        Args:
            config_id: Configuration variant ID (0-99999999)
            
        Returns:
            Game parameter dictionary
        """
        pass
    
    def format_action_history(
        self,
        action_history: List[Tuple[int, int]],
        state: pyspiel.State
    ) -> str:
        """
        Format action history (default implementation)
        
        Args:
            action_history: List of (player_id, action) tuples
            state: Current game state
            
        Returns:
            Formatted action history
        """
        if not action_history:
            return "No actions taken yet."
        
        lines = []
        for player_id, action in action_history:
            action_str = state.action_to_string(player_id, action)
            lines.append(f"Player {player_id}: {action_str}")
        
        return "\n".join(lines)
    
    def generate_system_prompt(self) -> str:
        """
        Generate system prompt (called once per game)
        
        Includes: game name, game rules, output format requirements
        
        Returns:
            System prompt text
        """
        rules = self.get_rules()
        
        parts = [
            f"You are playing {self.game_name}.",
        ]
        
        if rules:
            parts.append(f"\n# Game Rules\n{rules}\n")
        
        parts.extend([
            "\n# Output Format",
            "You must respond with ONLY the action ID (a single number).",
            "Do NOT include descriptions or explanations.",
            "\nExamples:",
            '- For action "0 -> roll": respond "0"',
            '- For action "89 -> a3": respond "89"',
        ])
        
        return "\n".join(parts)
    
    def generate_user_prompt(
        self,
        state: pyspiel.State,
        player_id: int,
        action_history: List[Tuple[int, int]]
    ) -> str:
        """
        Generate user prompt (called each turn)
        
        Includes: current state, action history, legal actions
        
        Args:
            state: Game state
            player_id: Player ID
            action_history: Action history
            
        Returns:
            User prompt text
        """
        # 1. Format state
        state_desc = self.format_state(state, player_id)
        
        # 2. Action history (optional, can be omitted if state includes it)
        history_desc = self.format_action_history(action_history, state)
        
        # 3. Legal actions
        legal_actions = state.legal_actions(player_id)
        actions_desc = [
            f"{action} -> {state.action_to_string(player_id, action)}"
            for action in legal_actions
        ]
        
        # 4. Build prompt
        prompt_parts = [
            f"Current State:\n{state_desc}\n",
        ]
        
        if history_desc and history_desc != "No actions taken yet.":
            prompt_parts.append(f"\nAction History:\n{history_desc}\n")
        
        prompt_parts.extend([
            f"\nYou are Player {player_id}.\n",
            f"Legal Actions:\n" + "\n".join(actions_desc) + "\n\n",
            "Your choice (ID only):"
        ])
        
        return "".join(prompt_parts)