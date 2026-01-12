"""API Bot for OpenSpiel games using OpenAI-compatible API"""

import pyspiel
import numpy as np
import requests
import re
from typing import Optional, Dict, List, Any

from base_agent import BaseGameAgent


class APIBot(pyspiel.Bot):
    """
    Bot that uses OpenAI-compatible API for game decisions.
    
    Sends only system message + last user message to save cost.
    Records full conversation including reasoning from API response.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        agent: BaseGameAgent,
        player_id: int = 0,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        api_key: str = "",
        model: str = "gpt-5-mini",
        # temperature: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Initialize API Bot
        
        Args:
            game: pyspiel.Game instance
            agent: BaseGameAgent for game-specific formatting
            player_id: Player ID (0, 1, ...)
            api_url: OpenAI-compatible API endpoint
            api_key: API key for authentication
            model: Model name to use
            temperature: Sampling temperature
            seed: Random seed (for fallback random actions)
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._agent = agent
        self._api_url = api_url
        self._api_key = api_key
        self._model = model
        # self._temperature = temperature
        self._seed = seed or 42
        self._rng = np.random.RandomState(self._seed)
        
        # Conversation history (full history for training data)
        self._conversation: List[Dict[str, str]] = []
        self._action_history: List[Dict[str, Any]] = []
        self._system_prompt: Optional[str] = None
        self._system_prompt_generated = False
        self._observation: Optional[str] = None

    def restart_at(self, state):
        """Reset for new game"""
        self._conversation.clear()
        self._action_history.clear()
        self._system_prompt = None
        self._system_prompt_generated = False
        self._observation = None

    def inform_action(self, state, player_id, action):
        """Record all players' actions"""
        try:
            action_str = state.action_to_string(player_id, action)
        except:
            action_str = str(action)

        # Mark if this is my action or opponent's action
        is_me = player_id == self._player_id
        self._action_history.append({
            "player_id": int(player_id),
            "action": int(action),
            "action_str": action_str,
            "is_me": is_me,
        })

        try:
            self._observation = state.observation_string()
        except:
            try:
                self._observation = str(state)
            except:
                self._observation = None

    def _format_action_history(self) -> str:
        """Format action history for inclusion in API prompt"""
        if not self._action_history:
            return ""

        lines = ["", "=== Action History ==="]
        
        turn = 0
        for i, action in enumerate(self._action_history):
            if action['player_id'] == 0 or action['player_id'] == 1:
                turn += 1
                player_label = "You" if action["is_me"] else "Opponent"
                lines.append(f"Turn {turn}: {player_label} played: {action['action_str']} (action_id: {action['action']})")
        lines.append("======================")

        return "\n".join(lines)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI-compatible API with system + user message only"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        
        # Only send system message + last user message
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self._model,
            "messages": messages,
            # "temperature": self._temperature,
            # "reasoning_effort":  "high"
        }
        
        response = requests.post(self._api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _parse_response(self, response: str, legal_actions: List[int]) -> tuple[str, str]:
        """
        Parse API response to extract reasoning and action.
        
        Expected format:
        <reasoning>Your reasoning here</reasoning>
        <action>ACTION_ID</action>
        
        Returns:
            Tuple of (reasoning_content, action_id)
        """
        reasoning_content = ""

        # Try to extract action from <action> tags
        action_match = re.search(r'<action>\s*(\d+)\s*</action>', response, re.IGNORECASE)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
        
        if action_match:
            reasoning_content = reasoning_match.group(1) if reasoning_match else ""
            action = int(action_match.group(1))
            if action in legal_actions:
                return reasoning_content, str(action)
        
        # Fallback: try to find any number that's a legal action
        numbers = re.findall(r'\b(\d+)\b', response)
        for num_str in reversed(numbers):  # Check from end first
            num = int(num_str)
            if num in legal_actions:
                return reasoning_content, str(num)
        
        # Final fallback: random action
        action = self._rng.choice(legal_actions)
        return reasoning_content, str(action)

    def step(self, state) -> int:
        """Get action from API and record in conversation format."""
        # Generate system prompt (first time only)
        if not self._system_prompt_generated:
            self._system_prompt = self._agent.generate_system_prompt()
            self._conversation.append({"role": "system", "content": self._system_prompt})
            # Add instruction for response format (after system prompt)
            self._system_prompt += "\n\nIMPORTANT: Format your response as:\n"
            self._system_prompt += "<reasoning>Your step-by-step reasoning</reasoning>\n"
            self._system_prompt += "<action>ACTION_ID</action>"
            self._system_prompt_generated = True

        # Get legal actions
        legal_actions = state.legal_actions(self._player_id)

        # Generate user prompt
        user_prompt = self._agent.generate_user_prompt(
            state=state,
            player_id=self._player_id,
            legal_actions=legal_actions
        )
        # Record original user prompt in conversation history
        self._conversation.append({"role": "user", "content": user_prompt})

        # Add action history to user prompt for API call (not stored in conversation)
        action_history_str = self._format_action_history()
        user_prompt_with_history = user_prompt + action_history_str

        # Call API (only system + current user message with action history)
        try:
            response = self._call_api(self._system_prompt, user_prompt_with_history)
            reasoning_content, action_str = self._parse_response(response, legal_actions)
        except Exception:
            # On API error: content is "ERROR", use random action
            action = self._rng.choice(legal_actions)
            action_str = str(action)
            reasoning_content = "ERROR"

        # Record reasoning_content as assistant message
        self._conversation.append({"role": "assistant", "content": action_str, "reasoning_content": reasoning_content})

        # Record in action history
        self.inform_action(state, self._player_id, int(action_str))

        return int(action_str)

    def get_conversation(self) -> List[Dict[str, str]]:
        """Return conversation history in LLM format"""
        return self._conversation.copy()

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Return action history"""
        return self._action_history.copy()

    def get_observation(self) -> Optional[str]:
        """Return last observation"""
        return self._observation

