"""
Dyck Language 2 Verifier

Verifies reverse bracket completion answers.
"""

from base.data import Data
from base.verifier import Verifier


class DyckLanguage2Verifier(Verifier):
    """Verify reverse Dyck language completion answers"""

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify if the solution is correct and minimal for reverse completion

        Args:
            data: Game data containing suffix sequence
            test_solution: Model's response

        Returns:
            bool: True if correct and minimal, False otherwise
        """
        from .generator import DyckLanguage2Generator

        generator = DyckLanguage2Generator()
        user_full_sequence = generator.extract_answer(test_solution)

        suffix_sequence = data.metadata.get("suffix_sequence", "")

        if not suffix_sequence or not user_full_sequence:
            return False

        # 1. Check if user's answer is valid Dyck language
        if not self._is_valid_dyck(user_full_sequence):
            return False

        # 2. Check if user's answer ends with the suffix sequence
        if not user_full_sequence.endswith(suffix_sequence):
            return False

        # 3. Extract the added part (should be only opening brackets)
        user_added_part = user_full_sequence[:-len(suffix_sequence)] if suffix_sequence else user_full_sequence

        # 4. Check if added part matches required opening brackets
        required_opening = self._calculate_required_opening(suffix_sequence)

        return user_added_part == required_opening

    def _is_valid_dyck(self, sequence: str) -> bool:
        """Check if a sequence is valid Dyck language"""
        stack = []
        bracket_pairs = {
            ")": "(", "]": "[", "}": "{", ">": "<",
            "⟩": "⟨", "⟧": "⟦", "⦄": "⦃", "⦆": "⦅"
        }

        for char in sequence:
            if char in "([{<⟨⟦⦃⦅":
                stack.append(char)
            elif char in ")]}>⟩⟧⦄⦆":
                if not stack or stack[-1] != bracket_pairs[char]:
                    return False
                stack.pop()

        return len(stack) == 0

    def _calculate_required_opening(self, suffix: str) -> str:
        """
        Calculate minimal opening brackets needed to prepend to suffix.

        Process suffix from left to right:
        - If closing bracket has no matching opener in stack, we need to prepend one
        - Opening brackets go on stack for potential matching
        """
        bracket_pairs = {
            ")": "(", "]": "[", "}": "{", ">": "<",
            "⟩": "⟨", "⟧": "⟦", "⦄": "⦃", "⦆": "⦅"
        }

        stack = []  # Stack of opening brackets in suffix
        needed = []  # Opening brackets we need to prepend

        for char in suffix:
            if char in "([{<⟨⟦⦃⦅":
                # Opening bracket - push to stack
                stack.append(char)
            elif char in ")]}>⟩⟧⦄⦆":
                expected_opener = bracket_pairs[char]
                if stack and stack[-1] == expected_opener:
                    # Matched with an opener in suffix
                    stack.pop()
                else:
                    # No matching opener in suffix, need to prepend one
                    needed.append(expected_opener)

        # needed is in the order they're encountered (left to right in suffix)
        # But we need to prepend them, so reverse the order
        # Actually no - they should be prepended in reverse order of encounter
        # because first unmatched closer needs innermost opener
        return "".join(reversed(needed))
