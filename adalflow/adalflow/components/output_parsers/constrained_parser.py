"""Constrained Selection Parser for forcing models to select from specific options.

This parser uses logprobs to force models to select from a predefined set of options,
similar to the Guidance library's constrained generation capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Union

from adalflow.core.component import DataComponent
from adalflow.core.types import TokenLogProb

log = logging.getLogger(__name__)


class ConstrainedSelectionParser(DataComponent):
    """Parser that forces model to select from a constrained set of options using logprobs.
    
    This parser is particularly useful for classification tasks where you want to ensure
    the model only outputs one of the predefined options.
    
    Example:
        parser = ConstrainedSelectionParser(
            options=["positive", "negative", "neutral"],
            allow_partial_match=True
        )
        
        # Use with Generator
        generator = Generator(
            model_client=model_client,
            output_processors=parser,
            # ... other args
        )
    """
    
    def __init__(
        self,
        options: List[str],
        allow_partial_match: bool = True,
        case_sensitive: bool = False,
        max_tokens: int = 10,
        temperature: float = 0.0,
    ):
        """Initialize the constrained selection parser.
        
        Args:
            options: List of valid options the model can select from
            allow_partial_match: Whether to allow partial matches (e.g., "pos" matches "positive")
            case_sensitive: Whether option matching is case sensitive
            max_tokens: Maximum number of tokens to consider for selection
            temperature: Temperature for logprob-based selection (0.0 = deterministic)
        """
        super().__init__()
        
        if not options or len(options) == 0:
            raise ValueError("Options list cannot be empty")
        
        self.options = options
        self.allow_partial_match = allow_partial_match
        self.case_sensitive = case_sensitive
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Normalize options for matching
        if not case_sensitive:
            self.normalized_options = {opt.lower(): opt for opt in options}
        else:
            self.normalized_options = {opt: opt for opt in options}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return text if self.case_sensitive else text.lower()
    
    def _find_best_match(self, text: str) -> Optional[str]:
        """Find the best matching option for the given text."""
        normalized_text = self._normalize_text(text.strip())
        
        # Exact match first
        if normalized_text in self.normalized_options:
            return self.normalized_options[normalized_text]
        
        if not self.allow_partial_match:
            return None
        
        # Partial match
        for normalized_option, original_option in self.normalized_options.items():
            if normalized_option in normalized_text or normalized_text in normalized_option:
                return original_option
        
        return None
    
    def _select_from_logprobs(self, logprobs: List[List[TokenLogProb]]) -> Optional[str]:
        """Select the best option using logprobs."""
        if not logprobs or len(logprobs) == 0:
            return None
        
        # Flatten all logprobs
        all_tokens = []
        for token_list in logprobs:
            all_tokens.extend(token_list)
        
        if not all_tokens:
            return None
        
        # Calculate scores for each option
        option_scores = {}
        
        for option in self.options:
            normalized_option = self._normalize_text(option)
            option_words = normalized_option.split()
            
            # score based on token probabilities
            score = 0.0
            matched_tokens = 0
            
            for i, token in enumerate(all_tokens[:self.max_tokens]):
                token_text = self._normalize_text(token.token)
                
                for word in option_words:
                    if word in token_text or token_text in word:
                        # Use logprob as score (higher is better)
                        score += token.logprob
                        matched_tokens += 1
                        break
            
            if matched_tokens > 0:
                # normalize score by number of matched tokens
                option_scores[option] = score / matched_tokens
        
        if not option_scores:
            return None
        
        best_option = max(option_scores.items(), key=lambda x: x[1])[0]
        return best_option
    
    def call(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Parse input and return the best matching option.
        
        Args:
            input_data: Either a string response or a dict with 'response' and 'logprobs' keys
            
        Returns:
            The best matching option from the predefined list
        """
        if isinstance(input_data, str):
            # Simple text-based matching
            result = self._find_best_match(input_data)
            if result is None:
                log.warning(f"No matching option found for: {input_data}")
                return self.options[0]  # Return first option as fallback
            return result
        
        elif isinstance(input_data, dict):
            # Use logprobs if available
            if 'logprobs' in input_data and input_data['logprobs']:
                result = self._select_from_logprobs(input_data['logprobs'])
                if result is not None:
                    return result
            
            # Fallback to text matching
            response_text = input_data.get('response', '')
            if response_text:
                result = self._find_best_match(response_text)
                if result is not None:
                    return result
            
            log.warning(f"No matching option found in: {input_data}")
            return self.options[0]
        
        else:
            log.warning(f"Unsupported input type: {type(input_data)}")
            return self.options[0]
    
    def get_format_instructions(self) -> str:
        """Get format instructions for the prompt."""
        options_str = ", ".join([f'"{opt}"' for opt in self.options])
        
        return f"""You must respond with exactly one of these options: {options_str}

Rules:
- Choose the option that best matches your analysis
- Respond with only the option text, no additional explanation
- If unsure, choose the most appropriate option from the list
- Your response must be one of: {options_str}"""
    
    def _extra_repr(self) -> str:
        return f"options={self.options}, allow_partial_match={self.allow_partial_match}, case_sensitive={self.case_sensitive}"


class MultiChoiceParser(ConstrainedSelectionParser):
    """Parser for multiple choice questions with options A, B, C, D, etc."""
    
    def __init__(
        self,
        num_choices: int = 4,
        choice_format: str = "letter",  # "letter" or "number"
        **kwargs
    ):
        """Initialize multi-choice parser.
        
        Args:
            num_choices: Number of choices (default 4 for A, B, C, D)
            choice_format: Format of choices - "letter" (A, B, C, D) or "number" (1, 2, 3, 4)
        """
        if choice_format == "letter":
            options = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D, ...
        elif choice_format == "number":
            options = [str(i + 1) for i in range(num_choices)]  # 1, 2, 3, 4, ...
        else:
            raise ValueError("choice_format must be 'letter' or 'number'")
        
        super().__init__(options=options, **kwargs)
        self.num_choices = num_choices
        self.choice_format = choice_format
    
    def get_format_instructions(self) -> str:
        """Get format instructions for multiple choice."""
        if self.choice_format == "letter":
            choices = [chr(65 + i) for i in range(self.num_choices)]
            return f"""You must respond with exactly one letter: {', '.join(choices)}

Rules:
- Choose the letter that corresponds to the best answer
- Respond with only the letter (A, B, C, D, etc.)
- No additional text or explanation
- Your response must be one of: {', '.join(choices)}"""
        else:
            choices = [str(i + 1) for i in range(self.num_choices)]
            return f"""You must respond with exactly one number: {', '.join(choices)}

Rules:
- Choose the number that corresponds to the best answer
- Respond with only the number (1, 2, 3, 4, etc.)
- No additional text or explanation
- Your response must be one of: {', '.join(choices)}"""
