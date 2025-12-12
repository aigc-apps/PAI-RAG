
"""Custom SentenceSplitter that uses estimate_tokens_in_text for token counting."""
from llama_index.core.node_parser import SentenceSplitter 
from pairag.file.utils.tokenization import estimate_tokens_in_text


class MySentenceSplitter(SentenceSplitter):
    """Custom SentenceSplitter that uses estimate_tokens_in_text for token counting.
    
    This class extends llama_index's SentenceSplitter to use a custom token estimation
    function instead of the default tokenizer-based counting.
    """
    
    def _token_size(self, text: str) -> int:
        """Calculate token size using estimate_tokens_in_text.
        
        Args:
            text: The text to calculate token size for.
            
        Returns:
            The number of tokens in the text.
        """
        return estimate_tokens_in_text(text)
