import re
# from typing import Optional

class MinimalStreamlitSanitizer:
    """
    Minimal text sanitizer that ONLY fixes LaTeX rendering issues in Streamlit
    while preserving ALL original formatting (bold, italic, etc.)
    """
    
    def __init__(self):
        pass
    
    def escape_latex_math_delimiters(self, text: str) -> str:
        """
        Only escape dollar signs that would trigger LaTeX math mode in Streamlit.
        This is the minimal fix needed to prevent rendering issues.
        
        Args:
            text: Input text that may contain dollar signs
            
        Returns:
            Text with escaped dollar signs that won't trigger LaTeX
        """
        if not text or not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Strategy: Replace $ with a safe alternative that displays the same
        # We'll use a method that doesn't interfere with any other formatting
        
        # Method 1: Replace $ with unicode dollar sign (looks identical)
        # This completely avoids LaTeX interpretation
        text = text.replace('$', '💲')  # Using emoji dollar sign
        
        # Alternative Method 2: Use HTML entity (if you prefer)
        # text = text.replace('$', '&#36;')
        
        # Alternative Method 3: Use zero-width space to break LaTeX pattern
        # text = text.replace('$', '$\u200B')
        
        return text
    
    def restore_dollar_signs(self, text: str) -> str:
        """
        Restore dollar signs from safe replacement if needed.
        (Usually not needed since the replacement looks identical)
        """
        return text.replace('💲', '$')
    
    def safe_currency_display(self, text: str) -> str:
        """
        Convert currency symbols to safe display format that won't trigger LaTeX
        but maintains the same visual appearance.
        """
        if not text or not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Replace $ with visually identical alternatives that don't trigger LaTeX
        currency_replacements = [
            ('$', '＄'),  # Full-width dollar sign (looks almost identical)
            # Add other problematic symbols if needed
            # ('€', '€'),  # Euro usually doesn't cause issues
            # ('£', '£'),  # Pound usually doesn't cause issues
        ]
        
        for original, replacement in currency_replacements:
            text = text.replace(original, replacement)
        
        return text


class SmartStreamlitSanitizer:
    """
    Smart sanitizer that detects LaTeX patterns and only escapes what's necessary
    """
    
    def __init__(self):
        self.minimal_sanitizer = MinimalStreamlitSanitizer()
    
    def has_latex_issues(self, text: str) -> bool:
        """
        Detect if text contains patterns that would cause LaTeX rendering issues
        """
        if not text:
            return False
        
        # Check for dollar signs (main culprit)
        if '$' in text:
            return True
        
        # Check for other LaTeX patterns that might cause issues
        latex_patterns = [
            r'\$.*?\$',      # Text between dollar signs
            r'\$\$.*?\$\$',  # Text between double dollar signs
            r'\\[a-zA-Z]+',  # LaTeX commands
        ]
        
        for pattern in latex_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def minimal_escape(self, text: str) -> str:
        """
        Apply only the minimal escaping needed to prevent LaTeX issues
        while preserving ALL other formatting
        """
        if not self.has_latex_issues(text):
            return text  # No changes needed
        
        # Only fix the specific LaTeX issue
        return self.minimal_sanitizer.safe_currency_display(text)


# Convenience functions for easy integration
def fix_latex_issues(text: str) -> str:
    """
    Fix only LaTeX rendering issues while preserving all other formatting.
    This is the recommended function for most use cases.
    
    Args:
        text: Input text that may have LaTeX rendering issues
        
    Returns:
        Text with LaTeX issues fixed but all other formatting preserved
    """
    sanitizer = SmartStreamlitSanitizer()
    return sanitizer.minimal_escape(text)


def safe_insight_display(text: str) -> str:
    """
    Prepare insight text for safe display in Streamlit.
    Only fixes rendering issues, preserves all formatting.
    
    Args:
        text: LLM-generated insight text
        
    Returns:
        Text safe for Streamlit display with formatting preserved
    """
    return fix_latex_issues(text)


# Even simpler approach - just replace problematic characters
def quick_fix_dollars(text: str) -> str:
    """
    Ultra-simple fix: just replace $ with visually identical character
    """
    if not text:
        return text
    return text.replace('$', '＄')  # Full-width dollar sign


# Test and demonstration
if __name__ == "__main__":
    # Test cases showing that formatting is preserved
    test_cases = [
        # Original problematic text
        "The Sales department has the highest total budget at $210,000, while both the HR and Management departments have the lowest total budget at $60,000 each.",
        
        # Text with intentional formatting that should be preserved
        "**Important:** The IT department has *significantly* higher costs at $580,000 compared to other departments.",
        
        # Mixed formatting
        "The analysis shows:\n- **Sales**: $210,000 budget\n- *HR*: $60,000 budget\n- `IT`: $580,000 budget",
        
        # Code blocks and other markdown
        "Results summary: `SELECT SUM(budget) FROM departments` returned $930,000 total.",
        
        # Multiple currency amounts
        "Quarterly comparison: Q1 had $150,000, Q2 had $200,000, and Q3 reached $250,000.",
    ]
    
   
    
  