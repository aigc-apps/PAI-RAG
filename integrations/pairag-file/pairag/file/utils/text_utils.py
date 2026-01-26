import re

def replace_consecutive_spaces(text: str) -> str:
    if not text:
        return text

    # 1. Replace multiple spaces with one space
    text = re.sub(r' +', ' ', text)
    
    # 2. Replace multiple tabs with one tab
    text = re.sub(r'\t+', '\t', text)
        
    # 4. Collapse 3 or more newlines into exactly TWO (\n\n)
    # This preserves paragraph breaks but removes excessive vertical gaps
    text = re.sub(r'(\s*\n\s*){2,}', '\n\n', text)

    
    return text

