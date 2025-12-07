import re
import textstat

def strip_latex(text):
    # Remove comments
    text = re.sub(r'%.*', '', text)
    # Remove commands but keep content where appropriate
    text = re.sub(r'\\section\*?\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\emph\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\pkg\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\code\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\cite\{([^}]+)\}', '', text)
    text = re.sub(r'\\citep\{([^}]+)\}', '', text)
    text = re.sub(r'\\citet\{([^}]+)\}', '', text)
    # Remove other commands
    text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{([^}]*)\})?', '', text)
    return text

def analyze_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    clean_text = strip_latex(content)
    
    print(f"--- Readability Analysis for {filepath} ---")
    print(f"Flesch Reading Ease: {textstat.flesch_reading_ease(clean_text)}")
    print(f"Flesch-Kincaid Grade: {textstat.flesch_kincaid_grade(clean_text)}")
    print(f"Gunning Fog: {textstat.gunning_fog(clean_text)}")
    print(f"Smog Index: {textstat.smog_index(clean_text)}")
    print("-" * 40)

if __name__ == "__main__":
    analyze_file("manuscript/jss_submission.tex")
