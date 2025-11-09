#!/usr/bin/env python3
"""
TextStat analysis of the canonical health economic analysis code.

This script analyzes the text complexity and readability of the codebase.
"""

import os
import glob
import textstat
from typing import Dict, List


def analyze_file(filepath: str) -> Dict:
    """Analyze a single file with textstat metrics."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return {}
        
        # Skip binary files
        if not all(ord(c) < 128 for c in content[:1000]):  # Check first 1000 chars
            return {}
        
        # Calculate text statistics
        stats = {
            'filename': filepath,
            'character_count': len(content),
            'syllable_count': textstat.syllable_count(content),
            'lexicon_count': textstat.lexicon_count(content),
            'sentence_count': textstat.sentence_count(content),
            'flesch_reading_ease': textstat.flesch_reading_ease(content),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(content),
            'smog_index': textstat.smog_index(content),
            'coleman_liau_index': textstat.coleman_liau_index(content),
            'automated_readability_index': textstat.automated_readability_index(content),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(content),
            'difficult_words': textstat.difficult_words(content),
            'linsear_write_formula': textstat.linsear_write_formula(content),
            'gunning_fog': textstat.gunning_fog(content)
        }
        
        return stats
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return {}


def analyze_directory(directory: str, extensions: List[str] = ['.py', '.md', '.txt']) -> List[Dict]:
    """Analyze all files in a directory."""
    all_stats = []
    
    for ext in extensions:
        files = glob.glob(f"{directory}/**/*{ext}", recursive=True)
        for file in files:
            stats = analyze_file(file)
            if stats:
                all_stats.append(stats)
    
    return all_stats


def print_summary(stats_list: List[Dict]):
    """Print a summary of the textstat analysis."""
    if not stats_list:
        print("No files analyzed.")
        return
    
    print("\n" + "="*80)
    print("TEXTSTAT ANALYSIS SUMMARY")
    print("="*80)
    
    # Calculate averages
    avg_flesch = sum(s['flesch_reading_ease'] for s in stats_list if s.get('flesch_reading_ease')) / len(stats_list)
    avg_grade = sum(s['flesch_kincaid_grade'] for s in stats_list if s.get('flesch_kincaid_grade')) / len(stats_list)
    avg_difficult_words = sum(s['difficult_words'] for s in stats_list if s.get('difficult_words')) / len(stats_list)
    avg_gunning_fog = sum(s['gunning_fog'] for s in stats_list if s.get('gunning_fog')) / len(stats_list)
    
    print(f"Total files analyzed: {len(stats_list)}")
    print(f"Average Flesch Reading Ease: {avg_flesch:.2f}")
    print(f"Average Flesch-Kincaid Grade Level: {avg_grade:.2f}")
    print(f"Average Difficult Words per File: {avg_difficult_words:.2f}")
    print(f"Average Gunning Fog Index: {avg_gunning_fog:.2f}")
    
    print("\nFile-by-file breakdown:")
    print("-" * 80)
    print(f"{'File':<40} {'Flesch':<8} {'Grade':<8} {'Gunning':<8} {'Complexity':<10}")
    print("-" * 80)
    
    for stat in stats_list:
        filename = os.path.basename(stat.get('filename', 'unknown'))[:39]
        flesch = stat.get('flesch_reading_ease', 0)
        grade = stat.get('flesch_kincaid_grade', 0)
        gunning = stat.get('gunning_fog', 0)
        
        # Simple complexity classification
        if grade < 8:
            complexity = "Low"
        elif grade < 12:
            complexity = "Medium"
        else:
            complexity = "High"
            
        print(f"{filename:<40} {flesch:<8.1f} {grade:<8.1f} {gunning:<8.1f} {complexity:<10}")
    
    print("\nReadability Interpretation:")
    print("-" * 40)
    print("Flesch Reading Ease: ")
    print("  90-100: Very easy to read")
    print("  60-70: Standard (13-15 years old)")
    print("  0-30: Very difficult")
    print("\nFlesch-Kincaid Grade Level:")
    print("  Grade 8: Easy to read")
    print("  Grade 12: High school graduate")
    print("  Grade 16+: University level")
    print("\nGunning Fog Index:")
    print("  Below 6: Easy to read")
    print("  7-8: Fairly easy to read")
    print("  9-10: Plain English")
    print("  11-12: Fairly difficult")
    print("  13-15: Difficult")
    print("  16+: Very difficult")


def main():
    """Main function to run textstat analysis."""
    project_dir = "/Users/doughnut/Library/CloudStorage/OneDrive-VictoriaUniversityofWellington-STAFF/Submitted/policy_societaldam_pharma/canonical_code"
    
    print(f"Analyzing text complexity of files in: {project_dir}")
    
    # Analyze Python, Markdown, and text files
    stats = analyze_directory(project_dir, ['.py', '.md', '.txt'])
    
    print_summary(stats)


if __name__ == '__main__':
    main()