#!/usr/bin/env python3
"""
TextStat analysis of a specified file.

This script analyzes the text complexity and readability of a given file.
"""

import os
import sys
from typing import Dict

import textstat


def analyze_file(filepath: str) -> Dict:
    """Analyze a single file with textstat metrics."""
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip():
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


def print_summary(stats: Dict):
    """Print a summary of the textstat analysis."""
    if not stats:
        print("No file analyzed.")
        return

    print("\n" + "="*80)
    print(f"TEXTSTAT ANALYSIS SUMMARY FOR: {os.path.basename(stats.get('filename', 'unknown'))}")
    print("="*80)

    flesch = stats.get('flesch_reading_ease', 0)
    grade = stats.get('flesch_kincaid_grade', 0)
    gunning = stats.get('gunning_fog', 0)

    print(f"Flesch Reading Ease: {flesch:.2f}")
    print(f"Flesch-Kincaid Grade Level: {grade:.2f}")
    print(f"Gunning Fog Index: {gunning:.2f}")

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
    if len(sys.argv) != 2:
        print("Usage: python run_textstat_analysis.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    print(f"Analyzing text complexity of file: {file_path}")

    stats = analyze_file(file_path)

    print_summary(stats)


if __name__ == '__main__':
    main()
