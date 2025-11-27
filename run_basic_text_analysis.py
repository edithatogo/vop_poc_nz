#!/usr/bin/env python3
"""
Basic Text Complexity Analysis of the canonical health economic analysis code.

Since external libraries like textstat couldn't be installed in this environment,
this script performs basic text analysis using built-in Python functions.
"""

import glob
import os
import re
from typing import Dict, List, Optional


def count_syllables_simple(word: str) -> int:
    """Simple syllable counter using vowel groups."""
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel

    # Adjust for silent 'e' at the end
    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    return max(1, syllable_count)  # At least 1 syllable


def analyze_file_basic(filepath: str) -> Dict:
    """Basic analysis of a single file."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return {}

        # Count lines, words, characters
        lines = content.split("\n")
        line_count = len(lines)
        word_pattern = r"\b\w+\b"
        words = re.findall(word_pattern, content)
        word_count = len(words)
        char_count = len(content)
        char_count_no_spaces = len(
            content.replace(" ", "").replace("\n", "").replace("\t", "")
        )

        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Calculate average sentence length (using simple ., !, ? as sentence separators)
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Calculate average syllables per word
        syllable_count = sum(count_syllables_simple(word) for word in words)
        avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0

        # Estimate complexity based on average word length and syllables
        complexity_score = (avg_word_length * 0.5) + (avg_syllables_per_word * 1.5)

        stats = {
            "filename": filepath,
            "line_count": line_count,
            "word_count": word_count,
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "total_syllables": syllable_count,
            "complexity_score": round(complexity_score, 2),
        }

        return stats
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return {}


def analyze_directory(
    directory: str, extensions: Optional[List[str]] = None
) -> List[Dict]:
    """Analyze all files in a directory."""
    if extensions is None:
        extensions = [".py", ".md", ".txt"]
    all_stats = []

    for ext in extensions:
        files = glob.glob(f"{directory}/**/*{ext}", recursive=True)
        for file in files:
            # Skip files in the output directory to avoid recursion
            if "/output/" not in file and "run_textstat_analysis.py" not in file:
                stats = analyze_file_basic(file)
                if stats:
                    all_stats.append(stats)

    return all_stats


def print_summary(stats_list: List[Dict]):
    """Print a summary of the basic text analysis."""
    if not stats_list:
        print("No files analyzed.")
        return

    print("\n" + "=" * 100)
    print("BASIC TEXT COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 100)

    # Calculate averages
    avg_words = (
        sum(s["word_count"] for s in stats_list) / len(stats_list) if stats_list else 0
    )
    avg_lines = (
        sum(s["line_count"] for s in stats_list) / len(stats_list) if stats_list else 0
    )
    avg_word_len = (
        sum(s["avg_word_length"] for s in stats_list) / len(stats_list)
        if stats_list
        else 0
    )
    avg_sentence_len = (
        sum(s["avg_sentence_length"] for s in stats_list) / len(stats_list)
        if stats_list
        else 0
    )
    avg_syllables = (
        sum(s["avg_syllables_per_word"] for s in stats_list) / len(stats_list)
        if stats_list
        else 0
    )
    avg_complexity = (
        sum(s["complexity_score"] for s in stats_list) / len(stats_list)
        if stats_list
        else 0
    )

    print(f"Total files analyzed: {len(stats_list)}")
    print(f"Average lines per file: {avg_lines:.2f}")
    print(f"Average words per file: {avg_words:.2f}")
    print(f"Average word length: {avg_word_len:.2f} characters")
    print(f"Average sentence length: {avg_sentence_len:.2f} words")
    print(f"Average syllables per word: {avg_syllables:.2f}")
    print(f"Average complexity score: {avg_complexity:.2f}")

    print("\nFile-by-file breakdown:")
    print("-" * 100)
    print(
        f"{'File':<30} {'Lines':<8} {'Words':<8} {'Avg Word':<10} {'Avg Sent':<10} {'Complexity':<12} {'Syllables/W':<12}"
    )
    print("-" * 100)

    for stat in stats_list:
        filename = os.path.basename(stat.get("filename", "unknown"))[:29]
        line_count = stat.get("line_count", 0)
        word_count = stat.get("word_count", 0)
        avg_word_len = stat.get("avg_word_length", 0)
        avg_sentence_len = stat.get("avg_sentence_length", 0)
        complexity_score = stat.get("complexity_score", 0)
        syllables_per_word = stat.get("avg_syllables_per_word", 0)

        print(
            f"{filename:<30} {line_count:<8} {word_count:<8} {avg_word_len:<10.2f} {avg_sentence_len:<10.2f} {complexity_score:<12.2f} {syllables_per_word:<12.2f}"
        )

    print("\nComplexity Interpretation:")
    print("-" * 50)
    print("Complexity Score (higher = more complex):")
    print("  < 3.0: Simple text (e.g., documentation)")
    print("  3.0-5.0: Moderate complexity (e.g., code comments)")
    print("  5.0-8.0: Complex text (e.g., technical documentation)")
    print("  > 8.0: Very complex text")
    print("\nAverage syllables per word:")
    print("  < 1.5: Simple vocabulary")
    print("  1.5-2.0: Moderate vocabulary")
    print("  > 2.0: Complex/technical vocabulary")

    # Identify most and least complex files
    if stats_list:
        most_complex = max(stats_list, key=lambda x: x["complexity_score"])
        least_complex = min(
            stats_list,
            key=lambda x: x["complexity_score"]
            if x["complexity_score"] > 0
            else float("inf"),
        )

        print(
            f"\nMost complex file: {os.path.basename(most_complex['filename'])} (score: {most_complex['complexity_score']})"
        )
        print(
            f"Least complex file: {os.path.basename(least_complex['filename'])} (score: {least_complex['complexity_score']})"
        )


def main():
    """Main function to run basic text analysis."""
    project_dir = "/Users/doughnut/Library/CloudStorage/OneDrive-VictoriaUniversityofWellington-STAFF/Submitted/policy_societaldam_pharma/canonical_code"

    print(f"Analyzing text complexity of files in: {project_dir}")

    # Analyze Python, Markdown, and text files
    stats = analyze_directory(project_dir, [".py", ".md", ".txt"])

    print_summary(stats)


if __name__ == "__main__":
    main()
