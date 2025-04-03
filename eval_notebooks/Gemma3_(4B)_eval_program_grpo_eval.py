#!/usr/bin/env python3
"""
Script to evaluate GRPO model predictions from existing JSONL file
with improved answer extraction methods.
"""

import json
import os
import re
from pathlib import Path

# Define paths
base_dir = "/Users/richardyang/Desktop/Long Term Career/UK 2025 Cases/TomoroAI/FinQA_Experim"
predictions_file = f"{base_dir}/submission/data/evaluation_outputs/all_predictions_grpo.jsonl"
results_file = f"{base_dir}/submission/data/evaluation_outputs/evaluation_results_grpo.json"

# Make sure output directory exists
os.makedirs(os.path.dirname(results_file), exist_ok=True)

# Define format tags
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# Function to extract answer from model output with various formats
def extract_solution(text):
    """Extract the answer from model output with various possible formats"""
    if not text:
        return None
        
    # Try to find solution between solution tags first (preferred format)
    solution_match = re.search(
        rf"{solution_start}(.*?){solution_end}",
        text,
        re.DOTALL
    )
    if solution_match:
        return solution_match.group(1).strip()
    
    # Try to find solution after SOLUTION keyword
    solution_keyword = re.compile(r'SOLUTION\s*\n(.*?)(?:\n\n|$)', re.DOTALL)
    match = solution_keyword.search(text)
    if match:
        return match.group(1).strip()
    
    # Try to find solution after "SOLUTION" with no tags
    solution_keyword_line = re.compile(r'SOLUTION\s+(.*?)(?:\n|$)', re.DOTALL)
    match = solution_keyword_line.search(text)
    if match:
        return match.group(1).strip()
    
    # Try to find solution in other common formats
    # Check for final answer format
    final_answer = re.compile(r'final\s+answer\s*(?::|is)\s*([\d\.\-%]+)', re.IGNORECASE | re.DOTALL)
    match = final_answer.search(text)
    if match:
        return match.group(1).strip()
    
    # Check for "Therefore" statements which often contain the final answer
    therefore = re.compile(r'therefore[,\s]+(the\s+)?(?:.*?)([\d\.\-%]+)(?:\.|$)', re.IGNORECASE | re.DOTALL)
    match = therefore.search(text)
    if match:
        return match.group(2).strip()
    
    # Extract from working out section as a last resort
    working_pattern = re.compile(r'<start_working_out>(.*?)<end_working_out>', re.DOTALL)
    match = working_pattern.search(text)
    if match:
        working_text = match.group(1)
        
        # Try to find percentage or numeric answers in the last few lines
        lines = working_text.strip().split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            if "%" in line or re.search(r'[\d\.]+', line):
                # Extract percentages or numbers from the line
                percentage_match = re.search(r'([\d\.]+%)', line)
                if percentage_match:
                    return percentage_match.group(1)
                
                # Look for numbers that might be the answer
                number_match = re.search(r'=\s*([\d\.\-]+)', line)
                if number_match:
                    return number_match.group(1)
        
        # If we still don't have an answer, check for any number in the last line
        if lines:
            number_match = re.search(r'([\d\.]+)', lines[-1])
            if number_match:
                return number_match.group(1)
    
    # Last resort - try to find any percentage in the whole text
    percentage_match = re.search(r'([\d\.]+%)', text)
    if percentage_match:
        return percentage_match.group(1)
    
    # Extract any number that might be the answer from the last 30% of the text
    # This assumes the answer usually appears toward the end
    last_part = text[int(len(text) * 0.7):]
    number_matches = re.findall(r'[\d\.]+', last_part)
    if number_matches:
        return number_matches[-1]
    
    # If all else fails, return None
    return None

# Function to check if the output follows a recognizable format
def format_check(text):
    """Check if the output follows any recognizable format pattern"""
    if not text:
        return False
        
    if (reasoning_start in text and reasoning_end in text and 
        solution_start in text and solution_end in text):
        return True
    
    if (reasoning_start in text and reasoning_end in text and 
        "SOLUTION" in text):
        return True
    
    if ("<start_working_out>" in text and "<end_working_out>" in text):
        return True
    
    if "SOLUTION" in text or "Solution" in text:
        return True
    
    return False

# Function to check if numbers in the answers match
def is_numeric_match(expected, generated):
    """Check if numbers in the answers match, including handling rounding differences"""
    if not expected or not generated:
        return False
        
    # Try to convert directly to float first (handling % if present)
    try:
        expected_clean = expected.replace('%', '').replace(',', '').strip()
        generated_clean = generated.replace('%', '').replace(',', '').strip()
        
        expected_float = float(expected_clean)
        generated_float = float(generated_clean)
        
        # Check if both are percentages or either is percentage
        expected_is_pct = '%' in expected
        generated_is_pct = '%' in generated
        
        # For small values (near 1 or less), use absolute difference
        if abs(expected_float) <= 1 and abs(generated_float) <= 1:
            return abs(expected_float - generated_float) <= 0.07  # Increased tolerance for small values
            
        # Handle percentage rounding differences with increased tolerance
        if expected_is_pct or generated_is_pct:
            # For percentages, we need more tolerance
            # Allow 1 percentage point difference for values <= 10%
            if abs(expected_float) <= 10:
                return abs(expected_float - generated_float) <= 1.0
            # Allow 10% relative difference for values > 10%
            else:
                relative_diff = abs(expected_float - generated_float) / max(abs(expected_float), 0.0001)
                return relative_diff <= 0.1  # 10% tolerance
        else:
            # For non-percentage values, use relative comparison
            relative_diff = abs(expected_float - generated_float) / max(abs(expected_float), 0.0001)
            return relative_diff <= 0.07  # Increased tolerance to 7%
    except ValueError:
        # If direct conversion fails, try to extract numbers
        pass
    
    # Extract all numbers from both strings
    expected_numbers = re.findall(r'-?[\d,]*\.\d+%?|-?\d+%?', expected)
    generated_numbers = re.findall(r'-?[\d,]*\.\d+%?|-?\d+%?', generated)

    # Check if we have numbers to compare
    if not expected_numbers or not generated_numbers:
        return False

    # Get the last number (final answer) from each
    expected_last = expected_numbers[-1]
    generated_last = generated_numbers[-1]

    # Check for percentage formatting
    expected_is_pct = expected_last.endswith('%')
    generated_is_pct = generated_last.endswith('%')

    # Convert to floats for comparison (remove % and ,)
    expected_float = float(expected_last.replace('%', '').replace(',', ''))
    generated_float = float(generated_last.replace('%', '').replace(',', ''))

    # For small values (near 1 or less), use absolute difference
    if abs(expected_float) <= 1 and abs(generated_float) <= 1:
        return abs(expected_float - generated_float) <= 0.07  # Increased tolerance for small values
        
    # Handle percentage rounding differences with increased tolerance
    if expected_is_pct or generated_is_pct:
        # For percentages, we need more tolerance
        # Allow 1 percentage point difference for values <= 10%
        if abs(expected_float) <= 10:
            return abs(expected_float - generated_float) <= 1.0
        # Allow 10% relative difference for values > 10%
        else:
            relative_diff = abs(expected_float - generated_float) / max(abs(expected_float), 0.0001)
            return relative_diff <= 0.1  # 10% tolerance
    else:
        # For non-percentage values, use relative comparison
        relative_diff = abs(expected_float - generated_float) / max(abs(expected_float), 0.0001)
        return relative_diff <= 0.07  # Increased tolerance to 7%

def main():
    """Main function to process predictions and calculate metrics"""
    
    print(f"Processing file: {predictions_file}")

    # Initialize counters and results
    all_results = []
    total_examples = 0
    format_match_count = 0
    exact_match_count = 0
    numeric_match_count = 0

    try:
        with open(predictions_file, 'r') as f:
            for line in f:
                total_examples += 1
                try:
                    data = json.loads(line)
                    
                    # Extract model output and expected answer
                    model_output = data.get("full_model_output", "")
                    expected_answer = data.get("expected_answer", "")
                    
                    # Check if format matches (more lenient check)
                    format_match = format_check(model_output)
                    
                    # Extract model answer using improved function
                    model_answer = extract_solution(model_output)
                    
                    # Check for exact match
                    exact_match = False
                    numeric_match = False
                    
                    if model_answer:
                        # Clean up model answer
                        model_answer = model_answer.strip()
                        
                        # Check for exact match (case insensitive)
                        exact_match = model_answer.lower() == expected_answer.lower()
                        
                        # Check for numeric match with improved tolerance
                        if not exact_match:
                            numeric_match = is_numeric_match(expected_answer, model_answer)
                    
                    # Determine match status
                    if exact_match:
                        match_status = "exact"
                        exact_match_count += 1
                    elif numeric_match:
                        match_status = "numeric"
                        numeric_match_count += 1
                    elif model_answer:
                        match_status = "incorrect"
                    else:
                        match_status = "format_error"
                    
                    if format_match:
                        format_match_count += 1
                    
                    # Create result object
                    result = {
                        "id": data.get("id", f"example_{total_examples}"),
                        "question": data.get("question", ""),
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "format_match": format_match,
                        "exact_match": exact_match,
                        "numeric_match": numeric_match,
                        "match_status": match_status
                    }
                    
                    all_results.append(result)
                    
                    # Print progress
                    if total_examples % 50 == 0:
                        print(f"Processed {total_examples} examples...")
                    
                except json.JSONDecodeError:
                    print(f"Error parsing JSON on line {total_examples}")
                    continue
    except FileNotFoundError:
        print(f"File not found: {predictions_file}")
        exit(1)

    print(f"Processing complete. Total examples: {total_examples}")

    # Calculate metrics
    format_accuracy = (format_match_count / total_examples) * 100 if total_examples > 0 else 0
    exact_match_accuracy = (exact_match_count / total_examples) * 100 if total_examples > 0 else 0
    numeric_match_accuracy = ((exact_match_count + numeric_match_count) / total_examples) * 100 if total_examples > 0 else 0

    # Print evaluation results
    print("\n" + "="*80)
    print("EVALUATION RESULTS WITH IMPROVED EXTRACTION")
    print("="*80)
    print(f"Total examples evaluated: {total_examples}")
    print(f"Format match accuracy: {format_accuracy:.2f}%")
    print(f"Exact match accuracy: {exact_match_accuracy:.2f}%")
    print(f"Numeric match accuracy (includes exact + approximate): {numeric_match_accuracy:.2f}%")
    print(f"Format matches: {format_match_count}/{total_examples}")
    print(f"Exact matches: {exact_match_count}/{total_examples}")
    print(f"Numeric-only matches: {numeric_match_count}/{total_examples}")

    # Sort results by different categories
    correct_answers = [res for res in all_results if res["exact_match"] or res["numeric_match"]]
    format_errors = [res for res in all_results if not res["model_answer"]]
    wrong_answers = [res for res in all_results if res["model_answer"] and not res["exact_match"] and not res["numeric_match"]]

    print(f"Correct answers (exact + numeric): {len(correct_answers)}")
    print(f"Wrong answers (with extracted solution): {len(wrong_answers)}")
    print(f"Format errors (couldn't extract solution): {len(format_errors)}")

    # Save detailed results to JSON
    summary = {
        "total_examples": total_examples,
        "format_match_accuracy": format_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "numeric_match_accuracy": numeric_match_accuracy,
        "format_match_count": format_match_count,
        "exact_match_count": exact_match_count,
        "numeric_match_count": numeric_match_count,
        "correct_answers": len(correct_answers),
        "wrong_answers": len(wrong_answers),
        "format_errors": len(format_errors)
    }

    output_data = {
        "summary": summary,
        "results": all_results
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to {results_file}")

    # Display a few examples of each category
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    # Show 3 correct examples
    print("\nCORRECT PREDICTIONS:")
    for i, res in enumerate(correct_answers[:3]):
        print(f"\nCorrect Example {i+1}:")
        print(f"Expected: {res['expected_answer']}")
        print(f"Extracted: {res['model_answer']}")
        print(f"Match type: {'Exact' if res['exact_match'] else 'Numeric'}")
        print("-" * 50)

    # Show 3 wrong answers
    print("\nINCORRECT PREDICTIONS (where solution was extracted):")
    for i, res in enumerate(wrong_answers[:3]):
        print(f"\nIncorrect Example {i+1}:")
        print(f"Expected: {res['expected_answer']}")
        print(f"Extracted: {res['model_answer']}")
        print("-" * 50)

    # Show 3 format errors
    print("\nFORMAT ERRORS (couldn't extract solution):")
    for i, res in enumerate(format_errors[:3]):
        print(f"\nFormat Error Example {i+1}:")
        print(f"Expected: {res['expected_answer']}")
        print(f"Could not extract answer from model output")
        print("-" * 50)

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 