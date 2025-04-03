# %%
from google.colab import drive
drive.mount('/content/drive')

# %%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
    !pip install --no-deps unsloth vllm
# Install latest Hugging Face for Gemma-3!
!pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# %%
!pip install bitsandbytes
!pip install unsloth_zoo
!pip install msgspec
!pip install blake3
!pip install gguf


# %%
# 1. Import required libraries
import json
import torch
from unsloth import FastModel
from transformers import TextStreamer
from pathlib import Path
import time
import re

# 2. Define paths and settings
base_dir = "/content/drive/MyDrive/2025_ConvFinQA_SFT_Agentic"
model_path = f"{base_dir}/finqa-gemma3-program_answer-full"
data_dir = Path(f"{base_dir}/data/processed_datasets")
dev_path = data_dir / "finqa_program_answer_dev.jsonl"

# Toggle for verbose output - set to True to see all examples
VERBOSE_OUTPUT = True

# Maximum number of examples to evaluate (set to None to evaluate all)
# MAX_EXAMPLES: int | None = 100
MAX_EXAMPLES = 100

# 3. Load the fine-tuned model and tokenizer
print("Loading fine-tuned model...")
model, tokenizer = FastModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    load_in_4bit = True,
)

# 4. Set up the tokenizer with Gemma 3 chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
print("Model loaded successfully!")

# 5. Load all dev examples
print("Loading development examples...")
with open(dev_path, 'r') as f:
    dev_data = [json.loads(line) for line in f]
print(f"Loaded {len(dev_data)} dev examples")

# 6. Define a function to test the model on examples with recommended Gemma 3 parameters
def test_model_on_example(example, model, tokenizer):
    """Test the model on a single example using recommended Gemma 3 parameters"""
    # Create message format
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": example["prompt"].replace("### Response:\n", "")
        }]
    }]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )

    # Generate response using recommended Gemma 3 parameters
    start_time = time.time()
    outputs = model.generate(
        **tokenizer([text], return_tensors="pt").to("cuda"),
        max_new_tokens = 1024,  # Increased for complex financial reasoning
        temperature = 1.0,      # Gemma 3 recommended
        top_p = 0.95,           # Gemma 3 recommended
        top_k = 64,             # Gemma 3 recommended
    )
    end_time = time.time()

    # Extract model output
    model_output = tokenizer.decode(outputs[0]).split("<start_of_turn>model\n")[1]
    if "<end_of_turn>" in model_output:
        model_output = model_output.split("<end_of_turn>")[0]

    generation_time = end_time - start_time

    return {
        "model_output": model_output,
        "expected_output": example.get("output", example.get("answer", "")),
        "generation_time": generation_time,
        "prompt": example["prompt"]  # Include prompt for reference
    }

# 7. Extract final answers for exact matching
def extract_final_answer(text):
    """Extract final answer from text, handling various formats"""
    if "Final Answer:" in text:
        final_answer = text.split("Final Answer:")[1].strip()
    elif "final answer:" in text.lower():
        final_answer = text.split("final answer:", 1)[1].strip()
    else:
        # If no explicit final answer marker, take the last sentence
        sentences = text.split('.')
        final_answer = sentences[-1].strip()

    # Clean up whitespace and punctuation for comparison
    final_answer = re.sub(r'\s+', ' ', final_answer).strip()
    final_answer = re.sub(r'[,.;:]$', '', final_answer).strip()

    return final_answer

# Updated numeric matching function with better rounding handling
def is_numeric_match(expected, generated):
    """Check if numbers in the answers match, including handling rounding differences"""
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

    # Handle percentage rounding differences
    if expected_is_pct and generated_is_pct:
        # Allow more tolerance for percentages
        # For example, 62.9% and 63% should match
        return abs(expected_float - generated_float) <= 0.2  # Higher tolerance for percentages
    else:
        # For non-percentage values, use stricter comparison
        relative_diff = abs(expected_float - generated_float) / max(abs(expected_float), 0.0001)
        return relative_diff < 0.01  # 1% relative tolerance

# Helper function to extract question from prompt
def extract_question(prompt):
    """Extract the question part from a prompt"""
    if "### Question:" in prompt:
        question = prompt.split("### Question:")[1].split("### Response:")[0].strip()
    else:
        question = prompt.split("\n\n")[-2].strip()
    return question

# 8. Run full evaluation on the entire dev set
print("\n" + "="*80)
print("RUNNING FULL EVALUATION ON DEV SET")
print("="*80)

if VERBOSE_OUTPUT:
    print("Verbose output enabled - showing all examples")
else:
    print("Verbose output disabled - showing summary only")

# Track metrics
all_results = []
total_time = 0
exact_match_count = 0
numeric_match_count = 0

# Run evaluation
dev_data_to_evaluate = dev_data[:MAX_EXAMPLES] if MAX_EXAMPLES is not None else dev_data
total_examples = len(dev_data_to_evaluate)
print(f"Starting evaluation on {total_examples} examples...")

# Create output file for all predictions
predictions_file = f"{base_dir}/evaluation_outputs/all_predictions_program_full.jsonl"
with open(predictions_file, 'w') as pred_file:
    for i, example in enumerate(dev_data_to_evaluate):
        # Print progress updates
        if not VERBOSE_OUTPUT and ((i+1) % 25 == 0 or i+1 == total_examples):
            print(f"Progress: {i+1}/{total_examples} examples evaluated")

        # Run the model on this example
        result = test_model_on_example(example, model, tokenizer)

        # Extract final answers for comparison
        expected_answer = extract_final_answer(result["expected_output"])
        model_answer = extract_final_answer(result["model_output"])

        # Extract question for display
        question = extract_question(example["prompt"])

        # Check for exact match
        exact_match = expected_answer.lower() == model_answer.lower()
        if exact_match:
            exact_match_count += 1
            match_status = "exact"
        else:
            # Check for numeric match with improved tolerance
            numeric_match = is_numeric_match(expected_answer, model_answer)
            if numeric_match:
                numeric_match_count += 1
                match_status = "numeric"
            else:
                match_status = "incorrect"

        # Save results
        result["exact_match"] = exact_match
        result["numeric_match"] = numeric_match
        result["expected_final"] = expected_answer
        result["model_final"] = model_answer
        result["question"] = question
        result["match_status"] = match_status

        all_results.append(result)
        total_time += result["generation_time"]

        # Write individual result to JSONL file
        pred_file.write(json.dumps({
            "id": example.get("id", f"example_{i}"),
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "exact_match": exact_match,
            "numeric_match": numeric_match,
            "match_status": match_status,
            "full_model_output": result["model_output"]
        }) + "\n")

        # Print verbose output if enabled
        if VERBOSE_OUTPUT:
            if exact_match:
                display_status = "[EXACT MATCH]"
            elif numeric_match:
                # Show the values for numeric matches to verify
                display_status = f"[NUMERIC MATCH: {expected_answer} ≈ {model_answer}]"
            else:
                display_status = "[INCORRECT]"

            print(f"\n{'-'*80}")
            print(f"Example {i+1}/{total_examples} {display_status}")
            print(f"Question: {question}")
            print(f"\nExpected answer: {expected_answer}")
            print(f"Model answer: {model_answer}")

            if not exact_match:  # Show full outputs for non-exact matches
                print("\nFull expected output:")
                print(result["expected_output"][:500] + "..." if len(result["expected_output"]) > 500 else result["expected_output"])
                print("\nFull model output:")
                print(result["model_output"][:500] + "..." if len(result["model_output"]) > 500 else result["model_output"])

            print(f"Generation time: {result['generation_time']:.2f} seconds")

print(f"\nAll predictions written to: {predictions_file}")

# 9. Calculate and print evaluation metrics
avg_generation_time = total_time / total_examples
exact_match_accuracy = (exact_match_count / total_examples) * 100
numeric_match_accuracy = ((exact_match_count + numeric_match_count) / total_examples) * 100

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Total examples evaluated: {total_examples}")
print(f"Average generation time: {avg_generation_time:.2f} seconds")
print(f"Exact match accuracy: {exact_match_accuracy:.2f}%")
print(f"Numeric match accuracy (includes exact + approximate): {numeric_match_accuracy:.2f}%")
print(f"Exact matches: {exact_match_count}/{total_examples}")
print(f"Numeric-only matches: {numeric_match_count}/{total_examples}")

# 10. Analyze rounding differences
print("\n" + "="*80)
print("ROUNDING DIFFERENCE ANALYSIS")
print("="*80)

# Find examples where the only difference is rounding
rounding_differences = []
for result in all_results:
    if not result["exact_match"] and result["numeric_match"]:
        # Extract numbers from both answers
        expected_numbers = re.findall(r'-?[\d,]*\.\d+%?|-?\d+%?', result["expected_final"])
        generated_numbers = re.findall(r'-?[\d,]*\.\d+%?|-?\d+%?', result["model_final"])

        if expected_numbers and generated_numbers:
            expected_last = expected_numbers[-1]
            generated_last = generated_numbers[-1]

            # Only include if both have % or neither has %
            if expected_last.endswith('%') == generated_last.endswith('%'):
                expected_float = float(expected_last.replace('%', '').replace(',', ''))
                generated_float = float(generated_last.replace('%', '').replace(',', ''))

                difference = abs(expected_float - generated_float)
                if difference <= 0.2:  # Small rounding difference
                    rounding_differences.append({
                        "question": result["question"],
                        "expected": result["expected_final"],
                        "model": result["model_final"],
                        "difference": difference
                    })

# Print summary of rounding differences
print(f"Found {len(rounding_differences)} examples with rounding differences")
print(f"These are counted as correct in the numeric match accuracy")

# Show a few examples
if rounding_differences:
    print("\nExamples of rounding differences:")
    for i, rd in enumerate(rounding_differences[:5]):
        print(f"\nRounding Example {i+1}:")
        print(f"Question: {rd['question']}")
        print(f"Expected: {rd['expected']}")
        print(f"Model: {rd['model']}")
        print(f"Difference: {rd['difference']:.2f}")

# 11. Analyze errors to understand where the model struggles
print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

# Categorize examples by steps required
examples_by_steps = {}
for example in dev_data:
    output = example.get("output", "")
    step_count = output.count("Step ")
    if step_count not in examples_by_steps:
        examples_by_steps[step_count] = []
    examples_by_steps[step_count].append(example)

# Calculate accuracy per step count
print("Accuracy by reasoning step count:")
for step_count in sorted(examples_by_steps.keys()):
    examples = examples_by_steps[step_count]
    correct = 0
    for ex in examples:
        for result in all_results:
            if result["expected_output"] == ex.get("output", "") and (result["exact_match"] or result["numeric_match"]):
                correct += 1
                break

    accuracy = (correct / len(examples)) * 100
    print(f"  {step_count} steps: {accuracy:.2f}% ({correct}/{len(examples)})")

# 12. Save detailed results to JSON for further analysis
results_file = f"{base_dir}/evaluation_outputs/evaluation_results_program_full.json"
with open(results_file, 'w') as f:
    # Create a summary object
    summary = {
        "total_examples": total_examples,
        "exact_match_accuracy": exact_match_accuracy,
        "numeric_match_accuracy": numeric_match_accuracy,
        "exact_match_count": exact_match_count,
        "numeric_match_count": numeric_match_count,
        "rounding_difference_count": len(rounding_differences),
        "avg_generation_time": avg_generation_time,
        "step_accuracies": {
            str(step_count): {
                "accuracy": (sum(1 for ex in examples_by_steps[step_count]
                             for res in all_results if res["expected_output"] == ex.get("output", "")
                             and (res["exact_match"] or res["numeric_match"])) / len(examples_by_steps[step_count])) * 100,
                "count": len(examples_by_steps[step_count])
            } for step_count in examples_by_steps
        }
    }

    # We need to clean the results for JSON serialization
    clean_results = []
    for res in all_results:
        clean_res = {
            "question": res["question"],
            "expected_final": res["expected_final"],
            "model_final": res["model_final"],
            "exact_match": res["exact_match"],
            "numeric_match": res["numeric_match"],
            "match_status": res["match_status"],
            "generation_time": res["generation_time"]
        }
        clean_results.append(clean_res)

    # Combine summary and results
    output_data = {
        "summary": summary,
        "rounding_differences": [{
            "question": rd["question"],
            "expected": rd["expected"],
            "model": rd["model"],
            "difference": rd["difference"]
        } for rd in rounding_differences],
        "results": clean_results
    }

    json.dump(output_data, f, indent=2)

print(f"\nDetailed results saved to {results_file}")

# 13. Display a few examples of correct and incorrect predictions if not in verbose mode
if not VERBOSE_OUTPUT:
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    # Get some correct and incorrect examples
    correct_examples = [res for res in all_results if res["exact_match"] or res["numeric_match"]]
    incorrect_examples = [res for res in all_results if not res["exact_match"] and not res["numeric_match"]]

    # Show 3 correct examples
    print("\nCORRECT PREDICTIONS:")
    for i, res in enumerate(correct_examples[:3]):
        print(f"\nCorrect Example {i+1}:")
        print(f"Question: {res['question']}")
        print(f"Expected: {res['expected_final']}")
        print(f"Model output: {res['model_final']}")
        print("-" * 50)

    # Show 3 incorrect examples
    print("\nINCORRECT PREDICTIONS:")
    for i, res in enumerate(incorrect_examples[:3]):
        print(f"\nIncorrect Example {i+1}:")
        print(f"Question: {res['question']}")
        print(f"Expected: {res['expected_final']}")
        print(f"Model output: {res['model_final']}")
        print("-" * 50)

print("\nEvaluation complete!")

# %%


# %%


# %%


# %%


# %%



