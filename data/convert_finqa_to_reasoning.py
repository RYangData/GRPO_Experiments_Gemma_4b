import json
import re
import random

def clean_context(context):
    """Clean up the context text by removing unnecessary formatting"""
    # Remove any HTML-like tags
    context = re.sub(r'<[^>]+>', '', context)
    return context.strip()

def generate_reasoning(question, answer, think_content):
    """Generate a detailed reasoning based on the question, answer and the thinking content"""
    # Extract steps from the thinking content
    steps = think_content.strip().split('\n')
    
    # Parse the operations from the steps
    operations = []
    for step in steps:
        if step.startswith("Step"):
            parts = step.split(':', 1)[1].strip()
            op_match = re.search(r'(\w+)\((.*?)\)', parts)
            if op_match:
                op_name = op_match.group(1)
                op_args = op_match.group(2).split(',')
                operations.append((op_name, op_args))
    
    # Create a detailed reasoning text in GRPO format
    reasoning = ""
    
    # Build context-aware reasoning introduction
    introductions = [
        "To solve this problem, I need to analyze the information carefully.",
        "To answer this question, I'll work through the calculations step by step.",
        "I'll solve this problem by carefully examining the data provided.",
        "Let me work through this problem systematically."
    ]
    reasoning += random.choice(introductions) + "\n\n"
    
    # Add each step with detailed explanation
    for i, (op_name, op_args) in enumerate(operations):
        if op_name == "subtract":
            try:
                val1 = float(op_args[0])
                val2 = float(op_args[1])
                result = val1 - val2
                
                # Create more natural explanations
                explanation_options = [
                    f"First, I need to find the difference between {val1} and {val2}.",
                    f"I start by calculating the difference between {val1} and {val2}.",
                    f"Looking at the data, I need to subtract {val2} from {val1}."
                ]
                reasoning += f"{random.choice(explanation_options)}\n"
                reasoning += f"{val1} - {val2} = {result}\n\n"
            except:
                reasoning += f"I need to subtract {op_args[1]} from {op_args[0]}.\n\n"
        
        elif op_name == "divide":
            try:
                # Check if the first argument refers to a previous step result
                if op_args[0].startswith("#"):
                    prev_result = result  # Use the result from the previous step
                else:
                    prev_result = float(op_args[0])
                
                divisor = float(op_args[1])
                percentage_result = (prev_result / divisor) * 100  # Convert to percentage
                
                explanation_options = [
                    f"To find the percentage change, I need to divide the difference by the original value and multiply by 100.",
                    f"Next, I calculate the percentage by dividing {prev_result} by {divisor} and multiplying by 100.",
                    f"Now I convert this to a percentage by dividing by the original value ({divisor}) and multiplying by 100."
                ]
                reasoning += f"{random.choice(explanation_options)}\n"
                reasoning += f"({prev_result} / {divisor}) × 100 = {percentage_result:.1f}%\n\n"
                
                result = percentage_result  # Update result for next operation
            except:
                reasoning += f"I need to divide {op_args[0]} by {op_args[1]} to find the percentage.\n\n"
        
        elif op_name == "multiply":
            try:
                val1 = float(op_args[0])
                val2 = float(op_args[1])
                result = val1 * val2
                
                explanation_options = [
                    f"I need to multiply {val1} by {val2}.",
                    f"The next step is to multiply {val1} by {val2}.",
                    f"Now I'll multiply {val1} and {val2} together."
                ]
                reasoning += f"{random.choice(explanation_options)}\n"
                reasoning += f"{val1} × {val2} = {result}\n\n"
            except:
                reasoning += f"I need to multiply {op_args[0]} by {op_args[1]}.\n\n"
        
        elif op_name == "add":
            try:
                val1 = float(op_args[0])
                val2 = float(op_args[1])
                result = val1 + val2
                
                explanation_options = [
                    f"I need to add {val1} and {val2}.",
                    f"Now I'll add {val1} and {val2} together.",
                    f"Next, I add {val1} plus {val2}."
                ]
                reasoning += f"{random.choice(explanation_options)}\n"
                reasoning += f"{val1} + {val2} = {result}\n\n"
            except:
                reasoning += f"I need to add {op_args[0]} and {op_args[1]}.\n\n"
        
        else:
            # For any other operation, just describe it
            reasoning += f"Next, I apply the operation {op_name} on {', '.join(op_args)}.\n\n"
    
    # Add conclusion
    conclusion_options = [
        f"Therefore, the answer is {answer}.",
        f"Based on these calculations, the final answer is {answer}.",
        f"From the calculations above, I can determine that the answer is {answer}.",
        f"The result is {answer}."
    ]
    reasoning += random.choice(conclusion_options)
    
    return reasoning

def format_response(reasoning, answer):
    """Format the reasoning and answer in the format expected by GRPO training code"""
    # Format the response with notebook-specific tags
    formatted = f"<start_working_out>\n{reasoning}\n<end_working_out>\n<SOLUTION>\n{answer}\n</SOLUTION>"
    return formatted

def extract_think_steps(output):
    """Extract the thinking steps from the output"""
    think_match = re.search(r'Step.*?(?=\n<\/think>|\Z)', output, re.DOTALL)
    if think_match:
        return think_match.group(0).strip()
    return ""

def convert_entry(entry):
    """Convert a single FinQA entry to the new reasoning format"""
    data = json.loads(entry)
    
    prompt = data["prompt"]
    output = data["output"]
    
    # Extract question from prompt
    question_match = re.search(r'### Question:\s*(.*?)(?=\n\n|\Z)', prompt, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    
    # Extract context from prompt
    context_match = re.search(r'### Context:\s*(.*?)(?=\n\n### Table|\n\n### Question|\Z)', prompt, re.DOTALL)
    context = context_match.group(1).strip() if context_match else ""
    
    # Extract table if present
    table_match = re.search(r'### Table:\s*(.*?)(?=\n\n### Additional Context|\n\n### Question|\Z)', prompt, re.DOTALL)
    table = table_match.group(1).strip() if table_match else ""
    
    # Extract additional context if present
    additional_context_match = re.search(r'### Additional Context:\s*(.*?)(?=\n\n### Question|\Z)', prompt, re.DOTALL)
    additional_context = additional_context_match.group(1).strip() if additional_context_match else ""
    
    # Combine all context
    full_context = clean_context(context)
    if table:
        full_context += "\n\n" + table
    if additional_context:
        full_context += "\n\n" + additional_context
    
    # Extract final answer from output
    final_answer_match = re.search(r'Final Answer:\s*(.*?)(?=\Z)', output, re.DOTALL)
    final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
    
    # Extract thinking steps from output
    think_content = extract_think_steps(output)
    
    # Generate detailed reasoning
    reasoning = generate_reasoning(question, final_answer, think_content)
    
    # Format the response with both reasoning and answer tags to match GRPO expectations
    response = format_response(reasoning, final_answer)
    
    # Sometimes just return the final answer as extracted, to match some of the examples
    use_simplified = random.random() < 0.2  # 20% chance to use simplified extracted
    extracted = final_answer if use_simplified else reasoning
    
    # Format into new structure matching the GRPO format
    new_entry = {
        "question": f"{full_context}\n\nQuestion: {question}",
        "answer": final_answer,
        "response": response,
        "extracted": extracted
    }
    
    return new_entry

def convert_file(input_file, output_file):
    """Convert the entire FinQA JSONL file to the new reasoning format"""
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                new_entry = convert_entry(line.strip())
                f_out.write(json.dumps(new_entry) + '\n')
            except Exception as e:
                print(f"Error processing line: {e}")

if __name__ == "__main__":
    # Convert both train and dev files
    input_files = [
        "data/processed_datasets/finqa_program_answer_train.jsonl",
        "data/processed_datasets/finqa_program_answer_dev.jsonl"
    ]
    
    output_files = [
        "data/processed_datasets/finqa_reasoning_train.jsonl",
        "data/processed_datasets/finqa_reasoning_dev.jsonl"
    ]
    
    for input_file, output_file in zip(input_files, output_files):
        print(f"Converting {input_file} to {output_file}...")
        convert_file(input_file, output_file)
        print(f"Conversion complete for {input_file}") 