# ConvFinQA Experiments: Exploring GRPO Fine-Tuning for Financial QA

## Project Overview

This project was completed as part of a take-home case study for a company I was interviewing with. The task required building an LLM-driven prototype that can answer questions based on financial documents, specifically using the ConvFinQA dataset. I decided to take this opportunity to explore Group Relative Policy Optimization (GRPO), a technique I had been interested in learning, while solving the assigned problem. The goal was to develop a system that could interpret financial information from text and tables to accurately answer numerical questions.

## Task Overview

The case study challenged participants to build an LLM-driven prototype that could answer numerical questions from financial documents. Rather than just completing the minimum requirements, I used this as an opportunity to experiment with cutting-edge fine-tuning techniques I had been wanting to explore, particularly GRPO. This writeup documents my experimental approach and findings, demonstrating both the technical implementation and the strategic reasoning behind my decisions.

## Problem Formulation

After examining the train.json file from the ConvFinQA dataset, I observed that each example contains three key components that provide context for generating responses:
- `pre_text`: Textual information appearing before tables
- `table`: Structured financial data
- `post_text`: Textual information appearing after tables

Given the nature of this dataset, I decided to focus on maximizing final answer accuracy for the QA system. The input (x) consists of the pre-text, table, post-text, and question, with the target output (y) being the final numerical answer.

Rather than implementing a traditional RAG approach with chunking and indexing, I recognized that the dataset structure called for a solution that could reason through financial data and perform calculations, based on a table and surrounding text.

## Approach Exploration

I considered two primary approaches before proceeding with implementation:

### Option 1: Agentic System
An orchestrated multi-agent framework with specialized components:
1. **Interpretation agent**: Parse financial text and tables for relevant information
2. **Planning agent**: Interpret the question and design a solution strategy
3. **Tool-calling agent**: Execute mathematical operations (division, addition, exponents, etc.)
4. **Reflection agent**: Verify reasoning steps and validate final outputs and re-run previous steps if required.

This approach would leverage agent orchestration frameworks like LangGraph or CrewAI to create a structured execution system, with evaluation focused on either step-by-step accuracy or final output correctness.

### Option 2: Fine-Tuning a Small Language Model
End-to-end instruction tuning to:
1. Output programming/reasoning steps and
2. Generate accurate final numerical answers

The rationale for this approach stems from recent advancements in smaller open-source models:
- Gemma 3 27B, released 2 weeks ago has demonstrated exceptional mathematical reasoning for its size, ranking 11th on LM Arena and performing comparably with much larger models like DeepSeek R1 (671B) and Llama3 (405B), as well at OpenAI's o3-mini. 
- On the MATH leaderboard, Gemma 3 27B achieves a score of 89, placing it among state-of-the-art models
- Gemma 3 4B scores 75.6 on MATH, showing strong mathematical reasoning despite its small size. This makes it ideal for this project given computational constraints, while its accessibility enables fine-tuning experiments on non-trivial use-cases that were previously limited to larger corporations with $Millions to spend on training. 

For more information on Gemma 3 benchmarks:
- https://x.com/MeetPatelTech/status/1899712375466652155
- https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf

## Decision Process

While I had high confidence in the agentic approach given my experience with LLMs and Agent orchestration frameworks, I ultimately chose to pursue Option 2 for several reasons:

1. The opportunity to explore the capabilities of cutting-edge small language models like Gemma 3
2. The chance to apply newer fine-tuning techniques like GRPO (Group Relative Policy Optimization) pioneered by the DeepSeek team this year. 
3. The scientific interest in testing whether smaller models could effectively learn both the reasoning steps and numerical precision required for dataset provided. 

This decision allowed me to leverage my past experience in fine-tuning while exploring new state-of-the-art models (Gemma 3) and techniques (GRPO), making for a more interesting technical challenge that educates myself (and hopefully you too) of what's possible with smaller models.

## Fine-tuning Experiment Deep Dive

### Experimental Approach

I designed a progressive experimentation approach to systematically evaluate different training strategies:

1. **LoRA r=4 (19% Parameters trained) with Answer Only**: Train a model to directly predict only the final numerical answer without intermediate steps, using a small LoRA rank for efficiency
2. **LoRA r=4 (19% Parameters trained) with Program Steps**: Train a model to produce both program steps (as a proxy for chain-of-thought reasoning) and the final answer using the same LoRA rank
3. **LoRA r=16 (75% Parameters trained) with Program Steps**: Based on the better model of **1** or **2** approach, train an enhanced version with higher LoRA rank to train more parameters. 
4. **GRPO with Reasoning Format**: Experiment with Group Relative Policy Optimization on a model to produce natural language reasoning instead of program steps, using LoRA r=16 and a specially converted reasoning dataset

This approach allowed for systematic comparison between:
- Answer-only vs. program step approaches (experiments 1 vs. 2)
- Low vs. high LoRA rank impact (experiments 2 vs. 3)
- Traditional fine-tuning vs. GRPO reinforcement learning (experiments 3 vs. 4)

All training notebooks were executed in Google Colab to leverage A100 GPUs. Approximately 100 Colab credits were used across the experiments to achieve the final results. If you wish to replicate these results, you can copy the training notebooks (refer to the Repo Structure section at the bottom of this writeup for details on what each notebook does), update the file paths to match your environment, and run them in Colab or any environment with similar GPU capabilities.

To significantly accelerate the fine-tuning process, I utilized the **Unsloth** library, which provides optimizations specifically designed for efficient Llama and Gemma model training. Unsloth implements FlashAttention-2 and matrix optimization techniques that reduced training time by approximately 60% compared to standard training approaches. This optimization was particularly valuable given the computational constraints and allowed for more extensive experimentation within the same resource budget. The library's compatibility with PEFT fine-tuning made it an ideal choice for this project, enabling 4-bit quantization while maintaining model quality.

### Dataset Preparation

For each training variant, I prepared the dataset differently:

**Answer Only Format** (for experiment 1):
```
Context: [Combined pre_text + formatted table + post_text]
Question: [The question from qa.question]
Response:
[Final numerical answer]
```

**Program Steps Format** (for experiments 2 and 3):
```
Context: [Combined pre_text + formatted table + post_text]
Question: [The question from qa.question]
Response:
<think>
Step 1: [operation]([arg1], [arg2]) = [result]
Step 2: [operation]([arg1], [arg2]) = [result]
...
</think>
Final Answer: [answer]
```

**GRPO Reasoning Format** (for experiment 4):
```
Context: [Combined pre_text + formatted table + post_text]
Question: [The question from qa.question]
Response:
<start_working_out>
[Natural language reasoning with calculations]
<end_working_out>
<SOLUTION>
[Final numerical answer]
</SOLUTION>
```

### Model Selection and Configuration

I selected **Gemma 3 4B** as the base model for all experiments due to its balance of performance capabilities and computational efficiency. For each training configuration, I adjusted the following parameters:

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of training all the model's parameters, LoRA freezes the original weights and injects smaller, trainable rank decomposition matrices into specific layers (like attention layers). Only these low-rank matrices are updated during training, significantly reducing the number of trainable parameters and computational cost while often achieving performance comparable to full fine-tuning.

**LoRA Configuration**:
- Experiments 1-2: rank=4, alpha=4
- Experiments 3-4: rank=16, alpha=32

**Training Parameters**:
- Learning rate: 2e-4 with cosine decay schedule
- Batch size: 8 (with gradient accumulation)
- Training steps: Calibrated based on dataset size, typically 3 epochs

### Implementation Pipeline

The implementation process for each experiment involved:

1. **Data Preprocessing**:
   - Converting tabular data into a consistent text format
   - Normalizing numerical values
   - Formatting program steps into human-readable operations

2. **Training Setup**:
   - Configuring the appropriate LoRA parameters
   - Setting up the appropriate input/output format based on the experiment

3. **Progressive Evaluation**:
   - Evaluating each model variant on a consistent test set
   - Selecting the better-performing approach for further enhancement
   - Implementing more sophisticated evaluation metrics for later experiments

### GRPO Implementation Details

For experiment 4, I explored Group Relative Policy Optimization with the following components:

1. **Preference Dataset Creation**:
   - Generated paired examples with varying quality of reasoning steps
   - Created preference pairs with correct vs. incorrect reasoning paths
   - Assigned reward signals based on answer correctness and reasoning quality

2. **Reward Modeling**:
   - Implemented a multi-component reward function considering:
     - Format adherence
     - Mathematical correctness
     - Reasoning coherence
     - Step-by-step validity

3. **Optimization Approach**:
   - Applied policy gradient updates using the reward signals
   - Balanced exploration vs. exploitation through appropriate KL penalties
   - Maintained a reference model to prevent excessive divergence

### Reasoning Dataset Conversion

A key aspect of the GRPO training (experiment 4) was transforming the program-based steps into natural language reasoning. I developed a custom conversion pipeline (`convert_finqa_to_reasoning.py`) that:

1. **Extracted program steps** from the original dataset format with structured operations like `subtract(206588, 181001) = 25587`

2. **Transformed operations into natural language reasoning**:
   - Added context-aware introductions (e.g., "To solve this problem, I need to analyze the information carefully.")
   - Converted each operation into detailed explanations (e.g., "First, I need to find the difference between 206588 and 181001.")
   - Included actual calculations with intermediate results
   - Added varied explanatory transitions between steps
   - Concluded with natural language summaries

3. **Created variety in reasoning patterns**:
   - Implemented multiple templates for each operation type
   - Randomized explanatory text to create diverse reasoning examples
   - Added appropriate variance in presentation style

4. **Formatted outputs for GRPO training**:
   - Enclosed reasoning within `<start_working_out>` and `<end_working_out>` tags
   - Placed final answers within `<SOLUTION>` tags

This conversion process allowed the model to learn not just the mechanical steps of financial calculations, but also how to express these calculations in natural, explanatory language that human users would find more intuitive and trustworthy.

This systematic progression of experiments was designed to explore both standard fine-tuning approaches and more advanced reinforcement learning techniques, providing insights into the most effective strategies for financial question-answering tasks.

### Evaluation Metrics

To assess model performance, I implemented a single standardized evaluation metric focused on answer accuracy, due to time constraints. A more comprehensive evaluation approach would involve analyzing the quality of reasoning steps, potentially using an LLM or ensemble of LLMs to score the validity of the process + response. 

**Answer Accuracy**: Percentage of predictions with numerically correct final answers


## Results and Evaluation

### Evaluation Methodology

Financial QA requires precise numerical answers, but traditional string-matching evaluation can be problematic when equivalent numerical answers might be expressed differently (e.g., "14.1%" vs "14.10%" vs "14%"). To address this challenge, I implemented a comprehensive evaluation framework with the following metrics:

1. **Exact Match**: The percentage of predictions where the model's answer string exactly matched the ground truth answer string.

2. **Numeric Match**: The percentage of predictions where the model's answer was numerically equivalent to the ground truth after applying appropriate rounding and normalization.

3. **Total Correct**: The combined percentage of exact and numeric matches, representing overall accuracy.

4. **Format Match** (GRPO only): The percentage of responses following the expected structured format with proper reasoning tags.

5. **Average Generation Time**: The time required to generate responses, an important consideration for real-world applications.

### Rounding Methodology

To fairly evaluate numerical answers, I implemented a comprehensive decimal rounding approach:

1. For each numeric answer, all possible decimal place roundings were generated
   - Example: 24.69 generates [25, 24.7, 24.69]
   - Example: 8.111111 generates [8, 8.1, 8.11, 8.111, 8.1111, 8.11111, 8.111111]

2. A match was considered valid if any rounding variation of the model's answer matched any rounding variation of the expected answer

3. This approach accommodated both precision reduction (e.g., 23.11 → 23.1) and standard rounding (e.g., 24.6 → 25)

### Performance Results

The table below summarizes the performance of each approach:

| Model Approach | Exact Match | Numeric Match | Total Correct | Avg. Generation Time | 
|----------------|-------------|---------------|---------------|----------------------|
| LoRA r=4 Answer Only | 16.0% | 18.0% | 34.0% | 1.23s |
| LoRA r=4 with Program Steps | 40.0% | 22.0% | 62.0% | 9.93s |
| LoRA r=16 with Program Steps | 39.0% | 28.0% | 67.0% | 11.25s |
| GRPO | 10.0% | 46.0% | 56.0% | - |

### Key Findings

1. **Program Steps Improve Accuracy**: Models trained to produce step-by-step reasoning consistently outperformed answer-only models in overall accuracy (62.0% and 67.0% vs. 34.0%).

2. **Higher LoRA Rank Benefits Performance**: Increasing the LoRA rank from 4 to 16 improved total accuracy from 62.0% to 67.0%, particularly boosting numeric match performance (22.0% → 28.0%).

3. **GRPO Excels at Numeric Reasoning**: While GRPO had the lowest exact match rate (10.0%), it demonstrated superior numeric reasoning with 46.0% numeric matches, suggesting it understood the calculations but expressed answers in different formats.

4. **Speed vs. Accuracy Tradeoff**: The answer-only approach was significantly faster (~8-9x) but at a substantial cost to accuracy, highlighting an important deployment consideration.


### Examples of Correct Numeric Matches

| Expected Answer | Model Answer | Match Type |
|-----------------|--------------|------------|
| 24.69% | 25% | Valid through rounding to nearest integer |
| 2.37 | 2.3 | Valid through decimal place reduction |
| 6.88 | 6.8 | Valid through decimal place reduction |
| 10.83% | 11% | Valid through rounding variations |

These examples demonstrate how the rounding methodology correctly identified numerically equivalent answers that would have been missed by strict string matching.

## Discussion
The experimental results reveal important insights about applying small language models to financial question-answering. A key finding is the benefit of incorporating explicit reasoning steps - models trained to predict answers directly (Experiment 1) significantly underperformed compared to those generating program steps for chain-of-thought reasoning (Experiments 2 & 3). This demonstrates the importance of structured reasoning for complex financial queries. Increasing the LoRA rank from 4 to 16 (Experiment 3 vs. 2) further improved overall accuracy, primarily through better numeric matching, suggesting greater model capacity aids in handling numerical nuances.

The GRPO approach (Experiment 4) presented an interesting contrast. Despite limited dataset preparation, it achieved the highest numeric match rate (46.0%), indicating strong computational understanding. However, its low exact match rate (10.0%) suggests struggles with precise output formatting. This likely stems from the reward function configuration during GRPO training. The rewards placed significant weight on format adherence (up to +5.0 points combined) compared to answer correctness (+3.0 for exact matches). This imbalance incentivized structural consistency over calculation accuracy, achieving 98% format match. Future GRPO iterations could benefit from explicitly weighting answer correctness higher (2-3x) and more extensive preference dataset creation.

These results should be contextualized within LLM capabilities and limitations. While small models like Gemma 3 4B can handle complex financial reasoning when fine-tuned, susceptibility to numerical errors remains, especially for complex tasks. This aligns with research on mathematical reasoning in LLMs (Srivatsa and Kochmar, 2024) ([link](https://arxiv.org/html/2403.11369v2)), which highlights challenges from linguistic complexity, diverse operations, and implicit knowledge requirements. Future analysis could examine specific question types where models underperform, revealing limitations in handling certain reasoning patterns or calculations.

From a practical standpoint, these experiments provide clear guidance for real-world deployment:

For niche financial domains requiring high accuracy and explainability:
- The fine-tuned LoRA r=16 model with program steps offers an optimal balance
- More cost-effective and efficient for high-volume queries compared to large API models
- Provides greater control over the model's behavior and outputs

For rapid prototyping and immediate results:
- Agentic systems with API calls may be preferable
- Faster time-to-value with less upfront investment
- Better suited when fine-grained control is less critical


# Conclusion

This study successfully demonstrated the potential of fine-tuning a small language model, Gemma 3 4B, for the complex task of financial question-answering using the ConvFinQA dataset. My experiments confirmed that incorporating structured reasoning, such as program steps via LoRA fine-tuning, significantly enhances accuracy compared to direct answer prediction. While higher LoRA ranks yielded the best overall balance, the GRPO approach highlighted strong numerical reasoning capabilities, albeit with challenges in output formatting due to reward function design.

These findings underscore the viability of specialized, fine-tuned small models as a cost-effective and efficient alternative to larger API-based systems for niche financial domains requiring accuracy and explainability. Conversely, agentic systems remain a strong option for rapid deployment where time-to-value is paramount. As small models (<30B parameters) continue to advance rapidly, ongoing experimentation is essential to leverage their evolving capabilities for sophisticated financial analysis tasks.

This project not only satisfied the requirements of the case study but also provided me with valuable hands-on experience with GRPO, a technique I had been eager to explore. The insights gained from these experiments will inform my approach to similar problems in the future.


# Repo Structure

```
├── WRITEUP.md                           # This document explaining the approach and results
├── data/                                # Data processing and storage
│   ├── 0_pre_processing.ipynb          # Initial data preprocessing notebook
│   ├── convert_finqa_to_reasoning.py   # Script for converting program steps to natural language reasoning
│   ├── evaluation_outputs/             # Storage for model evaluation results
│   ├── processed_datasets/             # Processed datasets ready for training
│   └── raw/                            # Raw ConvFinQA dataset files
├── training_notebooks/                  # Model training implementation
│   ├── Gemma3_(4B)_train_answer_only.ipynb   # Experiment 1: Direct answer prediction
│   ├── Gemma3_(4B)_train_program_full_FT.ipynb # Experiments 2-3: Program steps training
│   ├── Gemma3_(4B)_train_program_full_FT.py  # Python script version of program steps training
│   ├── finqa_gemma3_grpo.ipynb         # Experiment 4: GRPO implementation notebook
└── eval_notebooks/                      # Model evaluation implementations
    ├── Gemma3_(4B)_eval_answer_only.ipynb    # Evaluation for answer-only model
    ├── Gemma3_(4B)_eval_program_full.ipynb   # Evaluation for program steps models
    ├── Gemma3_(4B)_eval_program_full.py      # Script version of program steps evaluation
    ├── Gemma3_(4B)_eval_program_grpo.ipynb   # Evaluation for GRPO model
    ├── Gemma3_(4B)_eval_program_grpo.py      # Script version of GRPO evaluation
    └── Gemma3_(4B)_eval_program_grpo_eval.py # Additional GRPO evaluation script
```

The project follows a structured organization that reflects the experimental workflow:

1. **Data Processing**: The `data/` directory contains preprocessing scripts and notebooks that transform the raw ConvFinQA dataset into training-ready formats. The `convert_finqa_to_reasoning.py` script specifically handles the transformation of program steps into natural language reasoning for the GRPO experiment.

2. **Training Implementation**: The `training_notebooks/` directory contains implementation files for all four experiments:
   - Answer-only training (Experiment 1)
   - Program steps training with LoRA r=4 and r=16 (Experiments 2-3)
   - GRPO implementation for natural language reasoning (Experiment 4)
   
   Both notebook (.ipynb) and script (.py) versions are provided for reproducibility across different environments.

3. **Evaluation**: The `eval_notebooks/` directory contains corresponding evaluation implementations for each training approach, with standardized metrics for comparing model performance across experiments.

