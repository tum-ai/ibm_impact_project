from openai import OpenAI
#from evaluation import CustomMetrics
#from transformers import EvalPrediction
import json
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")
print(ds)
dataset = ds['test']

client = OpenAI(api_key='TODO-add-your-api-key')

def generate_gpt4_solution(problem: str) -> str:
    prompt = f"""Problem: {problem}
    
    Please solve the problem and provide the solution step-by-step. Conclude with the final answer in the format 'Final Answer: [your final answer here]'.
    """
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content.strip()

problems = []
ground_truth_solutions = []

for entry in dataset:
    problem = entry['question']  
    answer = entry['answer']  
    
    problems.append(problem)
    ground_truth_solutions.append(answer)

gpt4_solutions = [generate_gpt4_solution(problem) for problem in problems]

predictions = [{"problem": problem, "gpt4_solution": solution} for problem, solution in zip(problems, gpt4_solutions)]

output_file = 'gpt4_predictions.json'
with open(output_file, 'w') as f:
    json.dump(predictions, f, indent=4)

