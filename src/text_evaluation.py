import re
import numpy as np
import evaluate
import nltk
import json
from typing import List, Dict, Tuple
from src.tokenizer.abstract_tokenizer import NUMBER_REGEX

class TextEvaluation:
    """
    Evaluates a model's generated text against ground truth text using a variety of metrics.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.rouge_metric = evaluate.load("rouge")
        self.bleu_metric = evaluate.load("sacrebleu")
        nltk.download('punkt')
        self.batch_stats = []
        self.eval_count = 0

    def compute_bleu(self, generated: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU score for the generated text vs the ground truth text.
        """
        references = [[ref.strip()] for ref in references]
        generated = [gen.strip() for gen in generated]
        bleu_result = self.bleu_metric.compute(predictions=generated, references=references)
        return bleu_result

    def compute_rouge(self, generated: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE score for the generated text vs the ground truth text.
        """
        generated = ["\n".join(nltk.sent_tokenize(gen.strip())) for gen in generated]
        references = ["\n".join(nltk.sent_tokenize(ref.strip())) for ref in references]
        rouge_result = self.rouge_metric.compute(predictions=generated, references=references, use_stemmer=True)
        return rouge_result

    def compute_mse(self, generated: List[str], references: List[str]) -> float:
        """
        Compute the Mean Squared Error (MSE) between the numbers extracted from the generated and reference texts.
        """
        return np.nanmean([self.calculate_number_loss_per_sample(generated[i], references[i], 2) for i in range(len(generated))])

    def compute_mae(self, generated: List[str], references: List[str]) -> float:
        """
        Compute the Mean Absolute Error (MAE) between the numbers extracted from the generated and reference texts.
        """
        return np.nanmean([self.calculate_number_loss_per_sample(generated[i], references[i], 1) for i in range(len(generated))])

    def calculate_number_loss_per_sample(self, generated: str, reference: str, order: int):
        """
        Calculate the numerical difference between the numbers found in generated and reference texts.
        """
        gen_parts = generated.split("####")
        ref_parts = reference.split("####")

        if len(gen_parts)!=2 or len(ref_parts)!=2:
            return np.nan
        
        try:
            gen_number = float(gen_parts[1].replace(",", ""))
            ref_number = float(ref_parts[1].replace(",", ""))
        except ValueError:
            return np.nan
        
        if max(gen_number, ref_number)>1000000:
            print(gen_number, ref_number)

        return np.abs(gen_number - ref_number) ** order

    def compute_token_accuracy(self, generated: List[str], references: List[str]) -> Tuple[float, float]:
        """
        Compute token-level accuracy and whole-sentence token accuracy.
        """
        correct_tokens, correct_whole = 0, 0
        total_tokens = 0

        for gen, ref in zip(generated, references):
            gen_tokens = gen.split()
            ref_tokens = ref.split()

            total_tokens += len(ref_tokens)
            correct_tokens += sum(1 for gt, rt in zip(gen_tokens, ref_tokens) if gt == rt)
            if gen_tokens == ref_tokens:
                correct_whole += 1

        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        token_accuracy_whole = correct_whole / len(generated) if generated else 0

        return token_accuracy, token_accuracy_whole

    def __call__(self, batch_data: List[Dict[str, str]], compute_result: bool = True) -> Dict[str, float]:
        """
        Evaluate the metrics on a batch of generated and ground truth text.
        """
        generated_solutions = [entry['generated_solution'] for entry in batch_data]
        ground_truth_solutions = [entry['ground_truth_solution'] for entry in batch_data]

        bleu = self.compute_bleu(generated_solutions, ground_truth_solutions)
        rouge = self.compute_rouge(generated_solutions, ground_truth_solutions)
        mse = self.compute_mse(generated_solutions, ground_truth_solutions)
        mae = self.compute_mae(generated_solutions, ground_truth_solutions)
        token_accuracy, token_accuracy_whole = self.compute_token_accuracy(generated_solutions, ground_truth_solutions)

        batch_result = {
            'token_accuracy': token_accuracy,
            'token_accuracy_whole': token_accuracy_whole,
            'MSE': mse,
            'MAE': mae,
            'BLEU': bleu['score'],
            'ROUGE1': rouge['rouge1'],
            'ROUGE2': rouge['rouge2'],
            'ROUGEL': rouge['rougeL']
        }

        self.batch_stats.append(batch_result)

        if compute_result:
            aggregated_result = {
                'token_accuracy': np.mean([stat['token_accuracy'] for stat in self.batch_stats]),
                'token_accuracy_whole': np.mean([stat['token_accuracy_whole'] for stat in self.batch_stats]),
                'MSE': np.nanmean([stat['MSE'] for stat in self.batch_stats]),
                'MAE': np.nanmean([stat['MAE'] for stat in self.batch_stats]),
                'BLEU': np.mean([stat['BLEU'] for stat in self.batch_stats]),
                'ROUGE1': np.mean([stat['ROUGE1'] for stat in self.batch_stats]),
                'ROUGE2': np.mean([stat['ROUGE2'] for stat in self.batch_stats]),
                'ROUGEL': np.mean([stat['ROUGEL'] for stat in self.batch_stats]),
            }

            self.batch_stats = []
            return aggregated_result

        return batch_result

def read_json_file(file_path: str) -> List[dict]:
    """
    Reads the JSON file and returns the array of dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def batchify_data(data: List[dict], batch_size: int = 32) -> List[List[dict]]:
    """
    Batches the data into chunks of the specified batch size.
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def evaluate_json_file(file_path: str, batch_size: int = 32, output_dir: str = 'output'):
    """
    Reads the JSON file, batches the data, and passes it to TextEvaluation for evaluation.
    """
    text_eval = TextEvaluation(output_dir=output_dir)

    data = read_json_file(file_path)

    batches = batchify_data(data, batch_size)

    for batch_num, batch in enumerate(batches):
        print(f"Evaluating batch {batch_num + 1}/{len(batches)}...")

        result = text_eval(batch, compute_result=(batch_num == len(batches) - 1))  # Compute result on the last batch
        if result:
            print(f"Batch {batch_num + 1} result: {result}")

if __name__ == "__main__":
    json_file_path = 'predictions.json'

    output_directory = 'output'

    evaluate_json_file(json_file_path, batch_size=32, output_dir=output_directory)

