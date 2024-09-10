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

    def parse_number_result(self, prediction: List[str], label: List[str]) -> List[Tuple[float, float]]:
        number_results = [self.parse_number_result_per_sample(prediction[i], label[i]) for i in range(len(prediction))]

        return number_results

    def parse_number_result_per_sample(self, generated: str, reference: str):
        """
        Calculate the numerical difference between the numbers found in generated and reference texts.
        """
        gen_parts = generated.split("####")
        ref_parts = reference.split("####")

        if len(gen_parts)!=2 or len(ref_parts)!=2:
            return np.nan, np.nan
        
        try:
            gen_number = float(gen_parts[1].replace(",", ""))
            ref_number = float(ref_parts[1].replace(",", ""))
        except ValueError:
            return np.nan, np.nan
        
        if max(gen_number, ref_number)>1000000:
            print(gen_number, ref_number)

        return gen_number, ref_number

    def calculate_metrics(self, number_results, total_count):
        mae = np.mean([np.abs(result[0] - result[1]) for result in number_results if not np.isnan(result[0])])
        mse = np.mean([np.abs(result[0] - result[1]) ** 2 for result in number_results if not np.isnan(result[0])])
        r2 = 1 - np.nansum((number_results[:, 0] - number_results[:, 1]) ** 2) / np.nansum(
            (number_results[:, 1] - np.nanmean(number_results[:, 1])) ** 2)
        number_accuracy = np.mean(
            [np.isclose(result[0], result[1]) if not np.isnan(result[0]) else False for result in number_results])
        count_not_produced_valid_results = np.sum(np.isnan([result[0] for result in number_results]))
        average_count_not_produced_valid_results = count_not_produced_valid_results / total_count

        return mae, mse, r2, number_accuracy, count_not_produced_valid_results, average_count_not_produced_valid_results

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
        token_accuracy, token_accuracy_whole = self.compute_token_accuracy(generated_solutions, ground_truth_solutions)

        number_results = self.parse_number_result(generated_solutions, ground_truth_solutions)
        total_count = len(generated_solutions)

        batch_result = {
            'token_accuracy': token_accuracy,
            'token_accuracy_whole': token_accuracy_whole,
            'number_results': number_results,
            'total_count': total_count,
            "mae": np.mean([np.abs(result[0] - result[1]) for result in number_results if not np.isnan(result[0])]),
            'BLEU': bleu['score'],
            'ROUGE1': rouge['rouge1'],
            'ROUGE2': rouge['rouge2'],
            'ROUGEL': rouge['rougeL']
        }

        self.batch_stats.append(batch_result)

        if compute_result:
            total_count = sum([stat['total_count'] for stat in self.batch_stats])
            number_results = np.concatenate([stat['number_results'] for stat in self.batch_stats])
            (
                mae,
                mse,
                r2,
                number_accuracy,
                count_not_produced_valid_results,
                average_count_not_produced_valid_results
            ) = self.calculate_metrics(number_results, total_count)

            aggregated_result = {
                'token_accuracy': np.mean([stat['token_accuracy'] for stat in self.batch_stats]),
                'token_accuracy_whole': np.mean([stat['token_accuracy_whole'] for stat in self.batch_stats]),
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'number_accuracy': number_accuracy,
                "count_not_produced_valid_results": count_not_produced_valid_results,
                "average_count_not_produced_valid_results": average_count_not_produced_valid_results,
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
    json_file_path = '../predictions.json'

    output_directory = 'output'

    evaluate_json_file(json_file_path, batch_size=32, output_dir=output_directory)

