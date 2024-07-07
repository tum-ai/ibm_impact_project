import re
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

# method by Golkar et al. (https://arxiv.org/abs/2310.02989)

class RegressionTransformerTokenizer:
    
    def __init__(self, pretrained_path=None, vocab_files=None, save_file='tokenizer.json', number_token='[NUM]') -> None:
        """
        Initialize the RegressionTransformerTokenizer with a special token for numerical values.

        Args:
        number_token (str): Token to replace numerical values in the text with
        """
        self.number_token = number_token
        self.number_pattern = re.compile(r'^-?\d*\.?\d+([eE][-+]?\d+)?$')
        
        # to avoid training anew everytime:
        if pretrained_path:
            self.tokenizer = Tokenizer.from_file(pretrained_path)
        else:
            self.tokenizer = self.get_tokenizer(vocab_files, save_file)
    
    def parse_and_replace_numbers(self, xtext):
        """
        Parse a string and extract all numerical values. Then, replace them with a placeholder token.

        Args:
        x (str): Input string containing both numbers and text.

        Returns:
        tuple:
            xnum (list): A list of strings where each string is a numerical value extracted from the input.
            xtext (str): A new string where all numerical values in the input are replaced with the token '[NUM]'.
        """
        # Extract numerical values and normalize them to [-5, 5]
        xnums_strings = [num for num in self.number_pattern.findall(xtext)]
        
        for xnum in xnums_strings:
            xtext = xtext.replace(xnum, " ".join(self.numeric_string_to_tokens(xnum)))
        
        return xnums_strings, xtext
    
    def numeric_string_to_tokens(self, number):
        """ Convert numeric strings, including scientific notation, to tokens based on position. """
        tokens = []

        if number[0] == '-':
            tokens.append('-')
            number = number[1:]
        if number[0] == '+':
            tokens.append('+')
            number = number[1:]

        parts = re.split(r'([eE][-+]?\d+)', number)
        for part in parts:
            if part.startswith(('e', 'E')):
                exponent = part[1:]
                tokens.append('_e_')
                tokens.extend(numeric_string_to_tokens(exponent))
            else:
                if '.' in part:
                    integer_part, decimal_part = part.split('.')
                else:
                    integer_part, decimal_part = part, ''
                
                for index, char in enumerate(integer_part):
                    tokens.append(f"_{char}_{len(integer_part)-index-1}_")
                if '.' in part:
                    tokens.append('.')
                for index, char in enumerate(decimal_part):
                    tokens.append(f"_{char}_-{index + 1}_")

        return tokens

    def get_tokenizer(self, vocab_files, save_file):
        """
        Train a tokenizer and save it to a file.

        Args:
        vocab_files (list of str): List of paths to training corpus files.
        save_file (str): Path to save the trained tokenizer.

        Returns:
        Tokenizer: The trained tokenizer.
        """
        #special tokens (taken from xVal code)
        special_tokens = ["[END]", "[MASK]", "[PAD]", "[NUM]"]

        #train
        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_special_tokens(special_tokens)
        
        full_vocab = []
        if vocab_files is not None:
            for file_path in vocab_files:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        full_vocab.extend(line.strip().split())

        trainer = trainers.BpeTrainer(vocab=full_vocab, special_tokens=special_tokens)
        tokenizer.train(vocab_files, trainer)

        #post-processing
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()

        # save 
        tokenizer.save(save_file)
        
        return tokenizer
    
class RegressionTransformerEmbedding(nn.Module):
    
    def __init__(self, tokenizer, num_hidden, max_seq_len):
        """
        Initialize the RegressionTransformerEmbedding with a pre-trained model and embedding parameters.

        Args:
        model_name (str): The name of the pre-trained model to use.
        num_hidden (int): The size of the hidden layers in the model.
        max_seq_len (int): The maximum sequence length that the model can handle.
        """
        super(RegressionTransformerEmbedding, self).__init__()
        self.tokenizer = tokenizer
        self.num_hidden = num_hidden
        self.max_seq_len = max_seq_len
        
        # Embedding: provide information about the position of each token in the sequence
        # batch size 1 --> allows broadcasting to different batch sizes during training
        # each token embedding vector will have size self.num_hidden --> self.num_hidden depends on the
        #model that we use; the same for max_seq_len
        self.embeddings = nn.Embedding(self.tokenizer.get_vocab_size(), num_hidden)
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.num_hidden))
        
    
    def forward(self, text):
        numbers, replaced_text = self.tokenizer.parse_and_replace_numbers(text)
        tokenized = self.tokenizer.encode(replaced_text)
        input_ids = torch.tensor(tokenized.ids).unsqueeze(0) # batch size is 1; unsequeeze --> [seq_len] to [1, seq_len]

        embedded_text = self.word_embeddings(input_ids)

        return embedded_text

def main():
    text = "The price is 100 dollars and the discount is 20 percent. I'll buy 6 pieces. What about 2.3?"
    tokenizer_path = 'tokenizer.json'
    
    tokenizer = XValTokenizer(pretrained_path=None, vocab_files=['src/amc_extract.txt'], save_file=tokenizer_path)
    xnum, xtext = tokenizer.parse_and_replace_numbers(text)
    
    print(xnum)
    print(xtext)

if __name__ == "__main__":
    main()