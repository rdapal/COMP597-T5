"""
Synthetic Data Generator

MilaBench generates synthetic data using:
- vocab_size: Model's vocabulary size (T5=32128, BERT=30522, GPT2=50257)
- train_length: Sequence length (seems to usually be 512)
- n: Number of unique samples (=batch_size)
- repeat: How many times to repeat the dataset

For T5 (AutoModelForSeq2SeqLM), we generate:
- input_ids: Random tokens in bounds of [0, vocab_size)
- labels: Random tokens in bounds of [0, vocab_size)

Our reference: https://github.com/mila-iqia/milabench/blob/master/benchmarks/huggingface/bench/synth.py
"""

import torch
from torch.utils.data import Dataset
import src.config as config
import logging

logger = logging.getLogger(__name__)


class SyntheticData(Dataset):
    """
    MilaBench-style synthetic data generator.
    
    Mirrors the exact structure from milabench/benchmarks/huggingface/bench/synth.py
    
    Parameters match MilaBench conventions:
    - n: Number of unique samples to generate
    - repeat: Number of times to repeat the dataset
    - Total samples = n * repeat
    """
    
    def __init__(
        self,
        vocab_size: int = 32128,    # T5's vocab size
        train_length: int = 512,     # Sequence length
        n: int = 4,                  # Number of unique samples (MilaBench default)
        repeat: int = 1000,          # Number of repeats
        seed: int = 42,              # For reproducibility
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            vocab_size: Size of vocabulary (T5=32128, BERT=30522, GPT2=50257)
            train_length: Length of each sequence
            n: Number of unique samples to pre-generate
            repeat: How many times to repeat the n samples
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.train_length = train_length
        self.n = n
        self.repeat = repeat
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        logger.info(
            f"Generating synthetic data: vocab_size={vocab_size}, "
            f"train_length={train_length}, n={n}, repeat={repeat}, "
            f"total_samples={n * repeat}"
        )
        
        # Pre-generate n unique samples (MilaBench style)
        # Each sample has input_ids and labels (for seq2seq like T5)
        self.data = []
        for _ in range(n):
            sample = {
                'input_ids': torch.randint(0, vocab_size, (train_length,)),
                'attention_mask': torch.ones(train_length, dtype=torch.long),
                'labels': torch.randint(0, vocab_size, (train_length,)),
            }
            self.data.append(sample)
        
        logger.info(f"Synthetic dataset ready: {len(self)} total samples")
    
    def __len__(self) -> int:
        """Total length = n * repeat"""
        return self.n * self.repeat
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get sample at index.
        
        MilaBench cycles through n samples using modulo.
        """
        return self.data[idx % self.n]


def load_data(conf: config.Config) -> Dataset:
    """
    Load synthetic data based on configuration.
    
    Parses the split string to determine total samples:
    - "train[:500]" -> 500 total samples
    
    Args:
        conf: Configuration object
        
    Returns:
        SyntheticData dataset instance
    """
    # T5-specific parameters
    vocab_size = 32128   # T5's vocabulary size
    train_length = 512   # Matches the sequence length
    n = 4                # Number of unique samples (MILABENCH default = batch_size)
    
    # Parse total samples from split string
    total_samples = 1000  # default
    split = getattr(conf.data_configs.dataset, 'split', None)
    
    if split:
        try:
            split_clean = str(split).strip('"').strip("'")
            if "[:" in split_clean and "]" in split_clean:
                num_str = split_clean.split("[:")[1].split("]")[0]
                total_samples = int(num_str)
                logger.info(f"Parsed total_samples={total_samples} from split='{split}'")
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse split '{split}': {e}")
    
    # Calculate repeat to achieve desired total
    repeat = max(1, total_samples // n)
    
    logger.info(
        f"T5 Synthetic Data Config: "
        f"vocab_size={vocab_size}, train_length={train_length}, "
        f"n={n}, repeat={repeat}, total={n * repeat}"
    )
    
    return SyntheticData(
        vocab_size=vocab_size,
        train_length=train_length,
        n=n,
        repeat=repeat,
        seed=42,
    )
