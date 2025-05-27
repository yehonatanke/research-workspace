from torch.utils.data import Dataset
from collections import Counter

class TextDataset(Dataset):
    """Dataset for text denoising"""
    def __init__(self, texts, vocab_to_idx, max_length=64, noise_maker=None):
        self.texts = texts
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
        self.noise_maker = noise_maker
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        clean_text = self.texts[idx]
        if self.noise_maker:
            noisy_text = self.noise_maker.add_noise(clean_text)
        else:
            noisy_text = clean_text
        clean_indices = self.text_to_indices(clean_text)
        noisy_indices = self.text_to_indices(noisy_text)
        return {
            'noisy': torch.tensor(noisy_indices, dtype=torch.long),
            'clean': torch.tensor(clean_indices, dtype=torch.long),
            'noisy_text': noisy_text,
            'clean_text': clean_text
        }
    def text_to_indices(self, text):
        words = text.lower().split()
        indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in words]
        if len(indices) < self.max_length:
            indices.extend([self.vocab_to_idx['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        return indices

def build_vocabulary(texts, max_vocab_size=5000):
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    most_common = word_counts.most_common(max_vocab_size - 3)
    vocab_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2}
    idx_to_vocab = {0: '<PAD>', 1: '<UNK>', 2: '<START>'}
    for idx, (word, _) in enumerate(most_common, start=3):
        vocab_to_idx[word] = idx
        idx_to_vocab[idx] = word
    return vocab_to_idx, idx_to_vocab 