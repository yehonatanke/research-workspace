import random

class TextNoiseMaker:
    """Class to add various types of noise to text data"""
    def __init__(self, noise_prob=0.3):
        self.noise_prob = noise_prob
    def add_noise(self, text):
        words = text.split()
        if len(words) <= 1:
            return text
        noisy_words = words.copy()
        # Duplicate random words (30% chance)
        if random.random() < self.noise_prob:
            word_to_duplicate = random.choice(words)
            insert_pos = random.randint(0, len(noisy_words))
            noisy_words.insert(insert_pos, word_to_duplicate)
        # Add random word from the sentence (30% chance)
        if random.random() < self.noise_prob and len(words) > 1:
            random_word = random.choice(words)
            insert_pos = random.randint(0, len(noisy_words))
            noisy_words.insert(insert_pos, random_word)
        # Shuffle order of some words (30% chance)
        if random.random() < self.noise_prob and len(noisy_words) > 2:
            start_idx = random.randint(0, len(noisy_words) - 2)
            end_idx = min(start_idx + random.randint(2, 4), len(noisy_words))
            subsequence = noisy_words[start_idx:end_idx]
            random.shuffle(subsequence)
            noisy_words[start_idx:end_idx] = subsequence
        return ' '.join(noisy_words) 