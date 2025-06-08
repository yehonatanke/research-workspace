import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, t=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_cell = nn.RNNCell(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.grads = []
        self.t = t
    def forward(self, x, lengths):
        self.grads = []
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        h_t = torch.zeros(batch_size, self.rnn_cell.hidden_size, device=x.device)
        max_len = lengths.max().item()
        for i in range(max_len):
            mask = (i < lengths).float().unsqueeze(1)
            input_t = embedded[:, i, :]
            h_t_next = self.rnn_cell(input_t, h_t)
            h_t = mask * h_t_next + (1 - mask) * h_t
            if self.training and i >= max(0, max_len - self.t):
                h_t.register_hook(lambda grad: self.grads.append(grad.norm().item()))
            if i == max_len - self.t - 1 and max_len > self.t:
                h_t = h_t.detach()
        logits = self.fc(h_t)
        return logits
    def plot_grads(self, seq_len):
        plt.figure(figsize=(8, 6))
        plot_len = len(self.grads)
        if plot_len == 0:
            print("No gradients recorded.")
            return
        plt.plot(range(seq_len - plot_len, seq_len), self.grads, 'o-', label='Gradient Norm')
        plt.axvline(x=max(0, seq_len - self.t), color='r', linestyle='--', label=f'T - t = {max(0, seq_len - self.t)}\n(t={self.t})')
        plt.xlabel("Time Step")
        plt.ylabel("Gradient Norm")
        plt.title(f"Gradient Size for Sequence Length {seq_len}")
        plt.grid(True)
        plt.legend()
        plt.show()

class FlowControlRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        self.W_r = nn.Linear(input_dim, hidden_dim)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_t, h_prev):
        h_hat_t = self.tanh(self.W_h(x_t) + self.U_h(h_prev) + self.b_h)
        R_t = self.sigmoid(self.W_r(x_t) + self.b_r)
        h_t = h_hat_t * R_t
        return h_t

class FlowControlRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn_cell = FlowControlRNNCell(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        h_t = torch.zeros(batch_size, self.rnn_cell.W_h.out_features, device=x.device)
        max_len = lengths.max().item()
        for i in range(max_len):
            mask = (i < lengths).float().unsqueeze(1)
            input_t = embedded[:, i, :]
            h_t_next = self.rnn_cell(input_t, h_t)
            h_t = mask * h_t_next + (1 - mask) * h_t
        logits = self.fc(h_t)
        return logits 
