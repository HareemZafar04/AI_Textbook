---
sidebar_label: Language Models
---

# Language Models

Language models are fundamental components in Natural Language Processing that assign probabilities to sequences of words. They are used to predict the likelihood of a word given the preceding context or to generate new text that resembles human language.

## What is a Language Model?

A language model estimates the probability of a sequence of words occurring in a given language. Formally, for a sequence of words W = (w₁, w₂, ..., wₙ), a language model estimates:

P(W) = P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁, w₂) × ... × P(wₙ|w₁, w₂, ..., wₙ₋₁)

This probability distribution over sequences allows the model to:
- Predict the next word in a sequence
- Generate new text
- Evaluate the fluency of a sentence
- Power applications like machine translation and speech recognition

## Types of Language Models

### 1. N-gram Models

N-gram models are classical statistical models that predict the next word based on the previous N-1 words. The "N" refers to the number of previous words used as context.

#### Unigram Model (N=1)
Considers each word independently:
P(wᵢ) = count(wᵢ) / total word count

```python
from collections import Counter
import math

class UnigramModel:
    def __init__(self, texts):
        # Tokenize and flatten all texts
        all_tokens = []
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
        
        # Count word frequencies
        self.word_counts = Counter(all_tokens)
        self.total_words = len(all_tokens)
        
    def word_probability(self, word):
        return self.word_counts.get(word, 0) / self.total_words
    
    def sentence_probability(self, sentence):
        tokens = sentence.lower().split()
        prob = 1.0
        for token in tokens:
            prob *= self.word_probability(token)
        return prob

# Example usage
texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are pets"
]

model = UnigramModel(texts)
prob = model.sentence_probability("the cat ran")
print(f"Probability: {prob}")
```

#### Bigram Model (N=2)
Considers the previous word:
P(wᵢ|wᵢ₋₁) = count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)

```python
class BigramModel:
    def __init__(self, texts):
        self.bigram_counts = Counter()
        self.word_counts = Counter()
        
        for text in texts:
            tokens = text.lower().split()
            for i in range(len(tokens)):
                self.word_counts[tokens[i]] += 1
                
                if i > 0:
                    bigram = (tokens[i-1], tokens[i])
                    self.bigram_counts[bigram] += 1
    
    def bigram_probability(self, prev_word, current_word):
        bigram = (prev_word, current_word)
        if self.word_counts[prev_word] == 0:
            return 0
        return self.bigram_counts.get(bigram, 0) / self.word_counts[prev_word]
    
    def sentence_probability(self, sentence):
        tokens = sentence.lower().split()
        if len(tokens) == 0:
            return 1.0
        
        prob = self.word_counts[tokens[0]] / sum(self.word_counts.values())  # P(first_word)
        
        for i in range(1, len(tokens)):
            prob *= self.bigram_probability(tokens[i-1], tokens[i])
        
        return prob

# Example usage
model = BigramModel(texts)
prob = model.sentence_probability("the cat sat")
print(f"Bigram probability: {prob}")
```

#### Trigram Model (N=3) and Beyond
Considers the previous two words (for trigrams):
P(wᵢ|wᵢ₋₂, wᵢ₋₁) = count(wᵢ₋₂, wᵢ₋₁, wᵢ) / count(wᵢ₋₂, wᵢ₋₁)

### 2. Neural Language Models

Neural language models use neural networks to capture complex patterns in text.

#### Recurrent Neural Networks (RNNs) for Language Modeling
RNNs process sequences by maintaining a hidden state that captures information about the sequence seen so far.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNNLanguageModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        else:
            h0 = hidden
        
        # Embedding
        embedded = self.dropout(self.embedding(x))
        
        # RNN
        rnn_out, hidden_out = self.rnn(embedded, h0)
        
        # Linear layer to vocab size
        output = self.fc(self.dropout(rnn_out))
        
        return output, hidden_out

# Example usage would involve training with sequences of text data
```

#### Long Short-Term Memory (LSTM) Models
LSTMs address the vanishing gradient problem in RNNs, allowing for better modeling of long-term dependencies.

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMLanguageModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        else:
            h0, c0 = hidden
        
        # Embedding
        embedded = self.dropout(self.embedding(x))
        
        # LSTM
        lstm_out, hidden_out = self.lstm(embedded, (h0, c0))
        
        # Linear layer to vocab size
        output = self.fc(self.dropout(lstm_out))
        
        return output, hidden_out
```

### 3. Transformer-Based Models

Transformers use self-attention mechanisms to process input sequences in parallel, making them highly efficient and effective.

#### Attention Mechanism
The attention mechanism allows the model to focus on different parts of the input when generating each output token.

```python
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state for each time step
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate hidden and encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 2)))
        
        # Calculate attention energies
        energy = energy.transpose(2, 1)  # [batch_size, hidden_dim, seq_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        energy = torch.bmm(v, energy).squeeze(1)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        return torch.softmax(energy, dim=1)

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerLanguageModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, tgt, tgt_mask=None):
        # tgt: [batch_size, seq_len]
        tgt_emb = self.dropout(self.embedding(tgt) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer(tgt_emb, tgt_emb, tgt_mask)
        output = self.fc_out(output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## Pre-trained Language Models

### 1. BERT (Bidirectional Encoder Representations from Transformers)
- Contextual bidirectional model
- Uses masked language modeling during training
- Excellent for understanding context

### 2. GPT (Generative Pre-trained Transformer)
- Autoregressive language model
- Excels at text generation
- Based on decoder-only transformer architecture

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=50):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.8
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
# result = generate_text("The future of artificial intelligence", max_length=100)
# print(result)
```

## Evaluating Language Models

### 1. Perplexity
Perplexity is the standard metric for evaluating language models:

```python
def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            
            # Calculate loss
            loss = nn.CrossEntropyLoss(reduction='sum')(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity
```

### 2. BLEU Score
For generative models, BLEU measures how similar generated text is to reference text.

## Applications of Language Models

1. **Text Generation**: Creating new text that resembles training data
2. **Machine Translation**: Converting text from one language to another
3. **Text Summarization**: Condensing long documents into shorter summaries
4. **Question Answering**: Finding answers within a text
5. **Sentiment Analysis**: Determining the emotional tone of text
6. **Named Entity Recognition**: Identifying entities in text
7. **Chatbots and Virtual Assistants**: Providing conversational interfaces

## Challenges in Language Modeling

1. **Computational Complexity**: Training large models requires significant computational resources
2. **Data Requirements**: High-quality, diverse training data is essential
3. **Bias and Fairness**: Models can perpetuate biases present in training data
4. **Context Length**: Handling long sequences of text remains challenging
5. **Interpretability**: Understanding how models make decisions is difficult

Language models have evolved from simple n-gram models to sophisticated neural architectures, enabling increasingly human-like text generation and understanding capabilities.