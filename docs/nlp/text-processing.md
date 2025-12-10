---
sidebar_label: Text Processing Techniques
---

# Text Processing Techniques

Text processing forms the foundation of Natural Language Processing applications. This section explores the essential techniques used to clean, transform, and prepare textual data for analysis and modeling.

## Text Preprocessing Pipeline

The typical text preprocessing pipeline involves multiple sequential steps to convert raw text into a format suitable for computational analysis:

1. **Text Cleaning**: Removing unwanted characters and formatting
2. **Normalization**: Converting text to a standard format
3. **Tokenization**: Breaking text into individual units
4. **Noise Reduction**: Removing irrelevant information
5. **Feature Engineering**: Converting text to numerical representations

## 1. Text Cleaning

Text cleaning involves removing unwanted characters, formatting, and artifacts from the raw text:

### Removing Special Characters and Numbers
```python
import re
import string

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (for social media text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example
raw_text = "Check out this website: https://example.com! It's amazing @user #nlp"
cleaned_text = clean_text(raw_text)
print(cleaned_text)  # "Check out this website  Its amazing  nlp"
```

### Handling HTML Entities and Tags
```python
from html import unescape
import re

def remove_html_tags(text):
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    clean_text = unescape(clean_text)
    
    return clean_text

# Example
html_text = "This is &lt;em&gt;important&lt;/em&gt; content."
cleaned = remove_html_tags(html_text)
print(cleaned)  # "This is <em>important</em> content."
```

## 2. Text Normalization

Text normalization standardizes text to reduce variations in the same concept:

### Case Normalization
```python
def normalize_case(text):
    # Convert to lowercase
    return text.lower()

# Example
text = "This is MIXED case text."
normalized = normalize_case(text)
print(normalized)  # "this is mixed case text."
```

### Accent Removal
```python
import unicodedata

def remove_accents(text):
    # Normalize to NFD (decomposed form)
    nfd = unicodedata.normalize('NFD', text)
    # Filter out combining characters (accents)
    without_accents = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    return without_accents

# Example
text = "Café résumé naïve"
cleaned = remove_accents(text)
print(cleaned)  # "Cafe resume naive"
```

### Expanding Contractions
```python
import re

def expand_contractions(text):
    # Common contractions with their expansions
    contractions = {
        "don't": "do not",
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    
    # Create regex pattern for contractions
    pattern = re.compile('|'.join(re.escape(key) for key in contractions.keys()))
    
    # Replace contractions
    expanded = pattern.sub(lambda match: contractions[match.group(0)], text)
    return expanded

# Example
text = "I can't believe it's not butter. You're awesome!"
expanded = expand_contractions(text)
print(expanded)  # "I cannot believe it is not butter. You are awesome!"
```

## 3. Tokenization

Tokenization is the process of breaking text into smaller units (tokens):

### Sentence Tokenization
```python
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def sentence_tokenize(text):
    sentences = sent_tokenize(text)
    return sentences

# Example
text = "Hello world. This is a sentence. Here's another one!"
sentences = sentence_tokenize(text)
print(sentences)  # ['Hello world.', 'This is a sentence.', "Here's another one!"]
```

### Word Tokenization
```python
from nltk.tokenize import word_tokenize

def word_tokenize_custom(text):
    tokens = word_tokenize(text)
    return tokens

# Without NLTK (simple split)
def simple_word_tokenize(text):
    # Split by whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# Example
text = "Hello, world! This is a sample text."
tokens = word_tokenize_custom(text)
print(tokens)  # ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', '.']

simple_tokens = simple_word_tokenize(text)
print(simple_tokens)  # ['Hello', 'world', 'This', 'is', 'a', 'sample', 'text']
```

### Subword Tokenization
```python
# Using Hugging Face tokenizers for BPE (Byte-Pair Encoding)
from transformers import AutoTokenizer

def subword_tokenize(text, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    return tokens

# Example
text = "Tokenization is important for NLP."
tokens = subword_tokenize(text)
print(tokens)  # ['token', '##ization', 'is', 'important', 'for', 'nl', '##p', '.']
```

## 4. Stop Words Removal

Stop words are common words that often carry little semantic meaning:

```python
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Example
tokens = ['this', 'is', 'a', 'sample', 'text', 'with', 'stop', 'words']
filtered = remove_stopwords(tokens)
print(filtered)  # ['sample', 'text', 'stop', 'words']
```

## 5. Stemming and Lemmatization

### Stemming
Stemming reduces words to their root form by removing suffixes:

```python
from nltk.stem import PorterStemmer

def stem_words(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Example
tokens = ['running', 'runs', 'ran', 'easily', 'fairly']
stemmed = stem_words(tokens)
print(stemmed)  # ['run', 'run', 'ran', 'easili', 'fairli']
```

### Lemmatization
Lemmatization reduces words to their dictionary form (lemma):

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(token)) 
        for token in tokens
    ]
    return lemmatized_tokens

# Example
tokens = ['running', 'runs', 'ran', 'easily', 'fairly']
lemmatized = lemmatize_words(tokens)
print(lemmatized)  # ['running', 'run', 'run', 'easily', 'fairly']
```

## 6. Advanced Text Processing

### N-gram Generation
```python
from nltk import ngrams

def generate_ngrams(tokens, n):
    n_grams = list(ngrams(tokens, n))
    return [' '.join(gram) for gram in n_grams]

# Example
tokens = ['natural', 'language', 'processing', 'is', 'important']
bigrams = generate_ngrams(tokens, 2)
trigrams = generate_ngrams(tokens, 3)

print(bigrams)    # ['natural language', 'language processing', 'processing is', 'is important']
print(trigrams)   # ['natural language processing', 'language processing is', 'processing is important']
```

### Text Normalization Pipeline
```python
def text_normalization_pipeline(text):
    # Clean text
    cleaned_text = clean_text(text)
    
    # Normalize case
    normalized_text = normalize_case(cleaned_text)
    
    # Tokenize
    tokens = word_tokenize(normalized_text)
    
    # Remove stopwords
    filtered_tokens = remove_stopwords(tokens)
    
    # Lemmatize
    lemmatized_tokens = lemmatize_words(filtered_tokens)
    
    # Remove empty tokens and rejoin
    final_tokens = [token for token in lemmatized_tokens if token.strip()]
    
    return ' '.join(final_tokens)

# Example
text = "This is a SAMPLE text with various elements! Don't miss it."
processed = text_normalization_pipeline(text)
print(processed)  # "sample text various element do not miss"
```

## 7. Handling Special Cases

### Handling Negations
```python
def handle_negation(tokens):
    """Preserve negation patterns"""
    i = 0
    processed_tokens = []
    
    while i < len(tokens):
        token = tokens[i]
        
        if token == 'not' and i + 1 < len(tokens):
            # Combine with next token
            processed_tokens.append(f"not_{tokens[i+1]}")
            i += 2  # Skip next token
        else:
            processed_tokens.append(token)
            i += 1
    
    return processed_tokens

# Example
tokens = ['I', 'am', 'not', 'happy', 'about', 'this']
negation_handled = handle_negation(tokens)
print(negation_handled)  # ['I', 'am', 'not_happy', 'about', 'this']
```

### Spell Correction
```python
from textblob import TextBlob

def correct_spelling(text):
    blob = TextBlob(text)
    corrected = blob.correct()
    return str(corrected)

# Example
text = "I havv goood speling"
corrected = correct_spelling(text)
print(corrected)  # "I have good spelling"
```

## Best Practices for Text Processing

1. **Preserve Original Data**: Keep a copy of the original text when possible
2. **Consider Context**: Some processing steps might remove important context
3. **Domain-Specific Processing**: Adapt techniques to your specific domain
4. **Iterative Approach**: Start simple and add complexity as needed
5. **Validation**: Always validate the results of your preprocessing steps
6. **Performance**: Consider computational efficiency for large datasets

These text processing techniques form the backbone of NLP applications, preparing raw text for downstream tasks like classification, sentiment analysis, and information extraction.