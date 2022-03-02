from collections import defaultdict
import spacy
from tqdm import tqdm

class Tokenizer:
    def __init__(self, spacy_argument):
        # initialize tokenizer with spacy
        self.nlp = spacy.load(spacy_argument)
    
    def tokenize(self, text):
        lines = text.splitlines()
        doc = [[token.text for token in self.nlp.tokenizer(text.strip()) if not token.text.isnumeric()] for text in tqdm(lines)]
        return doc

class word_dict:
    def __init__(self, doc = None):
        self.vocab_size = 0
        self.word2idx = defaultdict(int)
        if doc:
            self.update(doc)
    
    def update(self, doc):
        # update word2idx with doc
        tokens = set()
        for line in doc:
            tokens.update(line)
        
        for token in tokens:
            if token not in self.word2idx:
                self.word2idx[token] = self.vocab_size
                self.vocab_size += 1
    
    def convert(self, doc):
        # convert doc to index
        doc_to_idx = [[self.word2idx[token] for token in line if token in self.word2idx] for line in doc]
        return doc_to_idx
