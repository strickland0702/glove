from collections import Counter, defaultdict
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_window(sentence, start_index, end_index):
    # returns sentence[start_index : end_index + 1]
    sentence_length = len(sentence) + 1
    window = sentence[max(start_index, 0): min(end_index, sentence_length) + 1]
    return window

def context_window(sentence, window_size):
    # returns a window for each word in the sentence
    for i, word in enumerate(sentence):
        start_index = i - window_size
        end_index = i + window_size
        left_context = get_window(sentence, start_index, i - 1)
        right_context = get_window(sentence, i + 1, end_index)
        yield (left_context, word, right_context)


class GloveDataset(Dataset):
    def __init__(self, coocurrence_matrix):
        print(len(coocurrence_matrix))
        self.coocurrence_matrix = coocurrence_matrix
    def __getitem__(self, index):
        return self.coocurrence_matrix[index]
    def __len__(self):
        return len(self.coocurrence_matrix)


class Glove(nn.Module):
    def __init__(self, embedding_size, window_size, vocab_size, min_freq = 3, x_max = 100, alpha = 3/4):
        super().__init__()

        self.embedding_size = embedding_size
        self.window_size = window_size
        
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_freq = min_freq
        self.x_max = x_max

        self.center_embeddings = nn.Embedding(vocab_size, embedding_size).type(torch.float64)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_size).type(torch.float64)
        self.center_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self.context_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        
        for params in self.parameters():
            init.uniform_(params, a=-1, b=1)

    def fit(self, corpus):
        # builds the cooccurence matrix
        word_counter = Counter()
        cooccurence = defaultdict(float)
        for sentence in tqdm(corpus):
            word_counter.update(sentence)
            for left_context, word, right_context in context_window(sentence, self.window_size):
                for i, context_word in enumerate(left_context[::-1]):
                    cooccurence[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(right_context):
                    cooccurence[(word, context_word)] += 1 / (i + 1)

        tokens = {word for word, count in word_counter.most_common(self.vocab_size) if count >= self.min_freq}
        print('tokenlength', len(tokens))
        coocurrence_matrix = [(words[0], words[1], count)
                              for words, count in cooccurence.items()
                              if words[0] in tokens and words[1] in tokens]
        self.dataset = GloveDataset(coocurrence_matrix)
    
    def loss(self, center_idx, context_idx, coocurrence_count):
        
        center_embedding = self.center_embeddings(center_idx)
        context_embedding = self.context_embeddings(context_idx)
        center_bias = self.center_biases(center_idx)
        context_bias = self.context_biases(context_idx)

        # count weight factor
        fX = torch.pow(coocurrence_count / self.x_max, self.alpha)
        fX[fX > 1] = 1

        embedding_products = torch.sum(center_embedding * context_embedding, dim=1)
        log_cooccurrences = torch.log(coocurrence_count)

        loss_val = (embedding_products + center_bias +
                         context_bias - log_cooccurrences) ** 2

        loss_val = fX * loss_val
        mean_loss = torch.mean(loss_val)
        return mean_loss

    
    
    def train(self, num_epoch, device, batch_size=25, learning_rate=0.05, print_interval = 50):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        glove_dataloader = DataLoader(self.dataset, batch_size, shuffle = True)
        total_loss = 0

        for epoch in tqdm(range(num_epoch)):
            for idx, batch in enumerate(glove_dataloader):
                optimizer.zero_grad()

                i_s, j_s, counts = batch
                i_s = i_s.to(device)
                j_s = j_s.to(device)
                counts = counts.to(device)
                loss = self.loss(i_s, j_s, counts)
                
                total_loss += loss.item()
                if idx % print_interval == 0:
                    avg_loss = total_loss / print_interval
                    print("epoch: {}, current step: {}, average loss: {}".format(
                        epoch, idx, avg_loss))
                    total_loss = 0

                loss.backward()
                optimizer.step()

        print("finish glove vector training")
    
    def get_embedding(self, tokens):
        tokens = torch.tensor(tokens, device = device)
        return self.center_embeddings(tokens) + self.context_embeddings(tokens)