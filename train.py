import zipfile
import logging
import pickle
import torch
from utils import Tokenizer, word_dict

FILE_PATH = './wiki-bios.DEBUG.txt'
LANG = 'en_core_web_sm'
EMBEDDING_SIZE = 150
CONTEXT_SIZE = 6
NUM_EPOCH = 100
LEARNING_RATE = 0.01

def read_data(file_path = FILE_PATH):    
    with open(file_path, mode='r', encoding='utf-8') as fp:
        text = fp.read()
    return text


def preprocess(file_path = FILE_PATH):
    # returns the converted index form of the corpus and vocab_size

    # read in data
    text = read_data(file_path)
    print("read raw data")
    tokenizer = Tokenizer(LANG)
    word_dictionary = word_dict()

    # tokenize
    doc = tokenizer.tokenize(text)
    logging.info("after generate tokens from text")

    # save doc
    # with open(DOC_PATH, mode='wb') as fp:
    #     pickle.dump(doc, fp)
    # logging.info("tokenized documents saved!")
    # load doc
#     with open(DOC_PATH, 'rb') as fp:
#         doc = pickle.load(fp)

    word_dictionary.update(doc)
    logging.info("after generate dictionary")
    corpus = word_dictionary.convert(doc)

    return corpus, word_dictionary.vocab_size, word_dictionary


from model import Glove

def train_glove_model():
    # preprocess
    corpus, vocab_size, word_dict = preprocess(FILE_PATH)

    # specify device type
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init vector model
    logging.info("init model hyperparameter")
    model = Glove(EMBEDDING_SIZE, CONTEXT_SIZE, vocab_size)
    model.to(device)
    # get occurence matrix
    model.fit(corpus)

    model.train(NUM_EPOCH, device, learning_rate=LEARNING_RATE)
    return model, word_dict


if __name__ == '__main__':
    from tqdm import tqdm
    import torch
    from scipy.spatial.distance import cosine
    model, word_dict = train_glove_model()
    model.target_embeddings = model.center_embeddings.weight + model.context_embeddings.weight
    word2idx = word_dict.word2idx
    torch.save(word2idx, 'a.pt')
    torch.save(model.target_embeddings, 'tensor.pt')
    word2idx['April']
    def get_neighbors(model, word_to_index, target_word):
        """ 
        Finds the top 10 most similar words to a target word
        """
        outputs = []
        for word, index in tqdm(word_to_index.items(), total=len(word_to_index)):
            similarity = compute_cosine_similarity(model, word_to_index, target_word, word)
            result = {"word": word, "score": similarity}
            outputs.append(result)

        # Sort by highest scores
        neighbors = sorted(outputs, key=lambda o: o['score'], reverse=True)
        return neighbors[1:11]

    def compute_cosine_similarity(model, word_to_index, word_one, word_two):
        '''
        Computes the cosine similarity between the two words
        '''
        try:
            word_one_index = word_to_index[word_one]
            word_two_index = word_to_index[word_two]
        except KeyError:
            return 0
        
        # convert back to cpu
        model.target_embeddings = model.target_embeddings.to('cpu')

        embedding_one = model.target_embeddings[word_one_index]
        embedding_two = model.target_embeddings[word_two_index]
        similarity = 1 - abs(float(cosine(embedding_one.detach().numpy(),
                                        embedding_two.detach().numpy())))
        return similarity
    get_neighbors(model, word2idx, 'April')