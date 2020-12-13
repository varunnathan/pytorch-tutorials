import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as torch_optim
from torchtext.data import Field, NestedField, LabelField, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab
from torchcrf import CRF
from tqdm import tqdm


class Corpus(object):
    def __init__(self, input_folder, min_word_freq, batch_size):
        self.word_field = Field(lower=True)
        self.tag_field = Field(unk_token=None)
        self.char_nested_field = Field(tokenize=list)
        self.char_field = NestedField(self.char_nested_field)
        self.train_ds, self.val_ds, self.test_ds = SequenceTaggingDataset.splits(
            path=input_folder,
            train='train.tsv',
            validation='val.tsv',
            test='test.tsv',
            fields=((('word', 'char'), (self.word_field, self.char_field)), ('tag', self.tag_field))
        )

        self.word_field.build_vocab(self.train_ds.word, min_freq=min_word_freq)
        self.tag_field.build_vocab(self.train_ds.tag)
        self.char_field.build_vocab(self.train_ds.char)

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_ds, self.val_ds, self.test_ds),
            batch_size=batch_size
        )

        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]


class CorpusPretrained(object):
    def __init__(self, input_folder, embedding_dim, min_word_freq, max_vocab,
                 batch_size, word_counter=None, word_vector=None):
        self.word_field = Field(lower=True)
        self.tag_field = Field(unk_token=None)
        self.train_ds, self.val_ds, self.test_ds = SequenceTaggingDataset.splits(
            path=input_folder,
            train='train.tsv',
            validation='val.tsv',
            test='test.tsv',
            fields=(('word', self.word_field), ('tag', self.tag_field))
        )
        self.embedding_dim = embedding_dim
        self.word_counter = word_counter
        self.word_vector = word_vector
        if word_counter is not None:
            self.word_field.vocab = Vocab(word_counter, min_freq=min_word_freq)
            vectors = []
            for word, idx in self.word_field.vocab.stoi.items():
                if word in self.word_vector.keys():
                    vectors.append(torch.tensor(self.word_vector[word].tolist()))
                else:
                    vectors.append(torch.zeros(self.embedding_dim))
            self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                vectors=vectors,
                dim=self.embedding_dim
            )
        else:
            self.word_field.build_vocab(self.train_ds.word, min_freq=min_word_freq)
        self.tag_field.build_vocab(self.train_ds.tag)
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_ds, self.val_ds, self.test_ds),
            batch_size=batch_size
        )
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]


corpus = Corpus(inp, 3, 64)
print('# train sentences: ', len(corpus.train_ds))


class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers, bidirectional, emb_dropout, dropout, word_pad_idx):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(num_embeddings=input_dim,
                                embedding_dim=embedding_dim,
                                padding_idx=word_pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if num_lstm_layers > 1 else 0)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, sentence):
        # sentence = (sent_len, batch_size)
        embedded = self.emb_dropout(self.emb(sentence)) # (sent_len, batch_size, embedding_dim)
        hidden, _ = self.lstm(embedded) # (sent_len, batch_size, hidden_dim*2)
        output = self.fc(self.fc_dropout(hidden)) # (sent_len, batch_size, output_dim)
        return output

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def init_embeddings(self, word_pad_idx, pretrained=None, freeze=True):
        self.emb.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
        if pretrained is not None:
            self.emb = nn.Embedding.from_pretrained(
                embeddings=pretrained, padding_idx=word_pad_idx, freeze=freeze
            )

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMCRF(nn.Module):
    def __init__(self, input_dim, char_input_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, embedding_dim, char_embedding_dim,
                 word_pad_idx, char_pad_idx, tag_pad_idx, emb_dropout, num_cnn_filters,
                 cnn_kernel_size, cnn_dropout, lstm_dropout, fc_dropout):
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=embedding_dim,
            padding_idx=word_pad_idx)
        self.char_embedding = nn.Embedding(
            num_embeddings=char_input_dim, embedding_dim=char_embedding_dim,
            padding_idx=char_pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.char_cnn = nn.Conv1D(
            in_channels=char_embedding_dim,
            out_channels= char_embedding_dim * num_cnn_filters,
            kernel_size=cnn_kernel_size, groups=char_embedding_dim)
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + (char_embedding_dim * num_cnn_filters),
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=lstm_dropout
        )
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.tag_pad_idx = tag_pad_idx
        self.crf = CRF(num_tags=output_dim)

        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, words, chars, tags=None):
        # words = (sent_len, batch_size)
        # chars = (batch_size, sent_len, word_len)
        # tags = (sent_len, batch_size)
        emb = self.emb_dropout(self.embedding(words))   # (sent_len, batch_size, embedding_dim)
        char_emb = self.emb_dropout(self.char_embedding(chars)) # (batch_size, sent_len, word_len, char_embedding_dim)
        batch_size, sent_len, word_len, char_embedding_dim = char_emb.shape
        char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels)
        for sent_i in range(sent_len):
            char_emb_out = char_emb[:, sent_i, :, :]
            char_emb_out_p = char_cnn_out.permute(0, 2, 1)  # (batch_size, char_embedding_dim, word_len)
            char_cnn_out = self.char_cnn(char_emb_out_p)    # (batch_size, char_embedding_dim*out_channels, word_len-kernal_size+1)
            char_cnn_max_out[:, sent_i, :] = torch.max(char_cnn_out, dim=2)
        char_cnn = self.cnn_dropout(char_cnn_max_out)   # (batch_size, sent_len, out_channels)
        char_cnn_p = char_cnn.permute(1, 0, 2)
        word_features = torch.cat((emb, char_cnn_p), dim=2)
        lstm_out, _ = self.lstm(word_features)  # (sent_len, batch_size, embedding_dim+out_channels)
        fc_out = self.fc(self.fc_dropout(lstm_out)) # (sent_len, batch_size, output_dim)

        if tags is not None:
            mask = tags != self.tag_pad_idx
            crf_out = self.crf.decode(fc_out, mask=mask)
            crf_loss = -self.crf(fc_out, tags=tags, mask=mask)
        else:
            crf_out = self.crf.decode(fc_out)
            crf_loss = None
        return crf_out, crf_loss

    def init_embeddings(self, pretrained, freeze, word_pad_idx):
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
        if pretrained:
            self.embedding = nn.Embedding.from_pretrained(
                embedding=pretrained, freeze=freeze, padding_idx=word_pad_idx
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def calc_accuracy(preds, true_tags):
    preds_flat = preds.view(-1)
    true_tags_flat = true_tags.view(-1)
    correct = [pred == tag for pred, tag in zip(preds_flat, true_tags_flat)]
    return sum(correct) / len(correct) if len(correct) > 0 else 0


def train(data, optimizer, model, tag_pad_idx, batch_size):
    total_acc, total_loss = 0, 0
    model.train()
    with tqdm(total = len(data.train_iter) // batch_size) as pbar:
        for batch in data.train_iter:
            words, chars, tags = batch.word, batch.char, batch.tag
            optimizer.zero_grad()
            batch_pred, batch_loss = model(words, chars, tags)
            tags_lst = [[tag for tag in sent_tag if tag != tag_pad_idx]
                        for sent_tag in tags.permute(1, 0).tolist()]
            batch_acc = calc_accuracy(batch_pred, tags_lst)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss
            total_acc += batch_acc
            pbar.update(1)

    return total_loss / len(data.train_iter), total_acc / len(data.train_iter)


def inference(model, data, sentence, true_tags=None):
    model.eval()
    nlp = Indonesian()
    tokens = [token.text for token in nlp(sentence)]
    max_len_token = max([len(token) for token in tokens])
    numeric_tokens = [data.word_field.vocab.stoi[token] for token in tokens]
    numeric_chars = []
    for token in tokens:
        numeric_chars.append([data.char_field.vocab.stoi[char] for char in token]
                             + [data.char_pad_idx for _ in range(max_len_token - len(token))])
    unk_idx = data.word_field.vocab.stoi[data.word_field.unk_token]
    unks = [t for t, n in zip(tokens, numeric_tokens) if n == unk_idx]
    # prediction
    numeric_tokens = torch.tensor(numeric_tokens)
    numeric_tokens = numeric_tokens.unsqueeze(-1)
    numeric_chars = torch.tensor(numeric_chars)
    numeric_chars = numeric_chars.unsqueeze(0)
    pred = model(numeric_tokens, numeric_chars)
    pred_tags = [data.tag_field.vocab.itos[idx] for idx in pred[0]]

    return pred_tags
