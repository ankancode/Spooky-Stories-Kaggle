import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


HIDDEN_SIZE = 100
N_LAYERS = 2
BATCH_SIZE = 256

N_EPOCHS = 100

test_dataset = NameDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)


train_dataset = NameDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)

N_AUTHORS = len(train_dataset.get_authors())
print(N_AUTHORS, "authors")
N_CHARS = 128  



def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def create_variable(tensor):
    
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)



def pad_sequences(vectorized_seqs, seq_lengths, authors):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

   
    target = authors2tensor(authors)
    if len(authors):
        target = target[perm_idx]

    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), \
        create_variable(seq_lengths), \
        create_variable(target)


# Create necessary variables, lengths, and target
def make_variables(stories, authors):
    sequence_and_length = [str2ascii_arr(story) for story in stories]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, authors)


def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr, len(arr)


def authors2tensor(authors):
    author_ids = [train_dataset.get_author_id(
        author) for author in authors]
    return torch.LongTensor(author_ids)


class RNNClassifier(nn.Module):
    # Our model

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        
        input = input.t()
        batch_size = input.size(1)

        
        hidden = self._init_hidden(batch_size)

        embedded = self.embedding(input)

        gru_input = pack_padded_sequence(
            embedded, seq_lengths.data.cpu().numpy())

        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)

        fc_output = self.fc(hidden[-1])
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)


# Train cycle
def train():
    total_loss = 0

    for i, (stories, authors) in enumerate(train_loader, 1):
        input, seq_lengths, target = make_variables(stories, authors)
        output = classifier(input, seq_lengths)

        loss = criterion(output, target)
        total_loss += loss.data[0]

        classifier.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                time_since(start), epoch,  i *
                len(stories), len(train_loader.dataset),
                100. * i * len(stories) / len(train_loader.dataset),
                total_loss / i * len(stories)))

    return total_loss

def test(story=None):
    if story:
        input, seq_lengths, target = make_variables([story], [])
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        author_id = pred.cpu().numpy()[0][0]
        print(story, "is", train_dataset.get_author(author_id))
        return

    print("evaluating trained model ...")
    correct = 0
    train_data_size = len(test_loader.dataset)

    for stories, authors in test_loader:
        input, seq_lengths, target = make_variables(stories, authors)
        output = classifier(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, train_data_size, 100. * correct / train_data_size))


if __name__ == '__main__':

    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_AUTHORS, N_LAYERS)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = nn.DataParallel(classifier)

    if torch.cuda.is_available():
        classifier.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
    
        train()
        test()
