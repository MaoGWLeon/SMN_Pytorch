import torch
from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow_src import utils
import time
from tensorflow_src import Evaluate

embedding_file = r"../data/Ubuntu/embedding.pkl"
evaluate_file = r"../data/Ubuntu/Evaluate.pkl"
response_file = r"../data/Ubuntu/responses.pkl"
history_file = r"../data/Ubuntu/utterances.pkl"


class Config():
    def __init__(self):
        self.max_num_utterance = 10
        self.negative_samples = 1  # 抽样一个负例
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        self.rnn_units = 200
        self.total_words = 434511
        self.batch_size = 40  # 用80可以跑出论文的结果，现在用论文的参数


class SCN_all_share(nn.Module):
    def __init__(self, config):
        # 标准动作
        super(SCN_all_share, self).__init__()
        print(f'this is SCN all share Model')
        # 参数设定
        self.max_num_utterance = config.max_num_utterance
        self.negative_samples = config.negative_samples
        self.max_sentence_len = config.max_num_utterance
        self.word_embedding_size = config.word_embedding_size
        self.rnn_units = config.rnn_units
        self.total_words = config.total_words
        # batch_size指的是正例的个数，然后从负例数据集中随机抽config.negative_samples个负例，再和utterance组成一个完整的负例
        self.batch_size = config.batch_size + config.negative_samples * config.batch_size

        # 需要的模块

        with open(embedding_file, 'rb') as f:
            embedding_matrix = pickle.load(f, encoding="bytes")
            assert embedding_matrix.shape == (434511, 200)
        self.word_embedding = nn.Embedding(self.total_words, self.word_embedding_size)
        self.word_embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.word_embedding.weight.requires_grad = False

        # 论文用的是单向的GRU，而且需要10个用于utterance
        # 这个版本的模型所有的模块都共享，即utterance都用相同的GRU
        self.utterance_GRU = nn.GRU(self.word_embedding_size, self.rnn_units, bidirectional=False, batch_first=True)
        ih_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)
        # 用于response的GRU
        self.response_GRU = nn.GRU(self.word_embedding_size, self.rnn_units, bidirectional=False, batch_first=True)
        ih_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_ih' in name)
        hh_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_r:
            nn.init.orthogonal_(k)
        for k in hh_r:
            nn.init.orthogonal_(k)
        # 1、初始化参数的方式要注意
        # 2、参数共享的问题要小心
        # 3、gru不共享参数，conv2d和linear共享参数
        # 正因为conv2d和linear共享参数 只需要定义一个就可以了，gru要一个个定义
        self.conv2d = nn.Conv2d(2, 8, kernel_size=(3, 3))
        conv2d_weight = (param.data for name, param in self.conv2d.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            init.kaiming_normal_(w)

        self.pool2d = nn.MaxPool2d((3, 3), stride=(3, 3))

        self.linear = nn.Linear(16 * 16 * 8, 50)
        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            init.xavier_uniform_(w)

        self.Amatrix = torch.ones((self.rnn_units, self.rnn_units), requires_grad=True)
        init.xavier_uniform_(self.Amatrix)
        self.Amatrix = self.Amatrix.cuda()
        # 最后一层的gru
        self.final_GRU = nn.GRU(50, self.rnn_units, bidirectional=False, batch_first=True)
        ih_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)
        # final_GRU后的linear层
        self.final_linear = nn.Linear(200, 2)
        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            init.xavier_uniform_(w)

    def forward(self, utterance, response):
        '''
            utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
            response:(self.batch_size, self.max_sentence_len)
        '''
        # (batch_size,10,50)-->(batch_size,10,50,200)
        all_utterance_embeddings = self.word_embedding(utterance)
        response_embeddings = self.word_embedding(response)

        # tensorflow:(batch_size,10,50,200)-->分解-->10个array(batch_size,50,200)
        # pytorch:(batch_size,10,50,200)-->(10,batch_size,50,200)
        all_utterance_embeddings = all_utterance_embeddings.permute(1, 0, 2, 3)

        # (batch_size,10)-->(10,batch_size)
        # 在pytorch里面貌似没啥用 这个是只是为了方便tf里面定义dynamic_rnn用的
        # all_utterance_len = all_utterance_len.permute(1, 0)

        # 先处理response的gru
        response_GRU_embeddings, _ = self.response_GRU(response_embeddings)
        response_embeddings = response_embeddings.permute(0, 2, 1)
        response_GRU_embeddings = response_GRU_embeddings.permute(0, 2, 1)
        matching_vectors = []

        for utterance_embeddings in all_utterance_embeddings:
            matrix1 = torch.matmul(utterance_embeddings, response_embeddings)

            utterance_GRU_embeddings, _ = self.utterance_GRU(utterance_embeddings)
            matrix2 = torch.einsum('aij,jk->aik', utterance_GRU_embeddings, self.Amatrix)
            matrix2 = torch.matmul(matrix2, response_GRU_embeddings)

            matrix = torch.stack([matrix1, matrix2], dim=1)
            # matrix:(batch_size,channel,seq_len,embedding_size)
            conv_layer = self.conv2d(matrix)
            # add activate function
            conv_layer = F.relu(conv_layer)
            pooling_layer = self.pool2d(conv_layer)
            # flatten
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)
            matching_vector = self.linear(pooling_layer)
            # add activate function
            matching_vector = F.tanh(matching_vector)
            matching_vectors.append(matching_vector)

        _, last_hidden = self.final_GRU(torch.stack(matching_vectors, dim=1))
        last_hidden = torch.squeeze(last_hidden)
        logits = self.final_linear(last_hidden)

        # use CrossEntropyLoss,this loss function would accumulate softmax
        # y_pred = F.softmax(logits)
        y_pred = logits
        return y_pred


def train_Model():
    config = Config()
    print(f'----------load data----------')
    with open(response_file, 'rb') as f:
        # negative
        actions = pickle.load(f)  # <class 'list'>,500000

    with open(history_file, 'rb') as f:
        history, true_utt = pickle.load(f)  # <class 'list'>,500000 ;<class 'list'>,500000
    '''
        SMN模型数据格式:
        刚开始读取的时候:   actions:[500000]    history:[500000,?](?代表不定长，因为对话历史轮数不确定，后续处理都会保存为十轮)
                         turn_utt:[500000]
                         actions的意思是错误的句子，训练时都是用一个正确加一个的错误的回复去训练
        处理后:    actions:(500000,50) 句子后向补零，长度都是50
                  actions_len:(500000,) 每个句子的真实长度
                  history:(500000,10,50) 选取最后十轮对话，每句话长度都是50
                  history_len:(500000,10) 十轮对话中每个对话的真实长度
                  true_utt:(500000,50) 句子后向补零，长度都是50 history和true_utt按index一一对应
                  true_utt_len:(500000,) 每个句子的真实长度
    
    '''
    history, history_len = utils.multi_sequences_padding(history, config.max_sentence_len)
    true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=config.max_sentence_len))  # (500000,)
    true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=config.max_sentence_len))  # (500000,50)
    # negative response
    actions_len = np.array(utils.get_sequences_length(actions, maxlen=config.max_sentence_len))  # (500000,)
    actions = np.array(pad_sequences(actions, padding='post', maxlen=config.max_sentence_len))  # (500000,50)

    history, history_len = np.array(history), np.array(history_len)  # (500000,10,50) (500000,10)

    # build model

    model = SCN_all_share(config)
    print(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.cuda()

    low = 0
    epoch = 1
    model.train()

    while epoch < 10:

        n_sample = min(low + config.batch_size, history.shape[0]) - low

        negative_indices = [np.random.randint(0, actions.shape[0], n_sample) for _ in range(config.negative_samples)]
        negs = [actions[negative_indices[i], :] for i in range(config.negative_samples)]
        negs_len = [actions_len[negative_indices[i]] for i in range(config.negative_samples)]

        utterance = np.concatenate([history[low:low + n_sample]] * (config.negative_samples + 1), axis=0)
        all_utterance_len = np.concatenate([history_len[low:low + n_sample]] * (config.negative_samples + 1), axis=0)
        response = np.concatenate([true_utt[low:low + n_sample]] + negs, axis=0)
        response_len = np.concatenate([true_utt_len[low:low + n_sample]] + negs_len, axis=0)
        y_true = np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * config.negative_samples, axis=0)

        # all_utterance_len 和 response_len 在pytorch中不需要用到
        utterance = torch.LongTensor(utterance).cuda()
        response = torch.LongTensor(response).cuda()
        y_true = torch.LongTensor(y_true).cuda()

        y_pred = model(utterance, response)
        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        low += n_sample

        if low % 100000 == 0:
            model.eval()
            print(f'loss:{loss.item()}')
            evaluate_Model(model)
            model.train()

        if low >= history.shape[0]:
            low = 0
            print(f'epoch:{epoch} had finish')
            epoch += 1


def evaluate_Model(model):
    config = Config()
    with open(evaluate_file, 'rb') as f:
        history, true_utt, labels = pickle.load(f)
    # evaluate的数据中history 每十条数据都是一样的，true_utt每十条都对应相同的history，这十个true_utt中只有第一个正确
    # 其他九个都是不正确的，所以labels中每十个中只有第一个是1，其他九个都是0
    # evaluate中这样的原因是为了符合评价标准，即recall@k，从k个候选答案中选择正确的答案
    all_candidate_scores = []
    history, history_len = utils.multi_sequences_padding(history, config.max_sentence_len)
    history, history_len = np.array(history), np.array(history_len)  # (,10,50)
    true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=config.max_sentence_len))
    true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=config.max_sentence_len))  # (,50)
    low = 0
    print_test = 0
    while True:
        utterance = np.concatenate([history[low:low + 200]], axis=0)
        response = np.concatenate([true_utt[low:low + 200]], axis=0)
        utterance = torch.LongTensor(utterance).cuda()  # (200,10,50)
        response = torch.LongTensor(response).cuda()  # (200,50)
        candidate_scores = model(utterance, response)
        candidate_scores = F.softmax(candidate_scores, 0).cpu().detach().numpy()
        # if print_test <= 1:
        #     print_test += 1
        #     print(f'candidate_scores:{candidate_scores}')
        all_candidate_scores.append(candidate_scores[:, 1])
        low = low + 200
        if low >= history.shape[0]:
            break
    all_candidate_scores = np.concatenate(all_candidate_scores, axis=0)
    Evaluate.ComputeR10_1(all_candidate_scores, labels)
    Evaluate.ComputeR2_1(all_candidate_scores, labels)


'''
embedding_file = r"../data/Ubuntu/embedding.pkl"
evaluate_file = r"../data/Ubuntu/Evaluate.pkl"
response_file = r"../data/Ubuntu/responses.pkl"
history_file = r"../data/Ubuntu/utterances.pkl"
'''
worddict_file = r"../data/Ubuntu/worddict.pkl"

if __name__ == "__main__":
    train_Model()
