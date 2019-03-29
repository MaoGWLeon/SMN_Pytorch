import pickle
import numpy as np

conf = {
    "batch_size": 100,  # 200 for test
    "_EOS_": 28270,  # 1 for douban data
    "max_turn_num": 10,
    "max_turn_len": 50,

}

DAM_data_file = r"/home/scutnlp131/maogw/Multi-Turn-Conversation/Dialogue/DAM/data/ubuntu/data.pkl"

def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c = np.array(data['c'])
    r = np.array(data['r'])

    assert len(y) == len(c) == len(r)
    p = np.random.permutation(len(y))
    shuffle_data = {'y': y[p], 'c': c[p], 'r': r[p]}
    return shuffle_data


def build_batches(data, conf=conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns_batches = []
    _tt_turns_len_batches = []
    _every_turn_len_batches = []

    _response_batches = []
    _response_len_batches = []

    _label_batches = []

    batch_len = len(data['y']) / conf['batch_size']
    batch_len = int(batch_len)
    print(f'batch_len:{batch_len}')  # batch 的数量
    for batch_index in range(batch_len):
        _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label = \
            build_one_batch(data, batch_index, conf, turn_cut_type='tail', term_cut_type='tail')

        _turns_batches.append(_turns)
        _tt_turns_len_batches.append(_tt_turns_len)
        _every_turn_len_batches.append(_every_turn_len)

        _response_batches.append(_response)
        _response_len_batches.append(_response_len)

        _label_batches.append(_label)

    ans = {
        "turns": _turns_batches,
        "tt_turns_len": _tt_turns_len_batches,
        "every_turn_len": _every_turn_len_batches,

        "response": _response_batches,
        "response_len": _response_len_batches,

        "label": _label_batches
    }

    return ans


def build_one_batch(data, batch_index, conf=conf, turn_cut_type='tail', term_cut_type='tail'):
    _turns = []
    _tt_turns_len = []
    _every_turn_len = []

    _response = []
    _response_len = []

    _label = []

    for i in range(conf['batch_size']):
        index = batch_index * conf['batch_size'] + i

        y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len = \
            produce_one_sample(data, index, conf['_EOS_'], conf['max_turn_num'], conf['max_turn_len'],
                               turn_cut_type, term_cut_type)

        _label.append(y)
        _turns.append(nor_turns_nor_c)
        _response.append(nor_r)
        _every_turn_len.append(term_len)
        _tt_turns_len.append(turn_len)
        _response_len.append(r_len)

    return _turns, _tt_turns_len, _every_turn_len, _response, _response_len, _label


def produce_one_sample(data, index, split_id, max_turn_num, max_turn_len, turn_cut_type='tail', term_cut_type='tail'):
    '''max_turn_num=10
       max_turn_len=50
       return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len
       y:label  nor_turns_nor_c:turns   nor_r:response
       turn_len:tt_turns_len    term_len:every_turn_len     r_len:response_len
    '''
    c = data['c'][index]
    r = data['r'][index][:]
    y = data['y'][index]

    turns = split_c(c, split_id)
    # normalize turns_c length, nor_turns length is max_turn_num
    nor_turns, turn_len = normalize_length(turns, max_turn_num, turn_cut_type)

    nor_turns_nor_c = []
    term_len = []
    # nor_turn_nor_c length is max_turn_num, element is a list length is max_turn_len
    for c in nor_turns:
        # nor_c length is max_turn_len
        nor_c, nor_c_len = normalize_length(c, max_turn_len, term_cut_type)
        nor_turns_nor_c.append(nor_c)
        term_len.append(nor_c_len)

    nor_r, r_len = normalize_length(r, max_turn_len, term_cut_type)

    return y, nor_turns_nor_c, nor_r, turn_len, term_len, r_len


def normalize_length(_list, length, cut_type='tail'):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list)
    if real_length == 0:
        return [0] * length, 0

    if real_length <= length:
        if not isinstance(_list[0], list):
            _list.extend([0] * (length - real_length))
        else:
            _list.extend([[]] * (length - real_length))
        return _list, real_length

    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length


def split_c(c, split_id):
    '''c is a list, example context
       split_id is a integer, conf[_EOS_]
       return nested list
    '''
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns


def read_data():
    '''
        train_data:<class 'dict'>,dict_keys(['y', 'c', 'r'])
        val_data:<class 'dict'>,dict_keys(['y', 'c', 'r'])
        test_data:<class 'dict'>,dict_keys(['y', 'c', 'r'])
    '''

    # data 已经被token化
    '''
        data_small.pkl: train_data:10000    val_data:1000   test_data:1000
        data.pkl: train_data:1000000(pos:500000;neg:500000)    val_data:500000(10个当一组，50000个数据)   test_data:500000(10个当一组，50000个数据)
        data['c']是一个list,用_EOS_作为划分句子的标记，_EOS_的index为28270
    '''
    print(f"starting loading data")
    with open(DAM_data_file, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)
    print(f"finish loading data")

    val_batches = build_batches(val_data, conf)
    val_batch_num = len(val_batches["response"])
    print(f"finish building val batches")

    batch_num=len(train_data['y'])/conf["batch_size"]
    batch_num=int(batch_num)
    print(f"train data batch num:{batch_num}")
    train_batches = build_batches(train_data, conf)

    turns = train_batches["turns"][1]
    tt_turns_len = train_batches["tt_turns_len"][1]
    every_turn_len = train_batches["every_turn_len"][1]
    response = train_batches["response"][1]
    response_len = train_batches["response_len"][1]
    label = train_batches["label"][1]

    print(type(turns),type(tt_turns_len),type(every_turn_len),type(response),type(response_len),type(label))



if __name__ == '__main__':
    read_data()
