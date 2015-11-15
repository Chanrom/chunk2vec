#coding=utf-8
import os
import codecs
from nltk.tag import CRFTagger
import time
import numpy as np

pos_trainfile = '../../data/Chunking_data/train.txt' #用于PoS的训练语料
pos_testfile = '../../data/Chunking_data/test.txt' #测试PoS，也可以为需要PoS的语料，仍需数据处理

chunk_trainfile = '../../data/Chunking_data/train.txt' #用于Chunking的训练语料
chunk_inputfile = '../../data/Chunking_data/for_chunk.txt' #测试Chunking，也可以为PoS好的数据
chunk_outputfile = '../../data/Chunking_data/chunked.txt' #chunking好的数据

max_chunk_num = 3
chunk_max_length = 4

def word2idx(filename, words, train=True):
    sents = []
#    words = {}
    sent = []
    chunk = []
    index = 1
    lines = codecs.open(filename, 'r', encoding='utf8').readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            sent.append(chunk[:])
            sents.append(sent[:])
            sent = []
            chunk = []
            continue
        l = line.split('\t')
        # a new chunk
        if l[2].startswith('B'):
            if len(chunk) != 0:
                sent.append(chunk[:])
                chunk = []
            if not words.has_key(l[0].lower()):
                if train:
                    words[l[0].lower()] = index
                    index += 1
                else:
                    l[0] = '__unknow__word__'
            chunk.append(words[l[0].lower()])

        # between a chunk
        if l[2].startswith('I'):
            if not words.has_key(l[0].lower()):
                if train:
                    words[l[0].lower()] = index
                    index += 1
                else:
                    l[0] = '__unknow__word__'
            chunk.append(words[l[0].lower()])

        if l[2].startswith('O'):
            sent.append(chunk[:])
            chunk = []

    sents.append(sent[:])
    words['__unknow__word__'] = index + 1

    return sents

def pos_sent(train_data, sents):
    print '->Training PoS Tagger'
    ct = CRFTagger()
    ct.train(train_data, 'model.crf.tagger')
    
    tagged_sents = ct.tag_sents(sents)
    print '->Done'
    return tagged_sents, ct

def chunk_traindata(filepath):
    print '->Loading Train data', filepath
    data = []
    lines = codecs.open(filepath, 'r', encoding='utf-8').readlines()
#    print len(lines)
    sent = []
    for line in lines:
        line = line.strip()
        #print line
        if len(line) == 0:
            data.append([x for x in sent])
            sent = []
            continue
        l = line.split(' ')
        for i in range(len(l) / 2):
            sent.append((l[2*i], l[2*i+1]))
 
    print '->Done'
    return data

def chunk_testdata(filepath):
    print '->Loading Test data', filepath
    data = []
    lines = codecs.open(filepath, 'r', encoding='utf-8').readlines()

    sent = []
    for line in lines:
        line = line.strip()
        #print line                                                                                                                                                                
        if len(line) == 0:
            data.append([x for x in sent])
            sent = []
            continue
        l = line.split(' ')
        for i in range(len(l) / 2):
            sent.append(l[2*i])

    print '->Done'
    return data

        
def pos_data(tagged_sents, filepath):
#    print tagged_sents[0]
    line = ''
    for sent in tagged_sents:
        for word in sent:
            line = line + word[0] + ' ' + word[1] + '\n'
        line += '\n'

    f = codecs.open(filepath, 'w', encoding='utf-8')
    f.write(line)
    f.close()
    
def chunking():
	
    os.chdir('/home/zqr/code/sent2vec/')

    start_time = time.time()
    ###PoS
    print '\n-->Start PoS'
    #(tagged_sents, ct) = pos_sent(chunk_traindata(pos_trainfile), 
    #                         chunk_testdata(pos_testfile))
    pos_testdata_gold = chunk_traindata(pos_testfile)
    #evaluate pos
    ct = CRFTagger()
    ct.set_model_file('model.crf.tagger')
    print 'PoS acc.:', ct.evaluate(pos_testdata_gold)
    end_time = time.time()
    print '-->Done, Time:', end_time - start_time, 's'
    #将PoS好的句子写文件
    #pos_data(tagged_sents, chunk_inputfile)
    #节省时间，暂时用测试语料
    pos_data(pos_testdata_gold, chunk_inputfile)
    
        
    start_time = time.time()
    ###Chunk，依赖系统安装YamCha，训练语料就用CoNLL的训练语料
    print '\n-->Start Chunking'
    os.system('yamcha-config --libexecdir')
    #os.chdir('/home/zqr/code/sent2vec/')
    os.system('cp /home/zqr/local/libexec/yamcha/Makefile .')
    #训练chunking模型
    #os.system('make CORPUS=' + pos_trainfile +' MODEL=chunk_model train')
    os.system('yamcha -m chunk_model.model < ' + chunk_inputfile + ' > ' + chunk_outputfile)
    print '-->Done, Time:', time.time() - start_time, 's'
    

def prepare_data(data, train_split, valid_split):
    '''Halve data for Siamese architecture.                                                                                                                                        
    :type data: list                                                                                                                                                               
    :param data: [(0.8, [[1], [2, 3], [4, 5]], [[6], [7]]), (0.7, [[9], [10], [11, 12]], [[13, 14], [15, 16]])]                                                                    
    type train_split: float                                                                                                                                                        
    param train_split: example proportion for training                                                                                                                             
    type test_split: float                                                                                                                                                         
    param test_split: example proportion for testing                                                                                                                               
    '''
    #random shuffle the data                                                                                                                                                       
    np.random.shuffle(data)

    data_size = len(data)
    train_size = int(data_size*train_split)
    valid_size = int(data_size*valid_split)

#    print train_size
    train_set_x, train_set_y = gene_data(data, 0, train_size)
    valid_set_x, valid_set_y = gene_data(data, train_size, train_size + valid_size)
    test_set_x, test_set_y = gene_data(data, train_size + valid_size, data_size)

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
    
    return train, valid, test

def gene_data(data, start, end):

    set_y = []
    set_left = []
    set_right = []
    for pair in data[start:end]:
        set_y.append(pair[0])
        set_left.append(pair[1])
        set_right.append(pair[2])

    def _gene(s):
        sent_size = len(s)
        sent = np.zeros((sent_size, max_chunk_num*chunk_max_length)).astype('int64')
        for i in range(sent_size):
            for j in range(len(s[i])):
                sent[i, j*chunk_max_length:j*chunk_max_length+len(s[i][j])] = s[i][j]

        return sent

    set_left_sent = _gene(set_left)
    set_right_sent = _gene(set_right)

    set_x = set_left_sent, set_right_sent
    
    return set_x, set_y

    
def gene_data2(data, start, end):

    set_y = []
    set_left = []
    set_right = []
    for pair in data[start:end]:
        set_y.append(pair[0])
        set_left.append(pair[1])
        set_right.append(pair[2])

    set_left_sent, set_left_sent_mask, set_left_chunk_mask = gene_mask2(set_left)
    set_right_sent, set_right_sent_mask, set_right_chunk_mask = gene_mask2(set_right)

    set_x = set_left_sent, set_left_sent_mask, set_left_chunk_mask, set_right_sent, set_right_sent_mask, set_right_chunk_mask
    
    return set_x, set_y

    
def gene_mask2(s):
    sent_size = len(s)
    sent = np.zeros((sent_size, max_chunk_num*chunk_max_length)).astype('int64')
    chunk_mask = np.zeros((sent_size * max_chunk_num, chunk_max_length)).astype('int64')
    sent_mask = np.zeros((sent_size, max_chunk_num*chunk_max_length)).astype('int64')
    for i in range(sent_size):
        for j in range(len(s[i])):
            sent[i, j*chunk_max_length:j*chunk_max_length+len(s[i][j])] = s[i][j]
            sent_mask[i, j*chunk_max_length:j*chunk_max_length+len(s[i][j])] = 1
            chunk_mask[i*max_chunk_num + j, :len(s[i][j])] = 1
            
    return sent, sent_mask, chunk_mask

def gene_data1(data, start, end):
    '''Give a dataset, generate its chunks ans mask matrixs
    :type data: list
    :param data: dataset
    :type start: int
    :param start: starte point in the dataset
    '''
    set_y = []
    set_left = []
    set_right = []
    for pair in data[start:end]:
        set_y.append(pair[0])
        set_left.append(pair[1])
        set_right.append(pair[2])

    set_left_chunk, set_left_chunk_mask, set_left_sent_mask = gene_mask(set_left)
    set_right_chunk, set_right_chunk_mask, set_right_sent_mask = gene_mask(set_right)
#    print set_left_chunk

    set_x = (set_left_chunk, set_left_chunk_mask, set_left_sent_mask, set_right_chunk, set_right_chunk_mask, set_right_sent_mask)
    
    return set_x, set_y
    
def gene_mask1(s):
    '''Generate mask matrix for sentences
    :type s: list
    :param s: sentences, should be the format such as [[[1], [2, 3]], [[4], [5]]]

    '''
    sent_size = len(s)
    chunk = np.zeros((sent_size * max_chunk_num, chunk_max_length)).astype('int64')
    chunk_mask = np.zeros((sent_size * max_chunk_num, chunk_max_length)).astype('int64')
    sent_mask = np.zeros((sent_size, max_chunk_num)).astype('int64')
    for i in range(len(s)):
        sent_mask[i, :len(s[i])] = 1
        for j in range(len(s[i])):
            chunk_mask[i*max_chunk_num + j, :len(s[i][j])] = 1
            chunk[i*max_chunk_num +j, :len(s[i][j])] = s[i][j]

    return chunk, chunk_mask, sent_mask

def get_data(trainfile, testfile, train_split, test_split):
    chunk


data = [(0.8, [[1], [2, 3], [4, 5]], [[6], [7]]),
        (0.7, [[9], [10], [11, 12]], [[13, 14], [15, 16]]),
        (0.9, [[17, 18], [19, 20]], [[21, 22, 23], [24, 25], [26]]),
        (1, [[27], [28]], [[29, 30], [31, 32]]),
        (0.2, [[33, 34, 35]], [[36, 37], [38]]),
        (1, [[27], [28]], [[29, 30], [31, 32]]),
        (1, [[27], [28]], [[29, 30], [31, 32]]),
        (1, [[27], [28]], [[29, 30], [31, 32]]),
        (0.2, [[33, 34, 35]], [[36, 37], [38]]),
        (0.1, [[39, 40], [41]], [[42], [43]])]

train, valid, test = prepare_data(data, 0.6, 0.2)

print 'left_sent', train[0][0]
print 'right sent', train[0][1]
print 'lables', train[1]
