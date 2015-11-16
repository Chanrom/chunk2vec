#coding=utf-8
import os
import codecs
from nltk.tag import CRFTagger
import time
import numpy as np

pos_trainfile = '../../data/Chunking_data/train.txt' #用于PoS的训练语料
#pos_testfile = '../../data/Chunking_data/test.txt' #测试PoS，也可以为需要PoS的语料，仍需数据处理

chunk_trainfile = '../../data/Chunking_data/train.txt' #用于Chunking的训练语料
#chunk_inputfile = '../../data/Chunking_data/for_chunk.txt' #测试Chunking，也可以为PoS好的数据
#chunk_outputfile = '../../data/Chunking_data/chunked.txt' #chunking好的数据

max_chunk_num = 40
chunk_max_length = 13

def word2idx(filename, words, train=True):
    sents = []
#    words = {}
    sent = []
    chunk = []
    index = 1
    lines = codecs.open(filename, 'r', encoding='utf8').readlines()
    for line in lines:
        line = line.strip()
        #print line
        if len(line) == 0:
            sent.append(chunk[:])
            sents.append(sent[:])
            sent = []
            chunk = []
            continue
        l = line.split('\t')
        #print l
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
#            sent.append(chunk[:])
#            chunk = []

    if len(sent[:]) != 0:
        sents.append(sent[:])
    if train:
        words['__unknow__word__'] = index

    return sents


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
    lines = []
    for sent in tagged_sents:
        for word in sent:
            lines.append(word[0] + ' ' + word[1])
        lines.append('')
        
    f = codecs.open(filepath, 'w', encoding='utf-8')
    f.write('\n'.join(lines) + '\n')
    f.close()
    
def chunking(sents, chunked_file):
    '''
    Chunking
    param sents: 列表，如[['dog', 'is', 'dog'], ['dog', 'good']]
    '''
	
    os.chdir('/home/zqr/code/chunk2vec/')

    start_time = time.time()
    #PoS
    print '\n-->Start PoS'
    #print '->Training PoS Tagger'
    #ct = CRFTagger()
    #ct.train(chunk_traindata(pos_trainfile), 'model.crf.tagger')
    #print '->Done'
    
    #pos_testdata_gold = chunk_traindata(pos_testfile)
    
    # pos corpus
    print '->Load CRF Tagger model'
    ct = CRFTagger()
    ###这个model是从chunk任务中学习到的PoS标签
    ct.set_model_file('model.crf.tagger')
    print '->Posing'
    tagged_sents = ct.tag_sents(sents)
    #print 'PoS acc.:', ct.evaluate(pos_testdata_gold)
    #将PoS好的句子写文件
    print '->Write posed file'
    pos_data(tagged_sents, 'tmp_for_chunking')
    end_time = time.time()
    print '-->Done, Time:', end_time - start_time, 's'
    #节省时间，暂时用测试语料
    #pos_data(pos_testdata_gold, chunk_inputfile)
        
    start_time = time.time()
    ###Chunk，依赖系统安装YamCha，训练语料就用CoNLL的训练语料
    print '\n-->Start Chunking'
    os.system('yamcha-config --libexecdir')
    #os.chdir('/home/zqr/code/sent2vec/')
    os.system('cp /home/zqr/local/libexec/yamcha/Makefile .')
    #训练chunking模型
    #os.system('make CORPUS=' + pos_trainfile +' MODEL=chunk_model train')
    os.system('yamcha -m chunk_model.model < tmp_for_chunking > ' + chunked_file)
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

        
        
def get_data(train_file, test_file):
    '''
    输入文本格式：score\tsentence1\tsentence2
    输出格式[(0.8, [[1], [2, 3], [4, 5]], [[6], [7]]), (0.7, [[9], [10], [11, 12]], [[13, 14], [15, 16]])]
    '''
    def split_data(filename):
        '''
        输入文本格式：score\tsentence1\tsentence2
        返回格式如：[['dog', 'is', 'dog'], ['dog', 'good']]
        '''
        lines = codecs.open(filename, 'r', encoding='utf-8').readlines()
        sents = []
        score = []
        for line in lines:
            l = line.strip().split('\t')
            assert len(l) == 3
            #score.append(int(l[0]))
            a = [0, 0, 0, 0, 0]
            a[int(l[0])] = 1
            score.append(a)
            sents.append(l[1].split(' '))
            sents.append(l[2].split(' '))
        return score, sents
    
    #把评分和句子（索引表示拼在一起）
    def merge(score, sents):
        assert len(score) * 2 == len(sents)
        data = []
        for i in range(len(score)):
                data.append((score[i], sents[2*i], sents[2*i+1]))
        return data
    
    start_time = time.time()
    print '\n-->Load train data'
    train_score, train_raw_sents = split_data(train_file)
    end_time = time.time()
    print '-->Done, Time:', end_time - start_time, 's'
    chunking(train_raw_sents, 'tmp_chunked_file')
    words = {}
    print '\n-->Words to id'
    train_sents = word2idx('tmp_chunked_file', words, train=True)
    train_data = merge(train_score, train_sents)
    end_time2 = time.time()
    print '-->Done, Time:', end_time2 - end_time, 's'
    
    start_time = time.time()
    print '\n-->Load test data'
    test_score, test_raw_sents = split_data(test_file)
    end_time = time.time()
    print '-->Done, Time:', end_time - start_time, 's'
    chunking(test_raw_sents, 'tmp_chunked_file')
    print '\n-->Words to id'
    test_sents = word2idx('tmp_chunked_file', words, train=False)
    test_data = merge(test_score, test_sents)
    end_time2 = time.time()
    print '-->Done, Time:', end_time2 - end_time, 's'
    
    train_set_x, train_set_y = gene_data(train_data, 0, len(train_data))
    test_set_x, test_set_y = gene_data(test_data, 0, len(test_data))
    
    train = (train_set_x, train_set_y)
    test = (test_set_x, test_set_y)
    
    return train, test


#data = [(0.8, [[1], [2, 3], [4, 5]], [[6], [7]]),
#        (0.7, [[9], [10], [11, 12]], [[13, 14], [15, 16]]),
#        (0.9, [[17, 18], [19, 20]], [[21, 22, 23], [24, 25], [26]]),
#        (1, [[27], [28]], [[29, 30], [31, 32]]),
#        (0.2, [[33, 34, 35]], [[36, 37], [38]]),
#        (1, [[27], [28]], [[29, 30], [31, 32]]),
#        (1, [[27], [28]], [[29, 30], [31, 32]]),
#        (1, [[27], [28]], [[29, 30], [31, 32]]),
#        (0.2, [[33, 34, 35]], [[36, 37], [38]]),
#        (0.1, [[39, 40], [41]], [[42], [43]])]

#train_file = '/home/zqr/data/SST/sst_train.txt'
#test_file = '/home/zqr/data/SST/sst_test.txt'
#train_file = 'test_train.txt'
#test_file = 'test_test.txt'
#train, test = get_data(train_file, test_file)

#print 'left_sent', train[0][0]
#print 'right sent', train[0][1]
#print 'lables', train[1]

#print 'left_sent', test[0][0]
#print 'right sent', test[0][1]
#print 'lables', test[1]
