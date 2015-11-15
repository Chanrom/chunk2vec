#coding=utf-8
import os
from data import *
from nltk.tag import CRFTagger
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, ChunkFlatten, Reshape, Activation, SentSengment
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import cPickle
import time

def train_chunk2vec(
        train_split = 0.6, # 75% examples for training
        valid_split = 0.2,
        vocab_size = 100,
        max_chunk_len = 4,
        max_chunk_num = 3,
        wordvec_dim = 32,
        chunkvec_dim = 16,
        sentvec_dim = 8,
        batch_size = 3
):    

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    #chunking()
    #word2idx() #还未编写

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

    cPickle.dump(data, open("sample_data.pkl", "wb"))
    data = cPickle.load(open("sample_data.pkl", "rb"))
    print 'sample_data:', data

    train, valid, test = prepare_data(data, train_split, valid_split)
    #train = > (left_sent, left_chunk_mask, right..., ), score
    #print train

    model = Sequential()
    model.add(Embedding(input_dim=model_options['vocab_size'],
                        output_dim=model_options['wordvec_dim'],
                        init='orthogonal',
                        input_length=model_options['max_chunk_len']*model_options['max_chunk_num'],
                        mask_zero=True))

    print 'embedding layer output', model.layers[-1].output_shape

    # transform sentence tensor (nb_sents, max_chunk_num*max_chunk_len, wordvec_dim) ->
    # (nb_chunks, max_chunk_len, wordvec_dim)
    model.add(SentSengment((model_options['max_chunk_len'],
                            model_options['wordvec_dim'])))
    
    print 'sentence seg layer output', model.layers[-1].output_shape

    # Chunk LSTM
    model.add(LSTM(output_dim=model_options['chunkvec_dim'],
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_dim=model_options['wordvec_dim'],
                   keep_mask=True))

    model.add(Dropout(0.5))
#    model.add(Dense(output_dim=1,
#                    input_dim=chunkvec_dim))
#    model.add(Activation('sigmoid'))

#    model.compile(loss='mean_squared_error', optimizer='adadelta')
    
#    print 'X', train[0][0].shape
#    print 'Y', train[1]
#    labels = [0.2, 0.5, 0.4, 0.6, 0.7, 0.2, 0.4, 0.9, 1, 0.5, 0.7, 0.3]
#    model.fit(train[0][0], labels, batch_size=2, nb_epoch=1000)
#    score = model.evaluate(train[0][0], labels, batch_size=2)

#    print 'score:', score
#    print 'acc:', acc

#    for layer in model.layers:
#        weights = layer.get_weights()
#        print weights

    print 'chunk lstm output', model.layers[-1].output_shape
    #model.add(Reshape(dims=(4,4)))
    model.add(ChunkFlatten((model_options['max_chunk_num'],
                            model_options['chunkvec_dim'])))

    # Sentence LSTM
    print 'chunk flatten output', model.layers[-1].output_shape
    model.add(LSTM(output_dim=model_options['sentvec_dim'],
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_dim=model_options['chunkvec_dim']))
    model.add(Dropout(0.5))
    print 'sent LSTM output', model.layers[-1].output_shape 
    model.add(Dense(output_dim=1,
                    input_dim=sentvec_dim))                                                                                                                           
    model.add(Activation('sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adadelta')

    print 'X', train[0][0], train[0][0].shape
    print 'Y', train[1]

    model.fit(train[0][0], train[1], batch_size=batch_size, nb_epoch=1000)
#    score = model.evaluate(train[0][0], labels, batch_size=2)                                                                                        



if __name__ == '__main__':

    train_chunk2vec(
    )
