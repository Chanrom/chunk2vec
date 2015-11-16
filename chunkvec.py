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
        vocab_size = 17591,
        max_chunk_len = 13,
        max_chunk_num = 40,
        wordvec_dim = 128,
        chunkvec_dim = 128,
        sentvec_dim = 128,
        batch_size = 64,
        train_raw_file = '/home/zqr/data/SST/sst_train.txt',
        test_raw_file = '/home/zqr/data/SST/sst_test.txt'
):    

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    #train, test = get_data(train_raw_file, test_raw_file)
    
    start_time = time.time()
    
    #cPickle.dump((train, test), open("sst_data.pkl", "wb"))
    print '\n-->Load data'
    train, test = cPickle.load(open("sst_data.pkl", "rb"))
    print '-->Done, Time:', time.time() - start_time
    
    start_time = time.time()
    print '\n-->Build model'
    model = Sequential()
    model.add(Embedding(input_dim=model_options['vocab_size'],
                        output_dim=model_options['wordvec_dim'],
                        init='uniform',
                        input_length=model_options['max_chunk_len']*model_options['max_chunk_num'],
                        mask_zero=True))

    print '->embedding layer output', model.layers[-1].output_shape

    # transform sentence tensor (nb_sents, max_chunk_num*max_chunk_len, wordvec_dim) ->
    # (nb_chunks, max_chunk_len, wordvec_dim)
    model.add(SentSengment((model_options['max_chunk_len'],
                            model_options['wordvec_dim'])))
    
    print '->sentence seg layer output', model.layers[-1].output_shape

    # Chunk LSTM
    model.add(LSTM(output_dim=model_options['chunkvec_dim'],
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_dim=model_options['wordvec_dim'],
                   keep_mask=True))

    model.add(Dropout(0.5))

    print '->chunk lstm output', model.layers[-1].output_shape
    #model.add(Reshape(dims=(4,4)))
    model.add(ChunkFlatten((model_options['max_chunk_num'],
                            model_options['chunkvec_dim'])))

    # Sentence LSTM
    print '->chunk flatten output', model.layers[-1].output_shape
    model.add(LSTM(output_dim=model_options['sentvec_dim'],
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   input_dim=model_options['chunkvec_dim']))
    model.add(Dropout(0.5))
    print '->sent LSTM output', model.layers[-1].output_shape 
    model.add(Dense(output_dim=5,
                    input_dim=sentvec_dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print '-->Done, Time:', time.time() - start_time

    model.fit(train[0][0], train[1], batch_size=batch_size,
              nb_epoch=10000, verbose=1, validation_split=0.15, show_accuracy=True)
#    score = model.evaluate(train[0][0], labels, batch_size=2)                                                                                        



if __name__ == '__main__':

    train_chunk2vec(
    )
