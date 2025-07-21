# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy
from keras.utils.data_utils import get_file
from zipfile import ZipFile
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import cifar10
import itertools
from keras import backend as K
import numpy as np
import sys
# limit memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#set_session(tf.compat.v1.Session(config=config))

# above lines are for GPU optimization

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics, reports per classs recall, precision and F1 score'''
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               round(corrects[i] / max(1, y_true_counts[i]),4),  # recall
               round(corrects[i] / max(1, y_pred_counts[i]),4),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('', 'recall', 'precision', 'f1-score', 'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total', float((sum(report2[0]) / N)*100), (sum(report2[1]) / N)*100, (sum(report2[2]) / N)*100, N) + '\n')


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'float32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'float32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])

        token_sequences.append(token_sequence)

    return token_sequences

def _parse_data(fh):
    string = fh.read()
    data = [[row.split() for row in sample.split('\n')] for sample in string.decode().strip().split('\n\n')]
    fh.close()
    return data

include_char_emb ='no'
include_pos_emb ='yes'
include_w2v_emb ='no'
include_elmo_emb ='yes'

path="data/mtl-data.zip"
archive = ZipFile(path, 'r')
train = _parse_data(archive.open('mtl-data/train-fun.mtl'))
test = _parse_data(archive.open('mtl-data/test-fun.mtl'))
#dev = _parse_data(archive.open('mtl-data/CLE-test.lu'))
dev=test

archive.close()

min_freq=1
word_counts = Counter(row[0] for sample in train for row in sample)
vocab = ['-PAD-', '-OOV-','-BOS-','-EOS-'] + [w for w, f in iter(word_counts.items()) if f >= min_freq]
pos_tags = sorted(list(set(row[1] for sample in train + test + dev for row in sample if row[1] not in ['-BOS-','-EOS-'])))  # in alphabetic order
pos_tags = ['-PAD-','-BOS-','-EOS-']+pos_tags
print(pos_tags)
lu_tags = sorted(list(set(row[2] for sample in train + test + dev for row in sample if row[2] not in ['-BOS-','-EOS-'])))  # in alphabetic order
lu_tags = ['-PAD-','-BOS-','-EOS-']+lu_tags
print(lu_tags)
seq_tags = sorted(list(set(row[3] for sample in train + test + dev for row in sample if row[3] not in ['-BOS-','-EOS-'])))  # in alphabetic order
seq_tags = ['-PAD-','-BOS-','-EOS-']+seq_tags
print(seq_tags)

print("Total POS Labels: ",len(pos_tags))
print("Total Tree Lu Labels: ",len(lu_tags))
print("Total Tree Seq Labels: ",len(seq_tags))

train_sents=[]
train_pos_tags=[]
train_lu_tags=[]
train_seq_tags=[]
for line in train:
    w_line=""
    pos_line=""
    lu_line = ""
    seq_line=""
    for w in line:
        #print(w[0])

        if len(w)>=4:
            w_line=w_line+" "+w[0]
            pos_line =pos_line+ " "+w[1]
            lu_line = lu_line + " " + w[2]
            seq_line = seq_line+" "+w[3]
    train_sents.append(w_line)
    train_pos_tags.append(pos_line)
    train_lu_tags.append(lu_line)
    train_seq_tags.append(seq_line)

test_sents=[]
test_pos_tags=[]
test_lu_tags=[]
test_seq_tags=[]
for line in test:
    w_line=""
    pos_line=""
    lu_line = ""
    seq_line=""
    for w in line:
        #print(w[0])

        if len(w)>=4:
            w_line=w_line+" "+w[0]
            pos_line =pos_line+ " "+w[1]
            lu_line = lu_line + " " + w[2]
            seq_line = seq_line+" "+w[3]
    test_sents.append(w_line)
    test_pos_tags.append(pos_line)
    test_lu_tags.append(lu_line)
    test_seq_tags.append(seq_line)

dev_sents=[]
dev_pos_tags=[]
dev_lu_tags=[]
dev_seq_tags=[]
for line in dev:
    w_line=""
    pos_line=""
    lu_line = ""
    seq_line=""
    for w in line:
        #print(w[0])

        if len(w)>=4:
            w_line=w_line+" "+w[0]
            pos_line =pos_line+ " "+w[1]
            lu_line = lu_line + " " + w[2]
            seq_line = seq_line+" "+w[3]
    dev_sents.append(w_line)
    dev_pos_tags.append(pos_line)
    dev_lu_tags.append(lu_line)
    dev_seq_tags.append(seq_line)

print(train_sents[0])
print(train_pos_tags[0])
print(train_lu_tags[0])
print(train_seq_tags[0])

#print(pos_tags)
#print(lu_tags)
#print(seq_tags)

##############################################################################
############################# ELMO settings ####################################
##############################################################################
if include_elmo_emb =='yes':
    sent_length=100
    elmo_dim=128
    elmo_layer=2

    new_train_sents=[]
    for i,j in enumerate(train_sents):
        new_train_sents.append(j.strip().split(' ')[:sent_length])

    new_test_sents=[]
    for i,j in enumerate(test_sents):
        new_test_sents.append(j.strip().split(' ')[:sent_length])

    new_dev_sents=[]
    for i,j in enumerate(dev_sents):
        new_dev_sents.append(j.strip().split(' ')[:sent_length])


    print("Loading pre-trained ELMO embeddings...")
    from allennlp.commands.elmo import ElmoEmbedder
    elmo = ElmoEmbedder(options_file='data/elmo_220m/options.json',weight_file='data/elmo_220m/weights.hdf5')

    #### ELMO vectors for training sentances
    vectors = elmo.embed_sentence(new_train_sents[0])
    elmo_train_x = np.zeros( (sent_length, elmo_dim) )
    elmo_train_x[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
    elmo_train_x = np.expand_dims(elmo_train_x, axis=0)
    for i in range(1,len(new_train_sents)):
        vectors = elmo.embed_sentence(new_train_sents[i])
        _x = np.zeros((sent_length, elmo_dim))
        _x[:vectors[elmo_layer].shape[0], :vectors[elmo_layer].shape[1]] = vectors[elmo_layer]
        _x = np.expand_dims(_x, axis=0)
        elmo_train_x = np.append(elmo_train_x,_x,axis=0)
        if (i % 10) == 0:
            sys.stdout.write('Train: [%d%%]\r' % int(round((i / len(new_train_sents)) * 100)))
            sys.stdout.flush()
    print('Train sentences '+str(elmo_train_x.shape))
    #elmo_test_x = elmo_train_x
    #elmo_dev_x = elmo_test_x

    #### ELMO vectors for test sentances
    vectors = elmo.embed_sentence(new_test_sents[0])
    elmo_test_x = np.zeros( (sent_length, elmo_dim) )
    elmo_test_x[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
    elmo_test_x = np.expand_dims(elmo_test_x, axis=0)
    for i in range(1,len(new_test_sents)):
        vectors = elmo.embed_sentence(new_test_sents[i])
        _x = np.zeros((sent_length, elmo_dim))
        _x[:vectors[elmo_layer].shape[0], :vectors[elmo_layer].shape[1]] = vectors[elmo_layer]
        _x = np.expand_dims(_x, axis=0)
        elmo_test_x = np.append(elmo_test_x,_x,axis=0)
        if (i % 10) == 0:
            sys.stdout.write('Test: [%d%%]\r' % int(round((i / len(new_test_sents)) * 100)))
            sys.stdout.flush()
    print('Test sentences '+str(elmo_test_x.shape))

    elmo_dev_x=elmo_test_x

    """
    #### ELMO vectors for dev sentances
    vectors = elmo.embed_sentence(new_dev_sents[0])
    elmo_dev_x = np.zeros( (sent_length, elmo_dim) )
    elmo_dev_x[:vectors[elmo_layer].shape[0],:vectors[elmo_layer].shape[1]]=vectors[elmo_layer]
    elmo_dev_x = np.expand_dims(elmo_dev_x, axis=0)
    for i in range(1,len(new_dev_sents)):
        vectors = elmo.embed_sentence(new_dev_sents[i])
        _x = np.zeros((sent_length, elmo_dim))
        _x[:vectors[elmo_layer].shape[0], :vectors[elmo_layer].shape[1]] = vectors[elmo_layer]
        _x = np.expand_dims(_x, axis=0)
        elmo_dev_x = np.append(elmo_dev_x,_x,axis=0)
        if (i % 10) == 0:
            sys.stdout.write('Dev: [%d%%]\r' % int(round((i / len(new_dev_sents)) * 100)))
            sys.stdout.flush()
    print('Dev sentences '+str(elmo_dev_x.shape))
    """

# code to embed pretrained embeddings
if include_w2v_emb =='yes':
    print("Loading pre-trained word embeddings...")
    import gensim
    word_model = gensim.models.KeyedVectors.load_word2vec_format('data/w2v_220m/urdu_220m_wv_100d', binary=False)
    pretrained_weights = word_model.wv.syn0

    print("Embedding Shape: "+str(pretrained_weights.shape))
    average_vector = numpy.average(pretrained_weights,axis=0, weights=None,returned=False)
    pretrained_weights[0]=average_vector
    pretrained_weights[1]=average_vector
    emb_list= sorted(pretrained_weights.tolist())

    print("Train vocab: "+str(len(vocab)))

    word2index={}
    for i, w in enumerate(sorted(word_model.wv.vocab)):
        word2index[w]= word_model.wv.vocab[w].index
    for i, w in enumerate(vocab): # Adding word from train set to total vocabulary
        if w not in word2index:
            word2index[w]= len(word2index)
            emb_list.append(average_vector.tolist())
    print("Total Vocabulary: "+str(len(word2index)))
    #print(len(emb_list))
    pretrained_weights=np.array(emb_list)
    print(pretrained_weights.shape)

    vocab_size, emdedding_size = pretrained_weights.shape
    vcb = word_model.wv.vocab

    # end of pretrained word embeddings
else: # include_w2v_emb =='no'
    word2index={}
    for i, w in enumerate(vocab):
        word2index[w]= i
    vocab_size =len(word2index)
    print("Total Vocabulary: "+str(len(word2index)))


tag2index = {t: i for i, t in enumerate(list(pos_tags))}
lu2index = {t: i for i, t in enumerate(list(lu_tags))}
seq2index = {t: i for i, t in enumerate(list(seq_tags))}

# train and test sentences to have number rather than words and tags for training
train_sentences_X, test_sentences_X, dev_sentences_X, train_tags_x, test_tags_x, dev_tags_x, train_lu_y, test_lu_y, dev_lu_y, train_seq_y, test_seq_y, dev_seq_y = [], [], [], [], [], [], [], [], [], [], [], []
for s in train_sents:
    s_int = []
    for w in s.split():
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sents:
    s_int = []
    for w in s.split():
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in dev_sents:
    s_int = []
    for w in s.split():
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    dev_sentences_X.append(s_int)

#print(train_sentences_X[0])
#print(train_chars[0])
#input()

for s in train_pos_tags:
    train_tags_x.append([tag2index[t] for t in s.split()])

for s in test_pos_tags:
    test_tags_x.append([tag2index[t] for t in s.split()])

for s in dev_pos_tags:
    dev_tags_x.append([tag2index[t] for t in s.split()])


for s in train_lu_tags:
    train_lu_y.append([lu2index[t] for t in s.split()])

for s in test_lu_tags:
    test_lu_y.append([lu2index[t] for t in s.split()])

for s in dev_lu_tags:
    dev_lu_y.append([lu2index[t] for t in s.split()])


for s in train_seq_tags:
    train_seq_y.append([seq2index[t] for t in s.split()])

for s in test_seq_tags:
    test_seq_y.append([seq2index[t] for t in s.split()])

for s in dev_seq_tags:
    dev_seq_y.append([seq2index[t] for t in s.split()])

print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_x[0])
print(test_tags_x[0])
print(train_lu_y[0])
print(test_lu_y[0])
print(train_seq_y[0])
print(test_seq_y[0])

# length of longest sentence to add padding because model takes the sentences with same length
MAX_LENGTH = len(max(train_sentences_X, key=len))
MAX_LENGTH = 100    # Limit number of hidden RNN layers
print("Maximum length of a sentence: ",MAX_LENGTH)  # 184
MAX_CHAR_LENGTH = 20 # number of chars for each word
chars = sorted(set([w_i for line in train_sents for w in line for w_i in w]))
n_chars = len(chars)
print("Total number of characters: ",n_chars)
char2index = {c: i + 2 for i, c in enumerate(chars)}
char2index["-OOV-"] = 1
char2index["-PAD-"] = 0
#print(len(char2index))

train_chars = []
for sentence in train_sents:
    sent_seq = []
    for i in range(MAX_LENGTH):
        word_seq = []
        for j in range(MAX_CHAR_LENGTH):
            try:
                word_seq.append(char2index.get(sentence[i][0][j]))
            except:
                word_seq.append(char2index.get("-PAD-"))
        sent_seq.append(word_seq)
    train_chars.append(np.array(sent_seq))

test_chars = []
for sentence in test_sents:
    sent_seq = []
    for i in range(MAX_LENGTH):
        word_seq = []
        for j in range(MAX_CHAR_LENGTH):
            try:
                if sentence[i][0][j] in char2index:
                    word_seq.append(char2index.get(sentence[i][0][j]))
                else:
                    word_seq.append(char2index.get("-OOV-"))
            except:
                word_seq.append(char2index.get("-PAD-"))
        sent_seq.append(word_seq)
    test_chars.append(np.array(sent_seq))

dev_chars = []
for sentence in dev_sents:
    sent_seq = []
    for i in range(MAX_LENGTH):
        word_seq = []
        for j in range(MAX_CHAR_LENGTH):
            try:
                if sentence[i][0][j] in char2index:
                    word_seq.append(char2index.get(sentence[i][0][j]))
                else:
                    word_seq.append(char2index.get("-OOV-"))
            except:
                word_seq.append(char2index.get("-PAD-"))
        sent_seq.append(word_seq)
    dev_chars.append(np.array(sent_seq))
# reshaping char vectors for training and testing
train_chars_X = np.array(train_chars).reshape((len(train_chars), MAX_LENGTH, MAX_CHAR_LENGTH))
test_chars_X = np.array(test_chars).reshape((len(test_chars), MAX_LENGTH, MAX_CHAR_LENGTH))
dev_chars_X = np.array(dev_chars).reshape((len(dev_chars), MAX_LENGTH, MAX_CHAR_LENGTH))

# Addign padding to train and test sentences
from keras.preprocessing.sequence import pad_sequences
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
dev_sentences_X = pad_sequences(dev_sentences_X, maxlen=MAX_LENGTH, padding='post')

train_tags_x = pad_sequences(train_tags_x, maxlen=MAX_LENGTH, padding='post')
test_tags_x = pad_sequences(test_tags_x, maxlen=MAX_LENGTH, padding='post')
dev_tags_x = pad_sequences(dev_tags_x, maxlen=MAX_LENGTH, padding='post')

train_lu_y = pad_sequences(train_lu_y, maxlen=MAX_LENGTH, padding='post')
test_lu_y = pad_sequences(test_lu_y, maxlen=MAX_LENGTH, padding='post')
dev_lu_y = pad_sequences(dev_lu_y, maxlen=MAX_LENGTH, padding='post')

train_seq_y = pad_sequences(train_seq_y, maxlen=MAX_LENGTH, padding='post')
test_seq_y = pad_sequences(test_seq_y, maxlen=MAX_LENGTH, padding='post')
dev_seq_y = pad_sequences(dev_seq_y, maxlen=MAX_LENGTH, padding='post')

print(train_sentences_X[0])
print(train_chars[0])
#input()
print(test_sentences_X[0])
print(train_tags_x[0])
print(test_tags_x[0])
print(train_lu_y[0])
print(test_lu_y[0])
print(train_seq_y[0])
print(test_seq_y[0])



# Definding BiLASTM model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Dropout,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
# # Defining model with BiLSTM

input_1 = Input(shape=(MAX_LENGTH,elmo_dim))    # input_shape=(2??,)
if include_w2v_emb=='yes':
    emb_1= Embedding(input_dim=vocab_size, output_dim=emdedding_size, input_length=MAX_LENGTH, weights=[pretrained_weights], trainable=False)(input_1)
else:
    emb_1= Embedding(input_dim=len(vocab), output_dim=50, input_length=MAX_LENGTH)(input_1)

if include_char_emb=='yes':
    # input and embeddings for characters
    char_in = Input(shape=(MAX_LENGTH, MAX_CHAR_LENGTH,))
    emb_char = TimeDistributed(Embedding(input_dim=len(char2index), output_dim=25, input_length=MAX_CHAR_LENGTH, mask_zero=True))(char_in)
    # character LSTM to get word encodings by characters
    emb_char=Dropout(0.25)(emb_char)
    char_enc = TimeDistributed(LSTM(units=64, return_sequences=False,recurrent_dropout=0.5))(emb_char)
    char_enc = Dropout(0.25)(char_enc)

    merged_1 = concatenate([emb_1, char_enc])

if include_pos_emb =='yes':
    input_2 = Input(shape=(MAX_LENGTH,),dtype='float32')    # input_shape=(2??,)
    emb_2= Embedding(input_dim=len(tag2index), output_dim=30, input_length=MAX_LENGTH)(input_2)
"""
if include_char_emb=='yes':
    if include_pos_emb =='yes':
        merged_2 = concatenate([merged_1, emb_2])
    else:
        merged_2 = merged_1
else:
    if include_pos_emb == 'yes':
        merged_2 = concatenate([emb_1, emb_2])
    else:
        merged_2 = emb_1
"""

merged_2 = Dropout(0.25)(concatenate([input_1,emb_2]))
main_LSTM= Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.5))(merged_2)
merged_3 = Dropout(0.25)(main_LSTM)
main_LSTM2= Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.5))(merged_3)
merged_4 = Dropout(0.25)(main_LSTM2)
#main_LSTM3= Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.5))(merged_4)
#merged_5 = Dropout(0.25)(main_LSTM3)
dense_out_lu = TimeDistributed(Dense(len(lu2index),activation='softmax'))(merged_4)

main_LSTM4= Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.5))(concatenate([merged_4, dense_out_lu]))
merged_6 = Dropout(0.25)(main_LSTM4)
dense_out_seq = TimeDistributed(Dense(len(seq2index),activation='softmax'))(merged_6)

model = Sequential()


if include_char_emb=='yes':
    if include_pos_emb == 'yes':
        model = Model(inputs=[input_1, char_in, input_2], outputs=[dense_out_lu,dense_out_seq])
    else:
        model = Model(inputs=[input_1, char_in], outputs=[dense_out_lu,dense_out_seq])
else:
    if include_pos_emb == 'yes':
        model = Model(inputs=[input_1, input_2], outputs=[dense_out_lu,dense_out_seq])
    else:
        model = Model(inputs=[input_1], outputs=[dense_out_lu,dense_out_seq])

#model = Model(inputs=[input_1, input_2], outputs=[dense_out_lu,dense_out_seq])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# training of the model

history = model.fit([elmo_train_x,train_tags_x], [to_categorical(train_lu_y, len(lu2index)),to_categorical(train_seq_y, len(seq2index))], batch_size=32, epochs=21, validation_data=([elmo_dev_x,dev_tags_x], [to_categorical(dev_lu_y, len(lu2index)),to_categorical(dev_seq_y, len(seq2index))]), shuffle = True)

#history = model.fit([train_sentences_X,train_tags_x], [to_categorical(train_lu_y, len(lu2index)),to_categorical(train_seq_y, len(seq2index))], validation_data=([dev_sentences_X,dev_tags_x], [to_categorical(dev_lu_y, len(lu2index)),to_categorical(dev_seq_y, len(seq2index))]), batch_size=32, epochs=10, shuffle = True)
model.save("trained_models/2+1_blstm-fun-pos-elmo-mtl21.model")

#print('Loading pretrained Labeling model ...')
#model = load_model("trained_models/3blstm-luseq.model")
#
# scores = model.evaluate([test_sentences_X,test_tags_x], to_categorical(test_seq_y, len(seq2index)))
# print(f"{model.metrics_names[1]}: {scores[1] * 100}")

print("Evaluating Trained/Loaded Labeling model...")
test_y_lu_pred = model.predict([elmo_test_x, test_tags_x])[0].argmax(-1)[test_sentences_X > 2]
test_y_seq_pred = model.predict([elmo_test_x, test_tags_x])[1].argmax(-1)[test_sentences_X > 2]#test_y_pred = model.predict([test_sentences_X,test_chars_X]).argmax(-1)[test_sentences_X > 2]

test_y_lu_true = test_lu_y[test_sentences_X > 2]
test_y_seq_true = test_seq_y[test_sentences_X > 2]

predictions_lu = model.predict([elmo_test_x, test_tags_x])[0]
predictions_seq = model.predict([elmo_test_x, test_tags_x])[1]
#predictions = model.predict([test_sentences_X,test_chars_X])
results_lu = logits_to_tokens(predictions_lu, {i: t for t, i in lu2index.items()})
results_seq = logits_to_tokens(predictions_seq, {i: t for t, i in seq2index.items()})
print('\n---- Result of BiLSTM-Sequence Labeling ----\n')
classification_report(test_y_seq_true, test_y_seq_pred, seq_tags)
classification_report(test_y_lu_true, test_y_lu_pred, lu_tags)

print("Writing labeling reuslts into a text file...")
out_file = open("trained_models/outputs/output.seq_r",'w')

for i,line in enumerate(test_sents):
    line= line.split()
    pos_line= test_pos_tags[i].split()
    index=i*len(line)
    for j,w in enumerate(line):
        if results_lu[i][j].strip()!="-EMPTY-":
            out_file.write(w + "\t" +results_lu[i][j]+"+"+ pos_line[j] + "\t" + results_seq[i][j]+ "\n")
        else:
            out_file.write(w + "\t" + pos_line[j] + "\t" + results_seq[i][j] + "\n")
    out_file.write("\n")
out_file.close()

"""
print(history.history.keys())
out_file = open("trained_models/train_history.txt",'w')
out_file.write("Train Accuracy:\n")
out_file.write(str(history.history['acc']))
out_file.write("\nVal Accuracy:\n")
out_file.write(str(history.history['val_acc']))

out_file.write("\nTrain Loss:\n")
out_file.write(str(history.history['loss']))
out_file.write("\nVal Loss:\n")
out_file.write(str(history.history['val_loss']))

out_file.close()
print('The Model is finished training and saved...')
"""
#print('The Model is being evaluated...')
import os
os.system("python labels2brackets.py")
os.system("./EVALB/evalb trained_models/outputs/gold.txt trained_models/outputs/output.brackets -p EVALB/new.prm")
#os.system("python dep2label-master/decode_labels2dep.py --input dep2label-master/toqeer_tests/output_label_relpos2.txt  --output dep2label-master/toqeer_tests/cand_label_relpos2.conll --encoding rel-pos")
#os.system("perl dep2label-master/toqeer_tests/eval-spmrl.pl -q -g dep2label-master/toqeer_tests/test_ref.conll -s dep2label-master/toqeer_tests/cand_label_relpos2.conll")
print('Results Saved ...')
