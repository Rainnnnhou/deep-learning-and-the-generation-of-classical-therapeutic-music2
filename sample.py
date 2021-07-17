import argparse
import os
import json

import numpy as np

from model import build_model, load_weights

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
from music21 import instrument, note, stream, chord

DATA_DIR = './data'
MODEL_DIR = './model'
OUTPUT_DIR = './output'

def build_sample_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))   #从稀疏矩阵到密集矩阵的过程，叫做embedding(升维/降维)
    for i in range(3):
        model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model

def char_idx_char_mappings():
    with open(os.path.join(DATA_DIR, 'char_to_idx.json')) as f:
        char_to_idx = json.load(f)   #json.load()方法就是将json文件对象转换为了python dict.
    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)   #vocab_size是char_to_idx字典的长度
    return vocab_size,idx_to_char,char_to_idx

def sample(epoch, header, num_chars,vocab_size,idx_to_char,char_to_idx):
    prediction = []
    

    model = build_sample_model(vocab_size)
    load_weights(epoch, model)
    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))

    sampled = [char_to_idx[c] for c in header]
    print('\n')
    print('*' * 100)
    for i in range(num_chars):
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = np.random.randint(vocab_size)
        result = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(vocab_size), p=result)
        sampled.append(sample)

    
    print('Model with epoch number : ' ,epoch)
    print('Generated notes : \n')
    for c in sampled:
        #print(c,end=',')
        prediction.append(idx_to_char[c])
    return prediction

def generateMidiFile(prediction_output,modelNo):
    
    """ convert the output from the prediction to notes and create a midi file
            from the notes """
    offset = 0   #偏移，防止数据覆盖
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:   #pattern相当于data
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():   #data中有.或者有数字
            notes_in_chord = pattern.split('.')   #用.分隔和弦中的每个音
            notes = []   #notes列表接收单音
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))   #把当前音符化成整数，在对应midi_number转换成note
                new_note.storedInstrument = instrument.Piano()   #乐器用钢琴
                notes.append(new_note)
            new_chord = chord.Chord(notes)   #再把notes中的音化成新的和弦
            new_chord.offset = offset   #初试定的偏移给和弦的偏移
            output_notes.append(new_chord)   #把转化好的和弦传到output_notes中
        # pattern is a note
        else:
            new_note = note.Note(pattern)   #note直接可以把pattern变成新的note
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()   #乐器用钢琴
            output_notes.append(new_note)   #把new_note传到output_notes中

        # increase offset each iteration so that notes do not stack
        offset += 0.5   #每次迭代都将偏移增加，防止交叠覆盖
                        # 指定两个音符之间的持续时间
    print('output_notes: \n' ,output_notes)
     #创建音乐流(stream)
    midi_stream = stream.Stream(output_notes)   #把上面的循环输出结果传到流
    outputFileName= OUTPUT_DIR+'/output_modelEpoch_No_'+str(modelNo)+'.mid'
    midi_stream.write('midi', fp=outputFileName)   #最终输出的文件名是output.mid，格式是mid

def predictModel(numChars):
    vocab_size,idx_to_char,char_to_idx = char_idx_char_mappings()
    modelWeights_epoch = [20,40,60,80,100]
    for epochs in modelWeights_epoch:
        prediction = sample(epochs,'',numChars,vocab_size,idx_to_char,char_to_idx)
        generateMidiFile(prediction , epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')   #创建解析器,创建一个 ArgumentParser 对象,ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    parser.add_argument('--chars', type=int, default=100, help='number of epochs to train for')   #给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的
    args = parser.parse_args()   #ArgumentParser 通过 parse_args() 方法解析参数
    OUTPUT_DIR = OUTPUT_DIR+'/Chars_'+str(args.chars)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
	
    predictModel(args.chars)