import os
import json
import argparse
from music21 import converter, instrument, note, chord
import numpy as np
import pickle
from model import build_model, save_weights
from datetime import datetime
import glob

DATA_DIR = './data'   #相对路径    绝对路径：D/yichao/.../data
LOG_DIR = './logs'
BATCH_SIZE = 16
SEQ_LENGTH = 64

class TrainLogger(object):             #训练记录
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:   #r > read   w>write  a>append
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):    #添加条目
        self.epochs += 1   #a+=1 > a=a+1 自加运算
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)

    
def log_notes_information(file, msg):  #记录音符信息
        with open(file, 'a') as f:
            f.write(msg)

def read_midi_files():   #读取midi文件
    start = datetime.now()   #预测/生成音乐的起始点
    notes=[]             #建立关于音符的空列表
    j=0
    print('Reading MIDI Files')
    for file in glob.glob(DATA_DIR +'/*.mid'):  #glob.glob(pathname) 返回所有匹配的文件路径列表
                                                #读取data文件夹中所有的mid文件,file表示每一个文件
        j += 1
        #if j % 5 == 0:
        #    print('%d  files processed ' %(j))

        print('Processing file : ' ,file)
        midi = converter.parse(file)   #midi文件的读取，解析，输出stream的流类型

        notes_to_parse = None   #获取所有的乐器部分，开始测试的都是单轨的
        # file has instrument parts
        try: 
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()   #如果有乐器部分，取第一个乐器部分
        # file has notes in a flat structure
        except: 
            notes_to_parse = midi.flat.notes   #纯音符组成

        for element in notes_to_parse:   #notes本身不是字符串类型
            if isinstance(element, note.Note):
                # 格式例如：E6
                notes.append(str(element.pitch))   #如果是note类型，取它的音高(pitch)
            elif isinstance(element, chord.Chord):
                # 转换后格式：45.21.78(midi_number)
                notes.append('.'.join(str(n) for n in element.normalOrder))   #用.来分隔，把n按整数排序
    
    print('All files read. Duming notes to file')
    with open(os.path.join(DATA_DIR, 'notes'), 'wb') as filepath:   #从路径中打开文件，写入
        pickle.dump(notes, filepath)   #把notes写入到文件中
    print('Total Time Taken : %s ' %(str(datetime.now() -start)))
    return notes   #返回提取出来的notes列表
			
def char_index_char_mapping(notes):   #字符索引字符映射
    msg =''
    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(notes)))) }   #sorted把notes中的所有音符做集合操作，去掉重复的音，然后按照字母顺序排列
                                                                                 # 创建一个字典，用于映射 音高 和 整数
    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }   #输出的时候反向索引
    vocab_size = len(char_to_idx)

    uniqueNotesLen = len(set(notes))   #uniqueNotesLen=num_pitch
    msg='\nTotal notes length : ' + str(len(notes))
    msg= msg + '\nTotal unique notes length : ' + str(len(char_to_idx))
    msg = msg + '\n*' *100
    msg = msg + '\nUnique notes : \n' + str(char_to_idx)
    msg = msg + '\n*' *100

    log_notes_information('NotesInfo.txt',msg)
    return char_to_idx,idx_to_char,vocab_size,uniqueNotesLen

def read_batches(T, vocab_size):
    length = T.shape[0]; 
    batch_chars = int(length / BATCH_SIZE);

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): 
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) 
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) 
        for batch_idx in range(0, BATCH_SIZE): 
            for i in range(0, SEQ_LENGTH): 
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] 
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y

def train(notes,char_to_idx,uniqueNotesLen, epochs=100, save_freq=10):

    #model_architecture
    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    #Train data generation
    T = np.asarray([char_to_idx[c] for c in notes], dtype=np.int32) #convert complete text into numerical indices
    #T_norm = T / float(uniqueNotesLen)  #T=prediction_input,uniqueNotesLen=num_pitch  输入归一化
    print("Length of text:" + str(T.size)) 
    print("Length of unique test: ," ,uniqueNotesLen)

    steps_per_epoch = (len(notes) / BATCH_SIZE - 1) / SEQ_LENGTH   
    print('Steps per epoch : ' ,steps_per_epoch)

    log = TrainLogger('training_log.csv')

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        
        losses, accs = [], []
        msg = ""
        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            
            print(X);

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))
        
        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))  #用checkpoint(检查点)文件在每一个Epoch结束时保存模型的参数
                                                                             #不怕训练过程中丢失模型参数，当对loss损失满意的时候可以随时停止训练

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
	
    notes = read_midi_files()
    char_to_idx,idx_to_char,vocab_size,uniqueNotesLen = char_index_char_mapping(notes)
    train(notes,char_to_idx,uniqueNotesLen, args.epochs, args.freq)
