import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding


MODEL_DIR = './model'

def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def load_weights(epoch, model):
    model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def build_model(batch_size, seq_len, vocab_size):
    model = Sequential()   #建立模型
    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))
    for i in range(3):
        model.add(LSTM(256, return_sequences=True, stateful=True))   #LSTM层神经元的数目是256，也是LSTM层输出的维度
                                                                     #return_sequences返回控制类型，此时是返回所有的输出序列
        model.add(Dropout(0.2))   #丢20%神经元，防止过拟合

    model.add(TimeDistributed(Dense(vocab_size)))   #输出的数目等于所有不重复的音调数  vocab_size=num_pitch
    model.add(Activation('softmax'))   #Softmax激活函数求概率  #softmax第一步将预测结果全部转化为非负数，第二步将转换后的结果归一化
    return model

if __name__ == '__main__':
    model = build_model(16, 64, 50)
    model.summary()
