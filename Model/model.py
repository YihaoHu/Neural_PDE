# model.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
#from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Bidirectional
from sklearn.metrics import mean_squared_error
from keras.losses import mean_absolute_percentage_error

output ='./graphs/'

#split the data by training time and test time period

def data_split(dat, train_hour, test_hour,stride):
    X, Y = [], []
    period = train_hour + test_hour
    i = 0
    while i + period < len(dat[0]):
        X.append(dat[:, i:i + train_hour])
        Y.append(dat[:, i+train_hour:i+period])
        i += stride
    return np.array(X), np.array(Y)

def split_final(dat, t1,t2):
    # To test the final state
    final_x = np.array(dat[:, -t1-t2:-t2]).reshape((1, dat.shape[0], t1))
    final_y = np.array(dat[:, -t2:]).reshape((1, dat.shape[0], t2))
    dat = dat[:, :-t1-t2]
    return final_x, final_y, dat

def flatten(y):
    return np.swapaxes(y, 0, 1).reshape(y.shape[1], y.shape[0]*y.shape[2])

def data_normalize(Dat):
    new_dat  = []
    for d in Dat:
        temp = []
        for val in d: temp.append(100.0*np.tanh(val) ) #temp.append((val - min)/a )
        new_dat.append(temp)
    return new_dat

#heatmap plots
def multi_heatmap(Test_y, Pred_y, plot_name):
    sns.set(font_scale = 2)
    py = flatten(Pred_y)
    ty = flatten(Test_y)
    #Plot the new heatmap of predict data vs test data
    plt.figure()
    print(len(py),len(ty))
    ax1 = sns.heatmap(np.array(ty).T,vmin = 0, vmax = 1)
    ax1.set_title('Exact Data')
    ax1.set(xlabel='X (grid point)', ylabel='Time Step')
    ax1.tick_params(labelsize=15)
    f1 = ax1.get_figure()
    f1.savefig(output + str(plot_name) + '_Exact_heatmap.png', bbox_inches='tight')
    #plt.show()
    plt.figure()
    ax2 = sns.heatmap(np.array(py).T,vmin = 0, vmax = 1)
    ax2.set_title('Predicted Data')
    ax2.set(xlabel='X (grid point)', ylabel='Time Step')
    ax2.tick_params(labelsize=15)
    f2 = ax2.get_figure()
    f2.savefig(output + str(plot_name) + '_Predicted_heatmap.png',bbox_inches='tight')
    #plt.show()
    return 

def stacked_LSTM(X, Y, training_epoch):
    time_step = X.shape[1]
    input_dim = X.shape[2]
    out = Y.shape[2]
    #Bidirectional LSTM
    #opt = SGD(clipvalue=5)
    #opt = optimizers.Adam(learning_rate=1e-6, clipvalue=.5)
    start = time.time()
    model = Sequential()
    #model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(20,activation='tanh', input_shape=(time_step, input_dim), return_sequences=True)))
    model.add(Bidirectional(LSTM(20,activation='tanh', input_shape=(time_step, input_dim), return_sequences=True)))
    model.add(Dense(out))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(X, Y, epochs=training_epoch, validation_split=.2,
              verbose=1, batch_size=2)
    model.summary()
    end = time.time()
    print("Total compile time: --------", end - start, 's')
    return model, hist

# with self attention layer
# def stacked_LSTM(X, Y, training_epoch):
#     time_steps = X.shape[1]
#     input_dim = X.shape[2]
#     out = Y.shape[2]
    
#     model_input = Input(shape=(time_steps, input_dim))
#     x = LSTM(64, return_sequences=True)(model_input)
#     x = Attention(32)(x)
#     x = Dense(out)(x)
#     model = Model(model_input, x)
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     hist = model.fit(X, Y, epochs=training_epoch, validation_split=.2,
#               verbose=1, batch_size=2)
#     model.summary()
#     end = time.time()
#     print("Total compile time: --------", end - start, 's')
#     return model, hist


def DE_Learner(data, train_time, predict_time, stride, test_portion, training_epoch, plot_name, plot = 0):
    print('###########################START##########################')
    data_x, data_y = data_split(data, train_time,predict_time, stride )
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_portion)
    print('Whole data size(batch, row, column)',data_x.shape, data_y.shape)
    print('Train data size(batch, row, column)',train_x.shape, train_y.shape)
    print('test data size(batch, row, column)',test_x.shape, test_y.shape)
    model, hist = stacked_LSTM(train_x, train_y, training_epoch)
    pred_y = model.predict(test_x, verbose=1)
    error = mean_squared_error(flatten(test_y), flatten(pred_y))
    py = flatten(pred_y)
    ty = flatten(test_y)
    if plot:
        multi_heatmap(test_y, pred_y, plot_name)
        fig1 = plt.figure()
        for j in range(ty.shape[0]):
            plt.scatter(range(ty.shape[1]), ty[j, :]-py[j, :])
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick',labelsize=15)
        plt.title('Test Errors')
        plt.ylim(-.5, .5)
        #fig1.savefig(output + str(plot_name)+'_Test_Error.pdf',bbox_inches='tight')
        plt.show()
    fig2 = plt.figure()
    #plot loss history
    sns.set(font_scale = 2)
    plt.plot(np.log(hist.history['loss']))
    plt.plot(np.log(hist.history['val_loss']))
    #plt.rc('axes', axisbelow=True)
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    plt.title('Model Log Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'cv'], loc='upper right')
    fig2.savefig(output + str(plot_name)+'_Train_Error.pdf',bbox_inches='tight')
    plt.show()
    print('----------------MSE: ', np.mean(error))
    py = np.array(py).T
    ty = np.array(ty).T
    print('###########################END##########################')
    return py, ty, error, model
