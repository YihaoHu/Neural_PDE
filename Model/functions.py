# functions.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
#from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from sklearn.metrics import mean_squared_error
from keras.losses import mean_absolute_percentage_error

output ='../graphs/'

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
    print(len(dat),len(dat[0]))
    final_x = [d[-t1-t2:-t2] for d in dat]
    final_y = [d[-t2:] for d in dat]
    final_x = np.array(final_x).reshape(1, len(final_x),len(final_x[0]))
    final_y = np.array(final_y).reshape(1, len(final_y),len(final_y[0]))
    dat = [d[:-t1-t2] for d in dat]
    print(len(dat),len(dat[0]))
    return final_x, final_y, dat

def flattern(p):
    pred_y_matrix = [[]for _ in range(len(p[0])) ]
    for pp in p:
        a = pp.tolist()
        #a = np.array(pp).T.tolist()
        for m in range(len(a)):
            pred_y_matrix[m] += a[m]
    return pred_y_matrix

def data_normalize(Dat):
    new_dat  = []
    for d in Dat:
        temp = []
        for val in d: temp.append(100.0*np.tanh(val) ) #temp.append((val - min)/a )
        new_dat.append(temp)
    return new_dat

#heatmap plots
def multi_heatmap(Test_y, Pred_y, plot_name):
    Test_y = Test_y.reshape(Test_y.shape[0],Test_y.shape[1],Test_y.shape[2])
    Pred_y = Pred_y.reshape(Pred_y.shape[0],Pred_y.shape[1],Pred_y.shape[2])
    py = flattern(Pred_y)
    ty = flattern(Test_y)
    #Plot the new heatmap of predict data vs test data
    plt.figure()
    print(len(py),len(ty))
    ax1 = sns.heatmap(np.array(ty).T,vmin = 0, vmax = 1)
    ax1.set_title('Exact Data')
    ax1.set(xlabel='X (grid point)', ylabel='Time Step')
    f1 = ax1.get_figure()
    #f1.savefig(output + str(plot_name) + '_Exact_heatmap.pdf',bbox_inches='tight')
    plt.show()
    plt.figure()
    ax2 = sns.heatmap(np.array(py).T,vmin = 0, vmax = 1)
    ax2.set_title('Predicted Data')
    ax2.set(xlabel='X (grid point)', ylabel='Time Step')
    f2 = ax2.get_figure()
    #f2.savefig(output + str(plot_name) + '_Predicted_heatmap.pdf',bbox_inches='tight')
    plt.show()
    mse_lis = []
    for i in range(len(py)):
        mse_lis.append(mean_squared_error(py[i], ty[i]))
    return mse_lis

def stacked_LSTM(X, Y):
    time_step = X.shape[1]
    input_dim = X.shape[2]
    out = Y.shape[2]
    #Bidirectional LSTM
    #opt = SGD(clipvalue=5)
    #opt = optimizers.Adam(learning_rate=1e-6, clipvalue=.5)
    start = time.time()
    model = Sequential()
    #model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(20,activation='tanh', input_shape=(time_step, input_dim),return_sequences=True)))
    model.add(Bidirectional(LSTM(20,activation='tanh', input_shape=(time_step, input_dim),return_sequences=True)))
    #model.add(LSTM(64,activation='relu', input_shape=(time_step, input_dim),return_sequences=True))
    #model.add(LSTM(64,activation='relu', input_shape=(time_step, input_dim),return_sequences=True))
    model.add(Dense(out))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(X, Y,epochs=100, validation_split=.2,
              verbose=1, batch_size=2)
    model.summary()
    end = time.time()
    print("Total compile time: --------", end - start, 's')
    return model, hist


def DE_Learner(data, train_time, predict_time, stride, test, plot_name):
    print('###########################START##########################')
    data_x, data_y = data_split(data, train_time,predict_time, stride )
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test)
    print('Whole data size(batch, row, column)',data_x.shape, data_y.shape)
    print('Train data size(batch, row, column)',train_x.shape, train_y.shape)
    print('test data size(batch, row, column)',test_x.shape, test_y.shape)
    model, hist = stacked_LSTM(train_x,train_y)
    pred_y = model.predict(test_x, verbose=1)
    error = multi_heatmap(test_y, pred_y, plot_name)
    py = flattern(pred_y)
    ty = flattern(test_y)
    fig1 = plt.figure()
    for j in range(len(ty)):
        plt.scatter(range(len(ty[j])),[ty[j][i]-py[j][i] for i in range(len(ty[j]))])
    plt.title('Test Errors')
    plt.ylim(-.5,.5)
    #fig1.savefig(output + str(plot_name)+'_Test_Error.pdf',bbox_inches='tight')
    plt.show()
    fig2 = plt.figure()
    #plot loss history
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'cv'], loc='upper right')
    #fig2.savefig(output + str(plot_name)+'_Train_Error.pdf',bbox_inches='tight')
    plt.show()
    print('----------------MSE: ', np.mean(error))
    py = np.array(py).T
    ty = np.array(ty).T
    # with open(str(plot_name) + "_mse.txt", "w") as txt_file:
    #     txt_file.write('mse: ' + str(np.mean(error)) + "\n")
    #     for m in error:
    #         txt_file.write(str(m) + "\n")
    print('###########################END##########################')
    return py, ty, error, model
