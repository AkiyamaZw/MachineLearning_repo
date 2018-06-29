import tensorflow as tf 
import numpy as np 
import xlrd
import os
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

data = "fire_theft.xls"
data_path = os.path.join(os.getcwd(),"data",data)
print(data_path)

def main():
    # process data
    x,y = readxls(data_path)
    x = x[1:]
    y = y[1:]
    print(max(x),max(y))
    # show the data
    plt.ion()


    # point of data
    #scat = plt.scatter(x,y)
 
    # train model
    linear_regression(x,y,100)
    plt.ioff()
    plt.show()

def readxls(data):
    workbook = xlrd.open_workbook(data_path)

    sheet_0 = workbook.sheet_by_index(0)
    print(sheet_0.name,sheet_0.nrows,sheet_0.ncols)

    x = sheet_0.col_values(0)
    y = sheet_0.col_values(1)
    return x,y

def linear_regression(x,y,epoch=10):
    a = np.linspace(0,40,100)
    x_ = tf.placeholder(shape=[42],dtype=tf.float32)
    y_ = tf.placeholder(shape=[42],dtype=tf.float32)
    n = len(x)
    with tf.variable_scope("regression"):
        w = tf.get_variable(name="w",shape=[],dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0.1,stddev=1))
        b = tf.get_variable(name="b",shape=[],dtype=tf.float32,
                            initializer=tf.constant_initializer(0))
        logits = w*x + b
    cost = tf.losses.mean_squared_error(y,logits)
    print(cost.get_shape())
    #train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    train_op = tf.train.AdamOptimizer(0.9).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            _,cost_,w_,b_ = sess.run([train_op,cost,w,b],feed_dict={x_:x,y_:y})
            print("at epoch %d, loss is %.4f"%(i,cost_))

            plt.cla()
            plt.title("linear regression")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.scatter(x,y)
            plt.plot(a,w_*a+b_,c='red')
            plt.text(20,20,"loss=%.2f"%cost_)
            plt.pause(0.1)

    

if __name__ == "__main__":
    main()
    
