import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    # rewards
    # a = [[[1,2],[3,4]], [[5,6],[7,8]]]
    # b = [[1, 2], [3, 4]]
    # print(tf.reduce_sum(tf.multiply(a, b), axis = 2).eval())

    # # q((s',x'), a')
    # a = [[[1],
    #       [3]],
    #      [[5],
    #       [7]],
    #      [[9],
    #       [11]]]
    # b = [[1, 2],
    #      [3, 4],
    #      [5, 6]]
    # # print(tf.transpose(a, [1,0,2]).eval())
    # print(tf.transpose(tf.reduce_sum(tf.multiply(tf.transpose(a, [1,0,2]), b), axis=2)).eval())

    # action selection argmax
    ax = [list(range(1, 37)), list(range(37, 73))]
    ax[0][0] = 123
    ax[1][10] = 123
    print(ax)
    print(tf.argmax(ax, axis=-1).eval() % 18)
    