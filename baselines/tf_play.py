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

    # # action selection argmax
    # ax = [list(range(1, 37)), list(range(37, 73))]
    # ax[0][0] = 123
    # ax[1][10] = 123
    # print(ax)
    # print(tf.argmax(ax, axis=-1).eval() % 18)

    # # * test
    # a = tf.constant([1,2])
    # b = tf.constant([[4,5,6], [7,8,9]])
    # print(a.shape, b.shape)
    # print((np.broadcast_to(a, (2, 3)) * b).eval())
    #
    # print(tf.transpose((a * tf.transpose(b))).eval())
    # print((a * tf.reduce_sum(b, axis=1)).eval())
    # print(tf.reduce_mean(a * tf.reduce_sum(b, axis=1)).eval())


    """
    # selected qs (one hot stuff)
    a = [[[1.,2],
          [3,4]],
         [[5,6],
          [7,8]],
         [[9,10],
          [11,12]]]
    # asimp = [[1, 2], [5, 6], [9, 10]]
    chosen = [0, 1, 1]
    result = tf.expand_dims(tf.one_hot(chosen, 2), -1)
    # shape= (3,2)
    result = tf.squeeze(tf.matmul(a, result))
    print(result.eval())
    # expected: [[1,3], [6,8], [10,12]], shape = [bs, x]
    """

    """
    # broadcasting
    #   x = tf.constant([1, 2, 3])
    # y = tf.broadcast_to(x, [3, 3]).eval()
    # print(y)
    # array([[1, 2, 3],
    #        [1, 2, 3],
    #        [1, 2, 3]], dtype=int32)
    """

    """
    action selection for test sampling:
    ([bs, num_task_states, num_actions], [bs]) --something-> [bs, num_actions] --argmax-> [bs]
    """

    a = [[[1.,2],
          [3,4]],
         [[5,6],
          [7,8]],
         [[9,100],
          [35,12]]]

    chosen = [0, 1, 1]
    chosen_one_hot = tf.one_hot(indices=chosen, depth=2)
    result = tf.transpose(tf.reduce_sum(chosen_one_hot * tf.transpose(a, [2,0,1]), axis=-1))
    result_actions = tf.argmax(result, axis=1)

    print(result_actions.eval())