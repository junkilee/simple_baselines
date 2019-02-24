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

    # """
    # action selection for test sampling:
    # ([bs, num_task_states, num_actions], [bs]) --something-> [bs, num_actions] --argmax-> [bs]
    # """
    #
    # a = [[[1.,2],
    #       [3,4]],
    #      [[5,6],
    #       [7,8]],
    #      [[9,100],
    #       [35,12]]]
    #
    # chosen = [0, 1, 1]
    # chosen_one_hot = tf.one_hot(indices=chosen, depth=2)
    # result = tf.transpose(tf.reduce_sum(chosen_one_hot * tf.transpose(a, [2,0,1]), axis=-1))
    # result_actions = tf.argmax(result, axis=1)
    #
    # print(result_actions.eval())

    # """
    # action selection for test sampling without fancy matrix mult stuff:
    # ([bs, num_task_states, num_actions], [bs]) --selection-> [bs, num_actions] --argmax-> [bs]
    # """
    # a = tf.constant([[[1., 2],
    #       [3, 4]],
    #      [[5, 6],
    #       [7, 8]],
    #      [[9, 100],
    #       [35, 12]]])
    #
    # task_state = tf.Variable(0)
    # sliced_by_state = a[:,0,:]
    # result = tf.argmax(sliced_by_state, axis=1)
    # print(result.eval())

    """
    action selection using expectations:
    ([bs, num_task_states, num_actions], [num_task_states]) --selection-> [bs, num_actions] --argmax-> [bs]
    
    getting the next_prob_task_state vector:
    ([num_task_states], [num_task_states, num_task_states]) --> [num_task_states] 
    """
    a = tf.constant([[[1., 2, 10],
          [3, 4, 10]],
         [[5, 6, 10],
          [7, 8, 10]],
         [[9, 100, 10],
          [35, 12, 10]]])
    p = tf.constant([0.5, 0.5])
    trans_mat = tf.constant([[0.1, 0.9], [1, 0]])
    expected_q_values_across_states = tf.tensordot(p, a, axes=[0, 1])

    # expected expected_q_values_across_states:
    # [[ 2.  3. 10.]
    #  [ 6.  7. 10.]
    #  [22. 56. 10.]]

    best_actions = tf.argmax(expected_q_values_across_states, axis=-1)

    # expected actions:
    # [2, 2, 1]

    print("Best actions per batch element:", best_actions.eval())

    next_p = tf.tensordot(p, trans_mat, axes=1)
    # expected next_p:
    # [0.55, 0.45]
    print("Next probabilities:", next_p.eval())

