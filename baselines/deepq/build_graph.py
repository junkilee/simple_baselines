"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= act (in case of parameter noise) ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)
    reset_ph: bool
        reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold_ph: float
        the desired threshold for the difference between non-perturbed and perturbed policy
    update_param_noise_scale_ph: bool
        whether or not to update the scale of the noise for the next time it is re-perturbed

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U

def default_param_noise_filter(var):
    if var not in tf.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        #TODO do I have to fix something here? yes.
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=[output_actions, update_eps_expr, eps],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act

def build_test_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None, test_epsilon=0.0):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0.0))

        q_func_results = q_func(observations_ph.get(), num_actions, scope="q_func")
        q_values = q_func_results['q']
        s_value = q_func_results['s']
        a_values = q_func_results['a']
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=[output_actions, q_values, s_value, a_values, update_eps_expr],
                         givens={update_eps_ph: test_epsilon, stochastic_ph: False},
                         updates=[update_eps_expr])
        return act

def build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None, param_noise_filter_func=None):
    """Creates the act function with support for parameter space noise exploration (https://arxiv.org/abs/1706.01905):

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float, bool, float, bool) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    if param_noise_filter_func is None:
        param_noise_filter_func = default_param_noise_filter

    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        update_param_noise_threshold_ph = tf.placeholder(tf.float32, (), name="update_param_noise_threshold")
        update_param_noise_scale_ph = tf.placeholder(tf.bool, (), name="update_param_noise_scale")
        reset_ph = tf.placeholder(tf.bool, (), name="reset")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        param_noise_scale = tf.get_variable("param_noise_scale", (), initializer=tf.constant_initializer(0.01), trainable=False)
        param_noise_threshold = tf.get_variable("param_noise_threshold", (), initializer=tf.constant_initializer(0.05), trainable=False)

        # Unmodified Q.
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")

        # Perturbable Q used for the actual rollout.
        q_values_perturbed = q_func(observations_ph.get(), num_actions, scope="perturbed_q_func")
        # We have to wrap this code into a function due to the way tf.cond() works. See
        # https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond for
        # a more detailed discussion.
        def perturb_vars(original_scope, perturbed_scope):
            all_vars = U.scope_vars(U.absolute_scope_name("q_func"))
            all_perturbed_vars = U.scope_vars(U.absolute_scope_name("perturbed_q_func"))
            assert len(all_vars) == len(all_perturbed_vars)
            perturb_ops = []
            for var, perturbed_var in zip(all_vars, all_perturbed_vars):
                if param_noise_filter_func(perturbed_var):
                    # Perturb this variable.
                    op = tf.assign(perturbed_var, var + tf.random_normal(shape=tf.shape(var), mean=0., stddev=param_noise_scale))
                else:
                    # Do not perturb, just assign.
                    op = tf.assign(perturbed_var, var)
                perturb_ops.append(op)
            assert len(perturb_ops) == len(all_vars)
            return tf.group(*perturb_ops)

        # Set up functionality to re-compute `param_noise_scale`. This perturbs yet another copy
        # of the network and measures the effect of that perturbation in action space. If the perturbation
        # is too big, reduce scale of perturbation, otherwise increase.
        q_values_adaptive = q_func(observations_ph.get(), num_actions, scope="adaptive_q_func")
        perturb_for_adaption = perturb_vars(original_scope="q_func", perturbed_scope="adaptive_q_func")
        kl = tf.reduce_sum(tf.nn.softmax(q_values) * (tf.log(tf.nn.softmax(q_values)) - tf.log(tf.nn.softmax(q_values_adaptive))), axis=-1)
        mean_kl = tf.reduce_mean(kl)
        def update_scale():
            with tf.control_dependencies([perturb_for_adaption]):
                update_scale_expr = tf.cond(mean_kl < param_noise_threshold,
                    lambda: param_noise_scale.assign(param_noise_scale * 1.01),
                    lambda: param_noise_scale.assign(param_noise_scale / 1.01),
                )
            return update_scale_expr

        # Functionality to update the threshold for parameter space noise.
        update_param_noise_threshold_expr = param_noise_threshold.assign(tf.cond(update_param_noise_threshold_ph >= 0,
            lambda: update_param_noise_threshold_ph, lambda: param_noise_threshold))

        # Put everything together.
        deterministic_actions = tf.argmax(q_values_perturbed, axis=1)
        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        updates = [
            update_eps_expr,
            tf.cond(reset_ph, lambda: perturb_vars(original_scope="q_func", perturbed_scope="perturbed_q_func"), lambda: tf.group(*[])),
            tf.cond(update_param_noise_scale_ph, lambda: update_scale(), lambda: tf.Variable(0., trainable=False)),
            update_param_noise_threshold_expr,
        ]
        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph, reset_ph, update_param_noise_threshold_ph, update_param_noise_scale_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True, reset_ph: False, update_param_noise_threshold_ph: False, update_param_noise_scale_ph: False},
                         updates=updates)
        return act

def build_act_ltl(make_obs_ph, q_func, num_actions, num_task_states, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    num_task_states: int
        number of states in task MDP
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        #TODO do I have to fix something here? yes.
        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")

        # if reshaped, q_values would be [-1, num_task_states, num_actions]
        deterministic_actions = tf.argmax(q_values, axis=-1) % num_actions

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=[output_actions, update_eps_expr, eps],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act


def build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
    double_q=True, scope="deepq", reuse=None, param_noise=False, param_noise_filter_func=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    if param_noise:
        act_f = build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse,
            param_noise_filter_func=param_noise_filter_func)
    else:
        act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
        print("q_t_selected shape: ", q_t_selected.shape, type(q_t_selected))

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        print("td error:", td_error.shape)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}


def build_train_ltl(make_obs_ph, q_func, num_actions, num_task_states, optimizer, grad_norm_clipping=None, gamma=1.0,
    double_q=True, scope="deepq", reuse=None, param_noise=False, param_noise_filter_func=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    num_task_states: int
        number of states in task MDP
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select an action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    if param_noise:
        act_f = build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse,
            param_noise_filter_func=param_noise_filter_func)
    else:
        act_f = build_act_ltl(make_obs_ph, q_func, num_actions, num_task_states, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")
        # placeholder for classified task states.
        transition_mats_ph = tf.placeholder(tf.float32, [None, num_task_states, num_task_states], name="trans_mats")


        # q network evaluation
        # new shape of qfunc output: |A| x |X| x batch_size

        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act

        # reshape q_t to be [batch_sz, num_task_states, num_actions]
        """
        Ex (when batch_sz = 1): 
        [a1x1 a1x2 ... a18x2] -> 
        [[a1x1 a2x1 ... a18x1],
         [a1x2 a2x2 ... a18x2]]
        """
        q_t = tf.reshape(q_t, [-1, num_task_states, num_actions])
        print("q_t shape:", q_t.shape)
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evaluation
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")

        # reshape q_tp1 to be [batch_sz, num_task_states, num_actions]
        q_tp1 = tf.reshape(q_tp1, [-1, num_task_states, num_actions])
        print("tp1 shape:", q_tp1.shape)
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))



        # q scores for actions which we know were selected in the given state.
        # shape should be [-1, num_task_states]
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 2)
        print("q_t_selected shape: ", q_t_selected.shape)
        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
            # gotta implement it here for double q too. TODO. (this is for target only).
            # what's happening here?
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 2)
            print("q_tp1_best shape:", q_tp1_best.shape)
            #  new shape of q_tp1_best output: [batch_size, |X|] (max q across actions for each x;
            #  so this is array [Q((s, x_1), _), ... , Q((s, x_n), _)] for each inputted s.

        print("done_mask shape:", done_mask_ph.shape)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best  # is this necessary?
        print("q_tp1_best_masked shape:", q_tp1_best_masked.shape)
        print("transition_mats_ph shape:", transition_mats_ph.shape)

        q_tp1_best_masked_xprime_expectation = tf.transpose(tf.reduce_sum(
            tf.multiply(tf.transpose(transition_mats_ph, [1, 0, 2]),
                        q_tp1_best_masked), axis=2))
        # compute RHS of bellman equation

        # target with combined planning/q-learning thing: ???
        # q_t_selected_target_tasks = rew_t_ph + gamma * sum(p(xprime) * q_tp1_best_masked)
        # so, really we need a 1D array of size |X| whose ith index maps to p(x_i | s', x)
        # do we need to know which x we are currently in?? we know s' right now at least.
        # we def need to know which x we are currently in--the question is then how do we know
        # which x we are in? 2 options: somehow have that be passed in, or recalculate it
        # how did it work before? --> I guess all of the x -> x' stuff just sorta happened
        #   to calculate reward and done, so what we're really doing before is instead of calculating
        #   an expectation of Q((s', x'), a) across x' we're just sampling one x' from the x' dist. across x, s.
        # When do we end?
        #   TBD
        # What really is P(x' | x, s')?
        #
        #   s' = s1, Classifier(s') = A (where A is some LTL proposition)
        #   Cols = x'
        #   [a1, b1, c1]
        #   [a2, b2, c2]   Rows = x
        #   [a3, b3, c3]
        #   a1 = P(x'| x) = P(x' = a | x = 1)
        #
        #   s' = s2, Classifier(s') = !A
        #   Cols = x'
        #   [a1, b1, c1]
        #   [a2, b2, c2]   Rows = x
        #   [a3, b3, c3]
        #
        #   a2s2 = P(x' | x, s') = P(x' = a | x = 2, s' = s2)
        #                        = P(x' = a | x = 2, classifier(s') = !A)

        # Transition matrix representing P(x'|x, s') is
        #   of shape: [num_possible LTL expressions, num_LTL_states, num_LTL_states]
        # THIS IS SOMETHING WE HAVE!!

        # Overall estimator equation for TARGET Q((s, x), a):
        # Q((s, x), a) = R(x) + gamma * sum_x'[P(x'|x, s') * max_a' Q((s', x'), a')],
        #   where R(x) = sum_x'[P(x'|x, s') * R(x, x')]
        #   where R(x, x') = 1 if x' = acc, x != acc

        # generate reward matrix (encoding R(x, x') formula)
        reward_mat = np.zeros([num_task_states, num_task_states])
        for x in range(1, num_task_states - 1):
            reward_mat[x][0] = 1

        # shape: [bs, num_task_states, num_task_states], [num_task_states, num_task_states]
        #           -> [bs, num_task_states]
        reward_x = tf.reduce_sum(tf.multiply(transition_mats_ph, reward_mat), axis=2)
        # def reward(x, next_x):
        #     """
        #     Method to return the reward R(x, x').
        #     :param x: current task state
        #     :param next_x: next task state
        #     :return: 1 if next task state is ACC (1) and current task state is not ACC,
        #                 otherwise 0.
        #     """
        #     return 1 if next_x == 1 and x != 1 else 0
        #
        # # REWARD:
        # # axis 0 gets you above or belowground matrix,
        # #   possible values = [0, num_possible_LTL_exps - 1]
        # # axis 1 gets you the row of probs for a given x; i.e. when axis 1 = k, then x = k.
        # #   possible values = [0, num_task_states - 1]
        # # axis 2 gets you the probability for a given x'
        # #   possible values = [0, num_task_states - 1]
        # # Example:
        # #   transition_mats[i][j][k] = probability of transitioning from state j to state k given
        # #       we are in LTL expression i.
        #
        # def reward(x, transition_mats, next_state):
        #     """
        #     Method to return the reward R(x) = sum_x'[P(x'|x, s') * R(x, x')]
        #     :param x:
        #     :param transition_mats:
        #     :return:
        #     """
        #     total = 0.0
        #     for next_x in range(0, num_task_states):
        #         total += reward(x, next_x) * transition_mats[classify(next_state)][x][next_x]
        #     return total



        # What do we have?
        #   - we have s, a, r, s' (an experience or transition sampled from the real dynamics by playing the game)
        #   - we don't have one specific x, but we can actually do updates for all x!!!
        #   - we have transitions = P(x'|x, s')
        #   - we have a list of all possible x' (let's call this next_x_list):
        #       multiple different representations:
        #       1. [0, 1, ... , |X| - 1],
        #       where each index represents both a specific state x (i.e. acc or rej)
        #       and is the index in P(x'|x, s') corresponding to it.
        #       2. [0, 1, ... , |X| - 1], but with constants for each index (i.e. ACC = 0, REJ = 1, ... )
        #           pro: basically 1 but a bit more understandable
        #       3. some weird thing with a dict?
        #           con: but then that changes the representation of our P(x'|x, s)
        #   - from q_func, we have Q((s', x'), a')
        #       (given s', q_func will spit out Q((s', x'), a') for all x', a'.
        #       q_func_out = q_func(obs_tp1_input.get(), num_actions, num_x_states,
        #                            scope="target_q_func")
        #   - from q_func_out, we can get sum_x'[P(x'|x, s') * max_a' Q((s', x'), a')]
        #       in 3 simple steps:
        #           1. Reshape output of q_func(s') from [|A| * |X|] to [|X|, |A|] by doing
        #               reshaped_out = tf.reshape(q_func_out, shape = [|X|, |A|])
        #           2. Get array rep. max_a' Q((s', x'), a') for all x' (shape = [|X|]) by doing
        #               q_func_out_max_a = tf.reduce_max(reshaped_out, axis=1)
        #           3. total = 0
        #              for next_x in next_x_list:
        #                  total += next_x_list[classify(next_state)][x][next_x] * q_func_out_max_a[next_x]
        #              return total
        #              *NOTE*
        #              in this case, classify(next_state) is returning an int representing
        #              which x->x' matrix to use. This may not be best.
        #              Alternatively, we could have it return (A, B), and have each matrix as the value to a dict whose
        #               key is a tuple of (A is satisfied, B is satisfied, ..., Z is satisfied).
        #       So, we have sum_x'[P(x'|x, s') * max_a' Q((s', x'), a')]!
        #   - R(x, a) = R((s, x), a). Well, you know what x is from s, so you can
        #       just in the wrapper 1) A = classify x from s. and
        #                           2) make rew appropriate value based on A
        #       NOTE:
        #       normally, rew from the env.step is R(s, a, s').
        #       So, we're rewriting R(s, a, s') to be 1 if classify(s') = acc and 0 otherwise
        #           (and with the whole not already in acc)
        #       I think the subtlety of Q(s, a) depending on R(s, a, s') is not something needed to be
        #       worried about in this modification because it worked before.
        #       CORRECTION:
        #       R(x) = sum_x' (P(x' | x, s') * R(x, x')) where R(x, x') = 1 if x' = acc and x != acc, 0 otherwise
        #   - Huh, so is that it?!
        #   - When do we terminate? What do we even mean by terminate here?
        #       When figuring out when to terminate an episode, we can potentially try only
        #           setting done = True when s s.t. x = acc.
        #               *does acc only occur when the LTL proposition is satisfied?
        #                no probability about it/"accidental" transitions to ACC?
        #       But then does it make sense trying to calculate Q((s, REJ), a) if s s.t. x = acc?
        #       Are we doing a lot of wasted computation?
        #       Terminate only at "reasonable" number of steps (according to ML).
        #       Hmmmmmmmmmm...
        #       blabber blabber blabber?
        #       if x == acc, then done = True
        #           *does this even make sense? bc we're iterating through all x?
        #       So only terminate upon acc?
        #       if x == rej,
        #       blabber blabber blabber!

        # dims: [bs, num_task_states]
        q_t_selected_target = reward_x + gamma * q_tp1_best_masked_xprime_expectation

        """
        BIGGEST QUESTION NOW:
        Q: Where does the classifier come in?
        A: Given s', you need to know which x it is for the P(x'|x, s'). you need to know 
        what the LTL expression status is at this point in time, so you can pick 
        the right array out of the stack of [x by x'] arrays.
        
        How do you know which x it is in?
        1. Somehow classify the 84x84x4 images. Sounds hard/annoying/lossy by hand. I guess with a NN
            you could just feed in the most recent image but this still sounds kinda suck 
            bc you lose a lot of info (color, 1/4 the data, etc.)
        2. Classify the 210x160x3 image, and then let state = (s, x) from then on.
            So, env.step(action) returns (state, x), rew, done, info, and then 
             you can just feed state into q_func, etc, and you can use x to pick from there.
            Cons:
             You're gonna have to edit the other wrappers to do stuff and actually work when the 
             state is a tuple not just a single nparray thing.
        3. OOH! Put it in the info!!! YES! SO MUCH EASIER!!
            info["task_state"] = x
            Does this work with the making state 4 images thing? Well, info 
             should just be independent of all of that so hopefully it will still work. i.e. the 4 images thing
             is just an INJECTIVE function of your current 210x160x3 state, so 
             when you set info["task_state"] to x, that should correspond to the correct x. WOOOHOO!!!!!

            
        """

        # For predicted Q((s, x), a):
        # Get Q((s, x), a) for all x, a, using q_func(s)
        #   (where s is a single state; in reality, s will be an array of many states.
        #    For simplicity, though, start with just s = 1 state)
        #   q_t = q_func(obs_t_input.get(), num_actions, num_x_states, scope="q_func", reuse=True)  # reuse parameters from act
        # Reshape output of q_func(s) from [|A| * |X|] to [|X|, |A|] by doing
        #   reshaped_q_t = tf.reshape(q_t, shape = [|X|, |A|])
        # Get Q((s, x), a) for specific a by using one-hot matmul or array indexing thing.
        #   q_t_selected = tf.reduce_sum(reshaped_q_t * tf.one_hot(act_t_ph, num_actions), 1)?
        # and we gucci good!

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                transition_mats_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}
