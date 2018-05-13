'''This code was generated from the notebook 2018-05-13-viterbi.ipynb'''
import numpy as np
import matplotlib.pyplot as plt

import nb_code.hmm_alpha_recursion as prev_post

def viterbi(visibles, p_hidden_start, p_transition, p_emission):
    # use log probability to avoid overflow
    eps = 1e-9
    log_p_hidden_start = np.log(p_hidden_start + eps)
    log_p_transition = np.log(p_transition + eps)
    log_p_emission = np.log(p_emission + eps)
    
    num_timestamps = visibles.shape[0]
    num_hidden_states = log_p_transition.shape[0]
    
    # I'll collect 
    best_log_prob_per_time_and_state = np.zeros((num_timestamps, num_hidden_states))
    best_state_by_time_and_prev_state = np.zeros((num_timestamps, num_hidden_states), dtype=int)
    
    most_likely_states = np.zeros((num_timestamps,), dtype=int)

    # alpha(h_1) = p(h_1) * p(v_1 | h_1)
    best_log_prob_per_time_and_state[0] = log_p_hidden_start + log_p_emission[visibles[0]]
    
    for t in range(1, num_timestamps):
        for new_state in range(num_hidden_states):
            r = [
                best_log_prob_per_time_and_state[t - 1][old_state] 
                + log_p_transition[new_state][old_state] 
                + log_p_emission[visibles[t], new_state]
                for old_state in range(num_hidden_states)
            ]
            best_log_prob_per_time_and_state[t][new_state] = np.max(r)
            best_state_by_time_and_prev_state[t][new_state] = np.argmax(r)

    # find the last state
    most_likely_states[-1] = np.argmax(best_log_prob_per_time_and_state[-1])
    print(most_likely_states[-1])
    
    for t in range(num_timestamps - 1, 0, -1):
        print('next timestamp is', t - 1)
        print('currently at {}'.format(most_likely_states[t]))
        
        most_likely_states[t - 1] = best_state_by_time_and_prev_state[t - 1][most_likely_states[t]]

    return most_likely_states

most_likely_states = viterbi(
    prev_post.visibles, 
    prev_post.p_hidden_start,
    prev_post.p_transition,
    prev_post.p_emission,
)

print(most_likely_states)

def viterbi(visibles, p_hidden_start, p_transition, p_emission):
    num_timestamps = visibles.shape[0]
    num_hidden_states = p_transition.shape[0]
    
    max_prob_per_time_and_state = np.zeros((num_timestamps, num_hidden_states))
    #best_state_by_time_and_prev_state = np.zeros((num_timestamps, num_hidden_states), dtype=int)
    
    most_likely_states = np.zeros((num_timestamps,), dtype=int)

    # alpha(h_1) = p(h_1) * p(v_1 | h_1)
    max_prob_per_time_and_state[-1] = np.ones(num_hidden_states)

    # normalize!
    max_prob_per_time_and_state[-1] /= np.sum(max_prob_per_time_and_state[-1])
    for t in range(num_timestamps - 1, 0, -1):
        # t corresponds to the new_state's timestep.
        
        # take the best prob for this state from this timestep, and elementwise multiply
        # it by the probability that the visible happens
        # max_prob_per_time_and_state[t]: new_state,
        # p_emission[visibles[t]]: new_state,
        # prob_visibles: new_state,
        p_state_visible = max_prob_per_time_and_state[t] * p_emission[visibles[t]]
        
        # now find the probability that it came from each old_state
        # prob_visibles.reshape(-1, 1): new_state x 1
        # np.tile(...): new_state x old_state
        # p_transition: new_state x old_state
        # np.tile(...) * p_transition: new_state x old_state
        p_state_transition = np.tile(
            p_state_visible.reshape(-1, 1),
            (1, num_hidden_states)
        ) * p_transition
        
        # the max_prob_per_time_and_state is the sum over all of the new_states
        max_prob_per_time_and_state[t - 1] = np.max(p_state_transition, axis=0)
        
        # and normalize
        max_prob_per_time_and_state[t - 1] /= np.sum(max_prob_per_time_and_state[t - 1])
    
    # now from the beginning!
    most_likely_states[0] = np.argmax(
        p_hidden_start 
        * p_emission[visibles[0]] 
        * max_prob_per_time_and_state[0]
    )
    for t in range(1, num_timestamps):
        # p_emission[v] = curr_state, 1
        # p_transition[:, most_likely_states[t - 1]] = curr_state
        most_likely_states[t] = np.argmax(
            p_emission[visibles[t], :] 
            * p_transition[:, most_likely_states[t - 1]] 
            * max_prob_per_time_and_state[t]
        )
    
    return most_likely_states

most_likely_states = viterbi(
    prev_post.visibles, 
    prev_post.p_hidden_start,
    prev_post.p_transition,
    prev_post.p_emission,
)

print(most_likely_states)

