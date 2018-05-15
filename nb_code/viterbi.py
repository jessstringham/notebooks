'''This code was generated from the notebook 2018-05-13-viterbi-message-passing.ipynb'''

import numpy as np
import matplotlib.pyplot as plt

import nb_code.hmm_alpha_recursion as prev_post

def viterbi(visibles, p_hidden_start, p_transition, p_emission):
    num_timestamps = visibles.shape[0]
    num_hidden_states = p_transition.shape[0]
    
    # messages[t] corresponds to mu(h_t), which is the message coming into h_t
    messages = np.zeros((num_timestamps, num_hidden_states))
    
    most_likely_states = np.zeros((num_timestamps,), dtype=int)

    # The message coming into the last node is 1 for all states
    messages[-1] = np.ones(num_hidden_states)

    # normalize!
    messages[-1] /= np.sum(messages[-1])
    
    # Compute the messages!
    for t in range(num_timestamps - 1, 0, -1):
        # use the data at time t to make mu[h_{t - 1}]
        
        # compute max p(v|h)p(h|h)mu(h)!
        
        # compute p(v|h)mu(h)
        message_and_emission = messages[t] * p_emission[visibles[t]]
        
        # compute p(v|h)p(h|h)mu(h)
        # message_and_emission.reshape(-1, 1): new_state x 1
        # np.tile(...): new_state x old_state
        # p_transition: new_state x old_state
        # np.tile(...) * p_transition: new_state x old_state
        all_h_ts = np.tile(
            message_and_emission.reshape(-1, 1),
            (1, num_hidden_states)
        ) * p_transition
        
        # the message is the value from the highest h_t
        messages[t - 1] = np.max(all_h_ts, axis=0)
        
        # and normalize
        messages[t - 1] /= np.sum(messages[t - 1])
    
    # now from the beginning! compute h_t* using these messages
    
    # argmax will give us the state.
    # argmax p(v_1|h_1)p(h_1)mu(h_1)
    most_likely_states[0] = np.argmax(
        p_hidden_start 
        * p_emission[visibles[0]] 
        * messages[0]
    )
    
    for t in range(1, num_timestamps):
        # argmax_h_t p(v_t|h_t)p(h_t|h_{t - 1})mu(h_t)
        most_likely_states[t] = np.argmax(
            p_emission[visibles[t], :]
            * p_transition[:, most_likely_states[t - 1]] 
            * messages[t]
        )
    
    return most_likely_states

most_likely_states = viterbi(
    prev_post.visibles, 
    prev_post.p_hidden_start,
    prev_post.p_transition,
    prev_post.p_emission,
)

print(most_likely_states)