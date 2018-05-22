'''This code was generated from the notebook 2018-05-02-hmm-alpha-recursion.ipynb'''

width = 6
height = 5
num_hidden_states = width * height

# prob of starting starting locations
p_hidden_start = np.ones(num_hidden_states) / num_hidden_states

# verify it's a valid probability distribution
assert np.all(np.isclose(np.sum(p_hidden_start), 1))
assert np.all(p_hidden_start >= 0)

def create_transition_joint(width, height):
    num_hidden_states = width * height
    
    # begin by building an unnormalized matrix with 1s for all legal moves.
    unnormalized_transition_joint = np.zeros((num_hidden_states, num_hidden_states))
    
    # This will help me map from height and width to the state
    map_x_y_to_hidden_state_id = np.arange(num_hidden_states).reshape(height, width).T
    
    for x in range(width):
        for y in range(height):
            h_t = map_x_y_to_hidden_state_id[x, y]
    
            # hax to go through each possible direction
            for d in range(4):
                new_x = x
                new_y = y
                if d // 2 == 0:
                    # move left or right!
                    new_x = x + ((d % 2) * 2 - 1)
                else:
                    # move up or down!
                    new_y = y + ((d % 2) * 2 - 1)
    
                # make sure they don't walk through walls
                if any((
                    new_x > width - 1,
                    new_x < 0,
                    new_y > height - 1,
                    new_y < 0
                )):
                    continue
    
                h_t_minus_1 = map_x_y_to_hidden_state_id[new_x, new_y]
                unnormalized_transition_joint[h_t_minus_1][h_t] = 1
    
    # normalize!
    p_transition_joint = unnormalized_transition_joint / np.sum(unnormalized_transition_joint)
    
    # make sure this is a joint probability
    assert np.isclose(np.sum(p_transition_joint), 1)
    # not super necessary, but eh
    assert np.all(p_transition_joint >= 0)

    return p_transition_joint

def create_transition(width, height):
    p_transition_joint = create_transition_joint(width, height)
    
    num_hidden_states = width * height

    p_transition = np.zeros((num_hidden_states, num_hidden_states))
    
    for old_state in range(num_hidden_states):
        p_transition[:, old_state] = p_transition_joint[:, old_state] / np.sum(p_transition_joint[:, old_state])
    
    # verify it's a conditional distribution
    assert np.all(np.sum(p_transition, axis=0)) == 1
    
    return p_transition
    
p_transition = create_transition(width, height)    

def plot_state_in_room(state_id, width=width, height=height):
    h = np.zeros(width * height)
    h[state_id] = 1
    return h.reshape(height, width)

def make_sound_map():
    NUM_SOUNDS = 10
    LOW_PROB = 0.1
    HIGH_PROB = 0.9

    # everything has at least LOW_PROB of triggering the sound
    grid = LOW_PROB * np.ones(num_hidden_states)
    # select NUM_BUMP_CREAKS to make HIGH_PROB
    locs = np.random.choice(
        num_hidden_states, 
        size=NUM_SOUNDS, 
        replace=False
    )
    grid[locs] = HIGH_PROB
    
    return grid

prob_bump_true_given_location = make_sound_map()
prob_creak_true_given_location = make_sound_map()

num_visible_states = 4

def get_emission_matrix(prob_bump_true_given_location, prob_creak_true_given_location):
    # prob_bump_given_state[v][state] = p(v | state)
    p_emission = np.vstack((
        prob_bump_true_given_location * prob_creak_true_given_location,
        prob_bump_true_given_location * (1 - prob_creak_true_given_location),
        (1 - prob_bump_true_given_location) * prob_creak_true_given_location,
        (1 - prob_bump_true_given_location) * (1 - prob_creak_true_given_location),
    ))
    
    assert np.all(np.sum(p_emission, axis=0)) == 1
    
    return p_emission

p_emission = get_emission_matrix(prob_bump_true_given_location, prob_creak_true_given_location)
    
# 1 means True. ex: [1, 0] means bump=True, creak=False
map_visible_state_to_bump_creak = np.vstack((
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],
))

timesteps = 10

hiddens = np.zeros(timesteps, dtype=int)
visibles = np.zeros(timesteps, dtype=int)

hiddens[0] = np.random.choice(num_hidden_states, p=p_hidden_start)
visibles[0] = np.random.choice(
    num_visible_states,
    p=p_emission[:, hiddens[0]]
)

for t in range(1, timesteps):
    hiddens[t] = np.random.choice(
        num_hidden_states,
        p=p_transition[:, hiddens[t - 1]]
    )

    visibles[t] = np.random.choice(
        num_visible_states,
        p=p_emission[:, hiddens[t]]
    )

def alpha_recursion(visibles, p_hidden_start, p_transition, p_emission):
    num_timestamps = visibles.shape[0]
    num_hidden_states = p_transition.shape[0]
    
    # There will be one alpha for each timestamp 
    alphas = np.zeros((num_timestamps, num_hidden_states))

    # alpha(h_1) = p(h_1) * p(v_1 | h_1)
    alphas[0] = p_hidden_start * p_emission[visibles[0]]

    # normalize to avoid overflow
    alphas[0] /= np.sum(alphas[0])
    for t in range(1, num_timestamps):
        # p(v_s | h_s)
        # size: new_states
        corrector = p_emission[visibles[t]]
        
        # sum over all hidden states for the previous timestep and multiply the 
        # transition prob by the previous alpha
        # transition_matrix size: new_state x old_state
        # alphas[t_minus_1].T size: old_state x 1
        # predictor size: new_state x 1,
        predictor = p_transition @ alphas[t - 1, None].T

        # alpha(h_s)
        alphas[t, :] = corrector * predictor[:, 0]

        # normalize
        alphas[t] /= np.sum(alphas[t])

    return alphas

alphas = alpha_recursion(
    visibles, 
    p_hidden_start,
    p_transition,
    p_emission,
)

assert np.all(np.isclose(np.sum(alphas, axis=1), 1))