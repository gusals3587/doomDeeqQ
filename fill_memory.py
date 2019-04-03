from hyperparam import *
from util import *

def fill_memory(memory):

    stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)
    game.new_episode()

    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frame(stacked_frames, state, True)
    for i in range(PRETRAIN_LENGTH):
        action = random.choice(actions)

        reward = game.make_action(action)

        done = game.is_episode_finished()

        if done:

            # next state is just nothing since we're finished
            next_state = np.zeros(state.shape)

            memory.add((state, action, reward, next_state, done))
            
            game.new_episode()

        # we may be adding a bit more to memory in the
        # edge case of an end of an episode, but frankly, I don't care
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)
        memory.add((state, actions, reward, next_state, done))

        state = next_state

