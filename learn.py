from util import *
from agent import *
from hyperparam import *
from fill_memory import fill_memory
import numpy as np
import tensorflow as tf

DQNetwork = DQNetwork(STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
memory = Memory(MEMORY_SIZE)
fill_memory(memory)


def choose_action(explore_max, explore_min, decay_rate, decay_step, state, actions, sess):
    exploit = np.random.rand()

    explore = explore_min + (explore_max - explore_min) * np.exp(-decay_rate * decay_step)

    if explore > exploit:
        action = random.choice(actions)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape(1, *state.shape)})

        choice = np.argmax(Qs)
        action = actions[int(choice)]

    return action, explore

def main():

    stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)
    saver = tf.train.Saver()
    game, actions = create_environment()

    if TRAINING:
        init_op = tf.initialize_all_variables() 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            game.init()

            for episode in range(TOTAL_EPISODES):
                step = 0

                episode_rewards = []

                game.new_episode()

                state = game.get_state().screen_buffer

                # stack frame function also preprocess our states (image)
                state, stacked_frames = stack_frame(stacked_frames, state, True)

                for step in range(MAX_STEPS):
                    decay_step = step

                    action, explore_probability = choose_action(EXPLORE_MIN, EPSILON_MAX, DECAY_RATE, decay_step, state, actions, sess) 

                    reward = game.make_action(action)

                    done = game.is_episode_finished()

                    episode_rewards.append(reward)

                    if done:
                        
                        next_state = np.zeros((84, 84), dtype=np.int)

                        next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)

                        total_reward = np.sum(episode_rewards)
                        print(
                                'Episode: {}\n'.format(episode),
                                'Total reward: {}\n'.format(total_reward),
                                'Training loss: {:.4f}'.format(loss)
                                )
                        
                        memory.add((state, action, reward, next_state, done))

                        break
                    else:
                        next_state = game.get_state().screen_buffer

                        next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)
                        memory.add((state, action, reward, next_state, done))

                        state = next_state

                    ## LEARNING PART

                    batch = memory.sample(BATCH_SIZE)
                    states_batch = np.array([each[0] for each in batch], ndmin=3)
                    actions_batch = np.array([each[1] for each in batch])
                    rewards_batch = np.array([each[2] for each in batch])
                    next_states_batch = np.array([each[3] for each in batch], ndmin=3)
                    dones_batch = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_batch})

                    for i in range(len(batch)):
                        terminal = dones_batch[i]

                        if terminal:
                            target_Qs_batch.append(rewards_batch[i])
                        else:
                            target = rewards_batch[i] + DISCOUNT * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)
                    
                    # REMOVE AFTER DEBUGGING
                    import pdb; pdb.set_trace()    

                    loss, _ = sess.run(
                            [DQNetwork.loss, DQNetwork.optimizer],
                            feed_dict={
                                DQNetwork.inputs_: states_batch,
                                DQNetwork.target_Q: target_Qs_batch,
                                DQNetwork.actions_: actions_batch}
                            )
                    
                    summary = sess.run(write_op,
                        feed_dict={
                            DQNetwork.inputs_: states_batch,
                            DQNetwork.target_Q: target_Qs_batch,
                            DQNetwork.actions__: actions_batch
                        })

                    writer.add_summary(summary, episode)
                    writer.flush()

                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Savved")

if __name__ == "__main__":
    main()
