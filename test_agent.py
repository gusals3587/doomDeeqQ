import numpy as np
import tensorflow as tf
from agent import *
from hyperparam import *
from util import *

DQNetwork = DQNetwork(STATE_SIZE, ACTION_SIZE, LEARNING_RATE)

def main():
    with tf.Session() as sess:
        totalScore = 0
        saver = tf.train.Saver()
        saver.restore(sess, "./models/model.ckpt")
        game.init()

        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)
        for i in range(100):
            game.new_episode()
            score = 0
            while not game.is_episode_finished():
                frame = game.get_state().screen_buffer
                state, _ = stack_frame(stacked_frames, frame, False)
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape(1, *state.shape)})
                action = np.argmax(Qs)
                action = actions[int(action)]
                game.make_action(action)
                score = game.get_total_reward()
            print("Score: ", score)
            totalScore += score
        print("TOTAL SCORE:", totalScore/100)
        game.close()

    

if __name__ == "__main__":
    main()

