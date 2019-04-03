import tensorflow as tf
import numpy as np
from vizdoom import * 

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def create_environment():
    game = DoomGame()
        
    # load configuration files
    game.load_config("basic.cfg")

    game.set_doom_scenario_path("basic.wad")
    game.init()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions

def test_environment():
    game, actions = create_environment()
    episodes = 10
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        action = random.choice(actions)
        reward = game.make_action(action)
        time.sleep(0.02)
    print("Result:", game.get_total_reward())
    time.sleep(2)
    game.close()

# grayscaling was done by the configuration file
# here, we crop the image, 'normalize' the image than resize it
# to be smaller
def preprocess_frame(frame):
    cropped_frame = frame[30:-10, 30:-30]
    print(cropped_frame)

    normalized_frame = cropped_frame/255.0

    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame

def test_process_frame():
    game, _ = create_environment()
    game.new_episode()
    state = game.get_state()
    img = state.screen_buffer
    time.sleep(2)
    game.close()

STACK_SIZE = 4

# frames is the deque, state is the stacked matrix of the frames (images)
def stack_frame(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:

        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

    else:

        stacked_frames.append(frame)
 
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

if __name__ == "__main__":
    test = input("1: random agent test, 2:test frame processing\n")

    if test == "1":
        test_environment()
    elif test == "2":
        test_process_frame()
