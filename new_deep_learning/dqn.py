import numpy as np
import random
from time import sleep
import tensorflow as tf

height = 4
width = 4

destination = (2, 1)
start_point = (0, 3)

total_state = height * width

matrix_map = []
matrix_state = {}
q_table = np.zeros([total_state, 5])
print("---------")
# left - up - right - down
for y in range(height):
    row = []
    for x in range(width):
        action_m = []
        state_p = y * width + x
        if x == 0:  # left
            action_m.append([1, state_p, -1, False, state_p])
        else:
            action_m.append([1, state_p - 1, -1, False, state_p])

        if y == 0:  # up
            action_m.append([1, state_p, -1, False, state_p])
        else:
            action_m.append([1, state_p - width, -1, False, state_p])

        if x == width - 1:  # right
            action_m.append([1, state_p, -1, False, state_p])
        else:
            action_m.append([1, state_p + 1, -1, False, state_p])

        if y == width - 1:  # down
            action_m.append([1, state_p, -1, False, state_p])
        else:
            action_m.append([1, state_p + width, -1, False, state_p])

        # action_m.append([1, state_p, -10, False, state_p])

        matrix_state[state_p] = action_m
        matrix_map.append(action_m)


def step(state, action_arg):  # return next_state, reward  , done
    current_p = matrix_state[state]
    value = current_p[action_arg]
    if state == 6:
        return value[1], 20, True
    else:
        return value[1], value[2], False


learning_rate = 0.1
discount_factor = 0.6
exploration = 0.1
epochs = 100

from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.optimizers import Adam

init = tf.keras.initializers.Zeros()
model = Sequential()
model.add(Embedding(500,4,input_length=1))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='linear'))
model.summary()
model.compile(loss='mse',
              optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

# pre = model.predict(np.array([9]))
# print(pre)
# pre[0][1] = 20 + 0.65 * np.max(model.predict(np.array([5])))if False else -1 + 0.65 * np.max(model.predict(np.array([5])))
# model.fit(np.array([9]), pre)
#
# pre = model.predict(np.array([5]))
# print(pre)
# pre[0][2] = 20 + 0.65 * np.max(model.predict(np.array([5])))if False else -1 + 0.65 * np.max(model.predict(np.array([6])))
# model.fit(np.array([5]), pre)
# print(model.predict(np.array([5])))

# pre = model.predict(np.array([9]))
# print(pre)
# pre[0][1] = 20 + 0.65 * np.max(model.predict(np.array([5])))if False else -1 + 0.65 * np.max(model.predict(np.array([5])))
# model.fit(np.array([9]), pre)
# print(model.predict(np.array([9])))
# print(pre)
# # reward = -1
# # pre = model.predict(np.array([5]))
# # print(pre)
# # print(pre)
# # action = 2
# # done = True
# # ns, r, d = step(5, action)
# pre[0][0] = 20 if True else -1 + 0.65 * np.max(model.predict(np.array([5])))
# print(pre)
# model.fit(np.array([7]), pre)
# new_pre = model.predict(np.array([7]))
# print(new_pre)
from collections import deque

memory = deque(maxlen=100000)
for pacman_run in range(100):
    if pacman_run == 99:
        print("")
    state_x = random.randint(0, 15)
    done = False
    while not done:
        pre = model.predict(np.array([state_x]))
        random_value = random.random()
        if random_value < exploration:
            action = random.randint(0, 3)
        else:
            action = np.argmax(pre)

        next_state, reward, done = step(state_x, action)
        if next_state == 6:
            reward = 20
            done = True
        # pre[0][action] = reward + discount_factor * np.max(model.predict(np.array([next_state])))
        memory.append((state_x, action, reward, next_state, done))
        # model.fit(np.array([state_x]) , np.array(pre))
        state_x = next_state

    minibatch = random.sample(
        memory, min(len(memory), 15))

    x_batch, y_batch = [], []
    for state, action, reward, next_state, done in minibatch:
        atnpo = model.predict(np.array([14]))
        pred = model.predict(np.array([state]))
        pred_new = model.predict(np.array([next_state]))
        if done:
            pred[0][0][action] = reward
        else:
            pred[0][0][action] = reward + discount_factor * np.max(pred_new)
        model.fit(np.array([state]), np.array(pred))
        # x_batch.append(state)
        # y_batch.append(pred)
    print("----{}----".format(pacman_run))
#
# for tripnum in range(1):
#     state_a = 15
#     done = False
#
#     while not done:
#         pre = model.predict(np.array([state_a]))
#         action = np.argmax(pre)
#         print(state_a)
#         next_state, reward, done = step(state_a, action)
#         sleep(.5)
#         state_a = next_state
#
#     sleep(2)
print(model.predict(np.array([7])))
print(model.predict(np.array([2])))