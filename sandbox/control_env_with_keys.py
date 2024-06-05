import gym
from pynput import keyboard

# Create the environment
env = gym.make("MountainCar-v0")

# Reset the environment
state = env.reset()

# Define the action mapping
action_mapping = {
    "a": 0,  # push left
    "s": 1,  # no push
    "d": 2,  # push right
}

# Define the action variable
action = action_mapping["s"]


# Define the key press and release functions
def on_press(key):
    global action
    try:
        key_char = key.char
        if key_char in action_mapping:
            action = action_mapping[key_char]
    except AttributeError:
        pass


def on_release(key):
    global action
    action = action_mapping["s"]


# Start the keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

done = False
while True:
    env.render()
    state, reward, done, info = env.step(action)
    if done:
        print("Episode finished")
        break

env.close()
