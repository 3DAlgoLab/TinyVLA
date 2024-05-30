# Check env.
import gradio as gr
import numpy as np
from icecream import ic

from language_table.environments import blocks, language_table
from language_table.environments.rewards import block2block

env = None
obs = None
reward = 0
step_id = 0
np.set_printoptions(precision=3)


def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


def init():
    """Initialize the environment"""
    global env, obs, step_id

    if env is not None:
        env.close()
        del env

    if obs is not None:
        del obs

    reward_factory = (
        block2block.BlockToBlockReward
    )  # CHANGE ME: change to another reward.
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=reward_factory,
        seed=0,
    )
    obs = env.reset()
    step_id = 0
    print("Environment initialized.")
    return get_current_observation()


def get_current_observation():
    if obs is not None:
        return obs["rgb"], get_current_status_text()
    return None, None


def get_current_status_text():
    if obs is None:
        return "No environment initialized."
    return (
        f"Step: {step_id:03}, Reward: {reward}, "
        f"ET: {obs['effector_translation']}, "
        f"ETT:{obs['effector_target_translation']}"
    )


def get_current_instruction():
    if obs is not None:
        return decode_inst(obs["instruction"])
    return "No Instruction!"


def step_action(action):
    global obs, reward, step_id
    if env is not None:
        obs, reward, done, info = env.step(action)
        step_id += 1
        return get_current_observation()
    return None


def random_step():
    action = env.action_space.sample()
    return step_action(action)


def step(dx, dy):
    return step_action((dx, dy))


with gr.Blocks(title="Robot Env. Viewer") as demo:
    gr.Markdown("## 🤖 Environment Viewer")
    gr.Textbox(value=get_current_instruction, label="Instruction", interactive=False)
    img = gr.Image(
        label="observation",
        interactive=False,
        value=lambda: get_current_observation()[0],  # type: ignore
    )
    status_text = gr.Textbox(
        value=get_current_status_text, label="Status", interactive=False
    )

    with gr.Row():
        btn_reset = gr.Button("Reset")
        btn_step_random = gr.Button("Random Step")

    current_dx = gr.Slider(minimum=-0.5, maximum=0.5, step=0.01, label="dx(m)", value=0)
    current_dy = gr.Slider(minimum=-0.5, maximum=0.5, step=0.01, label="dy(m)", value=0)

    with gr.Row():
        btn_zero = gr.Button("Zero")
        btn_step = gr.Button("Step")

    btn_reset.click(init, inputs=None, outputs=[img, status_text])
    btn_step_random.click(random_step, inputs=None, outputs=[img, status_text])
    btn_zero.click(lambda: (0, 0), outputs=[current_dx, current_dy])
    btn_step.click(step, inputs=[current_dx, current_dy], outputs=[img, status_text])


if __name__ == "__main__":
    init()
    demo.launch(favicon_path="./robo-favicon.png")
