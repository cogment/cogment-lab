import asyncio
import datetime
import logging
import multiprocessing as mp

import gradio as gr
import numpy as np
from transformers import pipeline

from cogment_lab.envs.gymnasium import GymEnvironment
from cogment_lab.process_manager import Cogment
from cogment_lab.utils.runners import setup_logging


def run_gradio_interface(send_queue: mp.Queue, recv_queue: mp.Queue, log_file: str | None = None):
    if log_file is not None:
        setup_logging(log_file)

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

    def transcribe(audio):
        if audio is None:
            return None
        sr, y = audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        return transcriber({"sampling_rate": sr, "raw": y})["text"]

    def get_starting_action():
        logging.info("Getting starting action inside gradio")
        obs, frame = recv_queue.get()
        logging.info(f"Received obs {obs} and frame inside gradio")
        # start_button.visible = False
        # text_input.visible = True
        return obs, frame

    def send_action(action: str):
        logging.info(f"Sending action {action} inside gradio")
        action = int(action)  # TODO: Handle non-integer actions
        send_queue.put(action)
        logging.info(f"Sent action {action} inside gradio")
        obs, frame = recv_queue.get()
        logging.info(f"Received obs {obs} and frame inside gradio")
        return obs, frame

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    image_output = gr.Image(label="Image Output")
            with gr.Column(scale=1):
                text_output = gr.Textbox(label="Text Output")
                with gr.Row():
                    text_input = gr.Textbox(label="Text Input")
                    start_button = gr.Button("Start")
                    audio_input = gr.Audio(label="Audio Input")

        start_button.click(fn=get_starting_action, outputs=[text_output, image_output])
        text_input.submit(fn=send_action, inputs=text_input, outputs=[text_output, image_output])
        audio_input.change(fn=transcribe, inputs=audio_input, outputs=text_input)

    demo.launch()


async def main():
    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    print(logpath)

    # Launch an environment in a subprocess

    cenv = GymEnvironment(env_id="FrozenLake-v1", render=True, make_kwargs={"is_slippery": False})

    await cog.run_env(env=cenv, env_name="flake", port=9011, log_file="env.log")

    # Launch gradio env

    await cog.run_gradio_ui(gradio_app_fn=run_gradio_interface, log_file="gradio_ui.log")

    trial_id = await cog.start_trial(
        env_name="flake",
        session_config={"render": True},
        actor_impls={
            "gym": "gradio",
        },
    )

    data = await cog.get_trial_data(trial_id)

    print(data)


if __name__ == "__main__":
    asyncio.run(main())
