import asyncio
import logging
import multiprocessing as mp
import signal

import gradio as gr

from cogment_lab.humans.gradio_actor import run_cogment_actor
from cogment_lab.utils.runners import setup_logging


async def shutdown():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()


def signal_handler(sig, frame):
    asyncio.create_task(shutdown())


def run_gradio_interface(send_queue: mp.Queue, recv_queue: mp.Queue):
    # transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

    # def transcribe(audio):
    #     sr, y = audio
    #     y = y.astype(np.float32)
    #     y /= np.max(np.abs(y))
    #
    #     return transcriber({"sampling_rate": sr, "raw": y})["text"]

    def get_starting_action():
        logging.info("Getting starting action inside gradio")
        obs, frame = recv_queue.get()
        logging.info(f"Received obs {obs} and frame inside gradio")
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

        # start_button.click(fn=generate_random_image, outputs=image_output)
        start_button.click(fn=get_starting_action, outputs=[text_output, image_output])
        text_input.submit(fn=send_action, inputs=text_input, outputs=[text_output, image_output])

    demo.launch()


async def gradio_actor_main(
    cogment_port: int,
    signal_queue: mp.Queue,
):
    gradio_to_actor = mp.Queue()
    actor_to_gradio = mp.Queue()

    logging.info("Starting gradio interface")
    process = mp.Process(target=run_gradio_interface, args=(gradio_to_actor, actor_to_gradio))
    process.start()

    logging.info("Starting cogment actor")
    cogment_task = asyncio.create_task(
        run_cogment_actor(
            port=cogment_port,
            send_queue=actor_to_gradio,
            recv_queue=gradio_to_actor,
            signal_queue=signal_queue,
        )
    )

    logging.info("Waiting for cogment actor to finish")

    await cogment_task

    logging.error("Cogment actor finished, runner exiting")


def gradio_actor_runner(
    cogment_port: int,
    signal_queue: mp.Queue,
    log_file: str | None = None,
):
    if log_file:
        setup_logging(log_file)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda s=sig, frame=None: signal_handler(s, frame))

    try:
        loop.run_until_complete(
            gradio_actor_main(
                cogment_port=cogment_port,
                signal_queue=signal_queue,
            )
        )
    finally:
        loop.close()
