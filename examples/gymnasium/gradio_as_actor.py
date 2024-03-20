import asyncio
import datetime

from cogment_lab.envs.gymnasium import GymEnvironment
from cogment_lab.process_manager import Cogment


async def main():
    logpath = f"logs/logs-{datetime.datetime.now().isoformat()}"

    cog = Cogment(log_dir=logpath)

    print(logpath)

    # Launch an environment in a subprocess

    cenv = GymEnvironment(env_id="FrozenLake-v1", render=True, make_kwargs={"is_slippery": False})

    await cog.run_env(env=cenv, env_name="flake", port=9011, log_file="env.log")

    # Launch gradio env

    await cog.run_gradio_ui(log_file="gradio_ui.log")

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
