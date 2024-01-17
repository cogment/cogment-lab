![cog-lab](https://github.com/RedTachyon/cogment_lab/assets/19414946/165557d0-fdf0-4d0a-99f1-3fc321fa194c)

# Human + AI = ❤️


## <a href="https://cogment.ai/cogment_lab"><strong>Docs</strong></a> | <a href="https://ai-r.com/blog"><strong>Blog</strong></a> | <a href="https://discord.gg/kh3t6esJRy"><strong> Discord </strong></a>


[![Package version](https://img.shields.io/pypi/v/cogment-lab?color=%23007ec6&label=pypi%20package)](https://pypi.org/project/cogment-lab)
[![Downloads](https://pepy.tech/badge/cogment-lab)](https://pepy.tech/project/cogment-lab)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/cogment-lab.svg)](https://pypi.org/project/cogment-lab)
[![License - Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-green)](https://github.com/cogment-lab/blob/main/LICENSE)
[![Follow @AI_Redefined](https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow%20@AI_Redefined)](https://twitter.com/AI_Redefined)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

# Introduction

Cogment Lab is a toolkit for doing HILL RL -- that is human-in-the-loop learning, with an emphasis on reinforcement learning.
It is based on [cogment](https://cogment.ai), a low-level framework for exchanging messages between
environments, AI agents and humans.
It's the perfect tool for when you want to interact with your environment yourself, and maybe even trained AI agents.

# Cogment interaction model

While it typically isn't necessary to interact with Cogment directly to use Cogment Lab, it is useful to understand the principles on which it operates.

Cogment exchanges messages between environments and actor. These messages contain the observations, actions, rewards, and anything
else that you might want to keep track of.

Interactions are split into Trials, which correspond to the typical notion of an episode in RL. Each trial has a unique ID, and

## Cogment Lab at a glance

Cogment Lab (as well as Cogment in general) follows a microservice-based architecture.
Each environment, agent, and human interface (collectively: service) is launched as a subprocess, and exchanges messages with the orchestrator,
which in turn ensures synchronization and correct routing of messages.

Generally speaking, you don't need to worry about any of that - Cogment Lab conveniently covers up all the rough edges,
allowing you to do your research without worries.

Cogment Lab is inherently asynchronous - but if you're not familiar with async python, don't worry about it.
The only things you need to remember are:
- Wrap your code in `async def main()`
- Run it with `asyncio.run(main())`
- When calling certain functions use the `await` keyword, e.g. `data = await cog.get_episode_data(...)`

If you are familiar with async programming, there's a lot of interesting things you can do with it - go crazy.


## Terminology

- A `service` is anything that interacts with the Cogment orchestrator. It can be an environment or an actor, including human actors.
- An `actor` in particular is the service that interacts with an environment, and often wraps an `agent`. The internal structure of an actor is entirely up to the user
- An `agent` is what we typically think of as an agent in RL - something that perceives its environment and acts upon it. We do not attempt to solve the agent foundation problem in this documentation.
- An `agent` is simultaneously the part of the environment that's taking an action - multiagent environments may have several agents, so we need to assign an actor to each agent.


## Known rough edges

- When running the web UI, you can open the tab only once per launched process. So if you open the UI, you can run however many trials you want, as long as you don't close it. If you do close it, you should kill the process and start a new one.


## Local installation

- Requires Python 3.10
- Install requirements in a virtual env with something similar to the following

    ```console
    $ python -m venv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements.txt
    $ pip install -e .
    ```
- For the examples you'll need to install the additional `examples_requirements.txt`.


### Apple silicon installation

To run on M1/2/3 macs, you'll need to perform those additional steps

```
pip uninstall grpcio grpcio-tools
export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
pip install grpcio==1.48.2 grpcio-tools==1.48.2 --no-binary :all:
```


## Usage

Run `cogmentlab launch base`.

Then, run whatever scripts or notebooks.

Terminology:
- Model: a relatively raw PyTorch (or other?) model, inheriting from `nn.Module`
- Agent: a model wrapped in some utility class to interact with np arrays
- Actor: a cogment service that may involve models and/or actors
