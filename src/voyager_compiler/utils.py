import copy
import datetime
import json
import logging
import os
import sys
from functools import wraps
from pprint import pformat

import wandb

__all__ = [
    "with_execution_context",
]

logger = logging.getLogger(__name__)

SLURM_LOG_DIR = "slurm_logs"
SLURM_SCRIPT_DIR = "slurm_scripts"
ENV_SETUP_SCRIPT = "setup_shell.sh"

SLURM_ARGS = {
    "job-name": {"type": str, "default": "test"},
    "partition": {"type": str, "default": "gpu"},
    "nodes": {"type": int, "default": 1},
    "time": {"type": str, "default": "48:00:00"},
    "gpus": {"type": str, "default": "1"},
    "cpus": {"type": int, "default": 8},
    "mem": {"type": str, "default": "16GB"},
    "output": {"type": str, "default": None},
    "error": {"type": str, "default": None},
    "exclude": {"type": str, "default": None},
    "nodelist": {"type": str, "default": None},
}

SLURM_NAME_OVERRIDES = {"gpus": "gres", "cpus": "cpus-per-task"}


def write_slurm_script(args, cmd):
    os.makedirs(SLURM_SCRIPT_DIR, exist_ok=True)
    os.makedirs(SLURM_LOG_DIR, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(SLURM_LOG_DIR, args.job_name + ".%j.out")
    if args.error is None:
        args.error = os.path.join(SLURM_LOG_DIR, args.job_name + ".%j.err")
    args.gpus = f"gpu:{args.gpus}" if args.gpus is not None else args.gpus

    with open(os.path.join(SLURM_SCRIPT_DIR, f"{args.job_name}.sbatch"), "w") as f:
        f.write('#!/bin/bash\n')

        for arg_name in SLURM_ARGS.keys():
            arg_value = vars(args)[arg_name.replace("-", "_")]
            if arg_name in SLURM_NAME_OVERRIDES:
                arg_name = SLURM_NAME_OVERRIDES[arg_name]
            if arg_value is not None:
                f.write(f"#SBATCH --{arg_name}={arg_value}\n")

        f.write('\n')
        f.write('echo "SLURM_JOBID = "$SLURM_JOBID\n')
        f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST\n')
        f.write('echo "SLURM_NNODES = "$SLURM_NNODES\n')
        f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR\n')
        f.write('echo "working directory = "$SLURM_SUBMIT_DIR\n')
        f.write('\n')
        f.write('source ' + ENV_SETUP_SCRIPT + '\n')
        f.write('python ' + ' '.join(cmd) + '\n')
        f.write('wait\n')


def write_bash_script(args, cmd):
    with open(f"{args.run_name or 'job'}.sh", "w") as f:
        f.write('#!/bin/bash\n')
        f.write('python ' + ' '.join(cmd) + '\n')


def with_execution_context(fn):
    @wraps(fn)
    def wrapper(args):
        if args.log_file == "datetime":
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            args.log_file = f'logs/{timestamp}.log'

        if args.log_file is not None:
            dirname = os.path.dirname(args.log_file)
            if dirname != "":
                os.makedirs(dirname, exist_ok=True)

        logging.basicConfig(
            filename=args.log_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=getattr(logging, args.log_level),
        )

        # Create W&B sweep from the sweep configuration
        if args.sweep_config_json is not None:
            try:
                sweep_config = json.loads(args.sweep_config_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid sweep JSON: {e}")

            args.sweep_id = wandb.sweep(sweep=sweep_config, project=args.project)

        # Write commands to a script
        if args.execution_mode is not None:
            command = copy.deepcopy(sys.argv)
            if args.sweep_config_json is not None:
                index = command.index('--sweep_config_json')
                command[index:index + 2] = ['--sweep_id', args.sweep_id]

            index = command.index(args.execution_mode)
            if args.execution_mode == "slurm":
                write_slurm_script(args, command[:index])
            elif args.execution_mode == "bash":
                write_bash_script(args, command[:index])
            return

        def agent_run_fn():
            wandb.init()
            run_args = copy.deepcopy(args)
            # W&B does not support specifying the sweep range (min, max) as float types
            # for grid searches. We use int types and multiply the learning rate by the
            # actual value in the sweep function.
            for k, v in wandb.config.items():
                if k == "learning_rate" and isinstance(v, int):
                    v = float(v) * args.learning_rate
                setattr(run_args, k, v)

            logger.info(f"Sweep Args: {pformat(vars(run_args))}")
            fn(run_args)

        if args.sweep_id is not None:
            wandb.agent(
                args.sweep_id,
                function=agent_run_fn,
                count=args.sweep_count,
                project=args.project,
            )
        else:
            if args.project is not None:
                wandb.init(
                    project=args.project,
                    name=args.run_name,
                    id=args.run_id,
                    resume="allow"
                )
            logger.info(f"Run Args: {pformat(vars(args))}")
            fn(args)

    return wrapper
