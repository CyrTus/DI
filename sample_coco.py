import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None, help='Visible GPU ids, e.g. 0,1,2,3')
    parser.add_argument('--nproc_per_node', type=int, default=None, help='Number of distributed worker processes')
    parser.add_argument('--master_port', type=str, default=None, help='Port used by torch.distributed.run')
    parser.add_argument('--no_launch', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


def count_gpus(gpus):
    if gpus is None or gpus.strip() == '':
        return 1
    return len([gpu for gpu in gpus.split(',') if gpu.strip() != ''])


def get_sample_port(sample_args, default_port='1237'):
    for idx, arg in enumerate(sample_args):
        if arg == '--port' and idx + 1 < len(sample_args):
            return sample_args[idx + 1]
        if arg.startswith('--port='):
            return arg.split('=', 1)[1]
    return default_port


def ensure_sample_port(sample_args, port):
    has_port = any(arg == '--port' or arg.startswith('--port=') for arg in sample_args)
    if has_port:
        return sample_args
    return [*sample_args, '--port', str(port)]


def run_sampler(sample_args):
    sampler_path = Path(__file__).with_name('sample_time_emb-traj.py')
    spec = importlib.util.spec_from_file_location('sample_time_emb_traj', sampler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Keep numpy available to the dynamically loaded sampler.
    module.np = np
    module.main(args=sample_args, prog_name='sample_coco.py', standalone_mode=True)


def launch_distributed(wrapper_args, sample_args):
    gpus = wrapper_args.gpus or os.environ.get('CUDA_VISIBLE_DEVICES') or '0'
    nproc = wrapper_args.nproc_per_node or count_gpus(gpus)
    master_port = wrapper_args.master_port or get_sample_port(sample_args)
    sample_args = ensure_sample_port(sample_args, master_port)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus

    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        f'--nproc_per_node={nproc}',
        f'--master_port={master_port}',
        str(Path(__file__).resolve()),
        '--no_launch',
        *sample_args,
    ]
    print('Launching distributed sampling:')
    print('CUDA_VISIBLE_DEVICES=' + gpus, ' '.join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    wrapper_args, sample_args = parse_wrapper_args(sys.argv[1:])
    is_torchrun_worker = 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1
    if wrapper_args.no_launch or is_torchrun_worker:
        run_sampler(sample_args)
    else:
        launch_distributed(wrapper_args, sample_args)


if __name__ == '__main__':
    main()
