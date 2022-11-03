import subprocess
import numpy as np

const_params = ['--max_eps', '1000', '--gpu', 'cuda:0', '--hidden_size', '[256,256]']
var_param_name = '--beta'
start = 0.01
stop = 0.03
step = 0.01

for var_param_val in np.arange(start, stop, step):
    for agent in range(1, 4):
        args = ['python3', 'ContinuousV1/DDPG.py']

        for param in const_params:
            args.append(param)

        args.append('--agent')
        args.append(f'{agent}')

        args.append(f'{var_param_name}')
        args.append(f'{var_param_val}')

        subprocess.run(args)