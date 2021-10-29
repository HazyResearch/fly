import math
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

max_bound = 1.0
nsteps = 1000
dots = torch.linspace(-max_bound, max_bound, nsteps)
d = 16
m = int(d * math.log(d))  # 44
seqlen = 1024
n_hashes = 4  # This is L in the LSH notation
nbuckets = 23  # This is k in the LSH notation
# We set these so that bucket_size = 1024 / (2 * 23) = 22, so Reformer has 22 * 4 = 88 parameters
# Which is the same as Performer (2 * 44)

sm = torch.exp(dots)
performer_mse = 1 / m * torch.exp(2 * max_bound**2 + 2 * dots) * sm**2 * (1 - torch.exp(-2 * max_bound**2 - 2 * dots))
performer_mse_half = 1 / (m / 2) * torch.exp(2 * max_bound**2 + 2 * dots) * sm**2 * (1 - torch.exp(-2 * max_bound**2 - 2 * dots))
performer_hyp_mse = 1 / 2 * (1 - math.exp(-max_bound**2)) * performer_mse_half
tau = torch.sqrt(2 * max_bound**2 - 2 * dots)
lsh_prob = torch.exp(-tau**2 / (4 - tau**2) * math.log(d))
lsh_prob_power = 1 - (1 - lsh_prob**nbuckets)**n_hashes
reformer_mse = sm**2 * (1 - lsh_prob_power)
lsh_prob_power_scatterbrain = 1 - (1 - lsh_prob**(nbuckets))**(n_hashes)
scatterbrain_mse = (1 - lsh_prob_power_scatterbrain) * performer_mse
scatterbrain_hyp_mse = (1 - lsh_prob_power_scatterbrain) * performer_hyp_mse

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use(['seaborn-colorblind'])
# import seaborn as sns
# sns.set_theme()
# sns.set_style('whitegrid')

plt.figure(figsize=(6, 4))
plt.plot(dots, reformer_mse, alpha=0.7, color='blue', label='Reformer')
plt.plot(dots, performer_mse, alpha=0.7, color='green', label='Performer')
plt.plot(dots, scatterbrain_mse, alpha=0.7, color='red', label='Scatterbrain')
plt.plot(dots, performer_hyp_mse, alpha=0.7, color='black', label='Performer hyp')
plt.plot(dots, scatterbrain_hyp_mse, alpha=0.7, color='brown', label='Scatterbrain hyp')
plt.xlabel(r'$q^\top k$', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
plt.legend(fontsize=14)
plt.savefig('theory_mse_new.pdf', bbox_inches='tight')
plt.close()
