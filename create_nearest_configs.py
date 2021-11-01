import yaml
import random
import copy

base_config = yaml.load(open('configs/nearest_crossval.yaml', 'r'))

configs = []
num_blocks = 4
num_layers = 4
latent_size = 64

alphas = [2**(-i) for i in range(2, 11, 4)] + [1-2**(-i) for i in range(2, 11, 4)] + [0.5]
alphas.sort()
print(alphas)

for alpha in alphas:
    for k in [4, 6, 8, 10]:
        for use_xyz in [True, False]:
            for reducer in ['mean', 'sum']:
                config = copy.deepcopy(base_config)

                config['training']['alpha'] = alpha
                config['training']['num_blocks'] = num_blocks
                config['training']['num_layers'] = num_layers

                config['model']['latent_size'] = latent_size
                config['model']['reducer'] = reducer

                config['data']['k'] = k
                config['data']['use_xyz'] = use_xyz

                configs.append(config)

# random.shuffle(configs)

# configs = configs[:20]

for i, config in enumerate(configs):
	for j in range(4):
		config['data']['fold_ind'] = j
		config['training']['gpu'] = j

		yaml.dump(config, open('configs/nearest_configs/nearest_config_{}_fold_{}.yaml'.format(i, j), 'w'))
