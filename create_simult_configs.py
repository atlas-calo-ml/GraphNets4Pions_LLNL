import yaml
import random
import copy

base_config = yaml.load(open('configs/simult_crossval.yaml', 'r'))

configs = []

alphas = [2**(-i) for i in range(2, 11)] + [1-2**(-i) for i in range(2, 11)] + [0.5]
alphas.sort()
print(alphas)

for alpha in alphas:
	config = copy.deepcopy(base_config)

	config['training']['alpha'] = alpha

	configs.append(config)

for i, config in enumerate(configs):
	for j in range(4):
		config['data']['fold_ind'] = j
		config['training']['gpu'] = j

		yaml.dump(config, open('configs/simult_config_{}_fold_{}.yaml'.format(i, j), 'w'))
