import yaml
import random
import copy

base_config = yaml.load(open('configs/crossval.yaml', 'r'))

configs = []

for num_blocks in [2, 3, 4]:
    for num_layers in [2, 4, 6]:
        for latent_size in [16, 32, 64]:
            for reducer in ['mean', 'sum']:
                config = copy.deepcopy(base_config)

                # config['model']['block_type'] = block_type
                config['model']['num_blocks'] = num_blocks
                config['model']['num_layers'] = num_layers
                config['model']['latent_size'] = latent_size
                # config['model']['concat_input'] = concat_input
                config['model']['reducer'] = reducer

                configs.append(config)

# random.shuffle(configs)

# configs = configs[:20]

for i, config in enumerate(configs):
	for j in range(4):
		config['data']['fold_ind'] = j
		config['training']['gpu'] = j

		yaml.dump(config, open('configs/class_config_{}_fold_{}.yaml'.format(i, j), 'w'))
