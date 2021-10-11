import numpy as np

seeds = range(1, 21)

for seed in seeds:
    seed_data = []
    with open('logs/seed{}process0.log'.format(seed), 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        time = line.split()[-1].split('=')[-1]
        seed_data.append(float(time))
    print(seed, np.average([seed_data[i] - seed_data[i-1] for i in range(1, len(seed_data))]))
