import matplotlib.pyplot as plt

def parse_data(outfile):
    x = []
    y = []
    for line in outfile.split('\n'):
        if line.startswith('Batch'):
            batch = int(line.split(':')[0].split()[1])
            time = float(line.split(':')[-1])
            x.append(batch)
            y.append(time)
    return x, y

def parse_dir(directory, cmfd=False):
    if cmfd:
        with open('{}/cmfd.out'.format(directory), 'r') as f:
            out = f.read()
    else:
        with open('{}/capi.out'.format(directory), 'r') as f:
            out = f.read()
    x, y = parse_data(out)
    return x, y

def make_plot(problem):
    plot_types = ['collision', 'fb', 'nofet', 'analog']
    plot_descrips = ['collision-estimator FET', 'fission bank FET', 'no FET calculation', 'analog-estimator FET']
    plt.figure(figsize=(10,5))
    plt.xlabel('# batches')
    plt.ylabel('Simulation time')
    for i in range(len(plot_types)):
        plot_type = plot_types[i]
        plot_descrip = plot_descrips[i]
        x_data, y_data = parse_dir(problem+'-'+plot_type)
        plt.plot(x_data, y_data, '-o', label='No CMFD, With ' + plot_descrip)
    plt.title(problem + ', no CMFD calculation')
    plt.legend()
    plt.savefig('fig-{}-nocmfd.png'.format(problem))

    plt.figure(figsize=(10,5))
    plt.xlabel('# batches')
    plt.ylabel('Simulation time')
    for i in range(len(plot_types)):
        plot_type = plot_types[i]
        plot_descrip = plot_descrips[i]
        x_data, y_data = parse_dir(problem+'-'+plot_type, cmfd=True)
        plt.plot(x_data, y_data, '-o', label='With CMFD, With ' + plot_descrip)
    plt.title(problem + ', with CMFD calculation')
    plt.legend()
    plt.savefig('fig-{}-cmfd.png'.format(problem))

if __name__ == "__main__":
    problems = ['1d-homog', '2d-beavrs']
    for problem in problems:
        make_plot(problem)
