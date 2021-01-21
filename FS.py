import pandas as pd
from GA import GA_FS
from PSO import PSO_FS
from HGP import HGP_FS
from utils import PrintResults, ImportantFeatures
# wine,ionosphere,musk1,WBDC,german,lung,Hill-Valley
selected_dataset = 'wine'
# GA,PSO,HGP
selected_alogrithm = 'HGP'
# a pandas dataframe with class attribute at last column
dataset = PrepareDataset(selected_dataset)
# rank features by gain ratio and pick top %split
dataset = dataset[ImportantFeatures(dataset,.85)]
# rename columns
dataset.columns = range(dataset.shape[1])
# create config object for GA_FS
if selected_alogrithm == 'GA':
    # specifying generations, population_size, mutation_rate, elite_size, selecgtion = 'Roulette' | 'Rank'
    ga_fs = GA_FS(generations = 50, population_size = 60, mutation_rate=0.5, elite_size = 2, selection = 'Roulette')
    ga_fs.run(dataset)
    PrintResults(ga_fs)
elif selected_alogrithm == 'PSO':
    # iterations, population_size, gbest_probability=1.0, pbest_probability=1.0
    pso_fs = PSO_FS(iterations=50, population_size=60, gbest_probability=0.05, pbest_probability=0.8)
    pso_fs.run(dataset)
    PrintResults(pso_fs)
elif selected_alogrithm == 'HGP':
    # generations, population_size, mutation_rate, ga_elite_size, pso_elite_size, gbest_probability=1.0, pbest_probability=1.0, selection = 'Roulette'
    # iterations, population_size, gbest_probability=1.0, pbest_probability=1.0
    pso_fs = HGP_FS(generations=50, population_size=60, mutation_rate=0.5, ga_elite_size=2, pso_elite_size=2,\
                    gbest_probability=0.05, pbest_probability=0.8, selection='Roulette')
    pso_fs.run(dataset)
    PrintResults(pso_fs)
