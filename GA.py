import random
from utils import SvmAccuracy

class State:
    def __init__(self, feature, dataset):
        self.feature = feature
        self.cost = self.Fitness(dataset)

    def Fitness(self, dataset):
        return SvmAccuracy(self.feature, dataset)

    # state comparison functions
    def __lt__(self, other):
      return self.cost < other.cost
    def __le__(self, other):
      return self.cost <= other.cost
    def __eq__(self, other):
      return self.cost == other.cost
    def __ne__(self, other):
      return self.cost != other.cost
    def __gt__(self, other):
      return self.cost > other.cost
    def __ge__(self, other):
      return self.cost >= other.cost



# Class genetic algorithm feature selection
class GA_FS:
    def __init__(self, generations, population_size, mutation_rate, elite_size, selection = 'Roulette'):
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.selection = selection
        self.current_population = []
        self.pool_probability_bins = []
        self.best_of_generation = []
        self.average_of_generation = []
        self.worst_of_generation = []
        self.best = None


    def RandomPopulation(self, dataset):
        result = []
        for i in range(self.population_size):
            s = []
            zero_flag = True
            p = random.random()
            # randomly choosing features to be in feature set
            # last column is considered to be labels which is added later
            for l in range(dataset.shape[1]-1):
                r = random.random()
                if r < p:
                    s.append(0)
                else:
                    s.append(1)
                    zero_flag = False
            # checking if selected feature set is empty if so invert a feature in random
            if zero_flag:
                s[int(random.uniform(0,len(s)))] = 1
            # adding last column which is labels to feature set
            s.append(1)
            result.append(State(s, dataset))
        return result


    def FindIndex(self, pool_probability_bins, index):
        l = 0
        r = len(self.pool_probability_bins)-1
        while l < r:
            m = int((l+r)/2)
            if self.pool_probability_bins[m] == input:
                break;
            elif index > self.pool_probability_bins[m]:
                if index < self.pool_probability_bins[m+1]:
                    break
                else:
                    l = m+1
            else:
                if index > self.pool_probability_bins[m-1]:
                    m -=1
                    break
                else:
                    r = m-1
        return m


    def GenerateBreedingPoolRank(self):
        breeding_pool = []
        for i in range(len(self.current_population)):
            r = int(random.uniform(0,self.pool_probability_bins[-1]))
            index = self.FindIndex(self.pool_probability_bins, r)
            breeding_pool.append(self.current_population[index])
        return breeding_pool


    def GenerateBreedingPoolRoulette(self):
        pool_probability_bins =[0]
        for item in self.current_population:
            pool_probability_bins.append(pool_probability_bins[-1] + item.cost)
        breeding_pool = []
        for i in range(len(self.current_population)):
            r = random.uniform(0,pool_probability_bins[-1])
            index = self.FindIndex(pool_probability_bins, r)
            breeding_pool.append(self.current_population[index])
        return breeding_pool


    def CrossOver(self, p1, p2, dataset):
        a = int(random.uniform(0,len(p1.feature)-.01))
        b = int(random.uniform(0,len(p1.feature)-.01))
        s = []
        for item in range(len(p1.feature)):
            if item >= min(a,b) or item <= max(a,b):
                s.append(p1.feature[item])
            else:
                s.append(p2.feature[item])
        # checking if selected feature set is empty if so invert a feature in random
        zero_flag = True
        for item in s[:-1]:
            if item == 1:
                zero_flag = False
        if zero_flag:
            r = int(random.uniform(0,len(s)-1.01))
            s[r] = abs(s[r] - 1)
        return State(s, dataset)

    def Mutate(self, state, dataset):
        r = int(random.uniform(0,len(state.feature)-1.01))
        s = state.feature
        s[r] = abs(s[r] - 1)
        # checking if selected feature set is empty if so invert a feature in random
        zero_flag = True
        for item in s[:-1]:
            if item == 1:
                zero_flag = False
        if zero_flag:
            r = int(random.uniform(0,len(s)-1.01))
            s[r] = abs(s[r] - 1)
        return State(s,dataset)

    # function to run GA_FS
    def run(self, dataset):
        self.current_population = self.RandomPopulation(dataset)
        self.pool_probability_bins = [0]
        for i in range(self.population_size):
            self.pool_probability_bins.append(self.pool_probability_bins[i]+self.population_size-i)
        for g in range(self.generations):
            self.current_population.sort(reverse=True)
            # store best and worst of generation
            self.best_of_generation.append(self.current_population[0].cost)
            self.worst_of_generation.append(self.current_population[-1].cost)
            # compute average of generation and store it
            avg = 0
            for item in self.current_population:
                avg += item.cost
            self.average_of_generation.append(avg/len(self.current_population))
            # generate breeding pool
            if self.selection == 'Rank':
                breeding_pool = self.GenerateBreedingPoolRank(self.pool_probability_bins)
            else:
                breeding_pool = self.GenerateBreedingPoolRoulette()
            # add elites to next generation's population
            next_population = self.current_population[:self.elite_size]
            for i in range(self.population_size-self.elite_size):
                p1 = int(random.uniform(0,len(breeding_pool)-1))
                p2 = int(random.uniform(0,len(breeding_pool)-1))
                offspring = self.CrossOver(breeding_pool[p1], breeding_pool[p2], dataset)
                r = random.random()
                if r < self.mutation_rate:
                    offspring = self.Mutate(offspring, dataset)
                next_population.append(offspring)
            self.current_population = next_population

        self.current_population.sort(reverse=True)
        self.best = self.current_population[0]
        # store best and worst of generation
        self.best_of_generation.append(self.current_population[0].cost)
        self.worst_of_generation.append(self.current_population[-1].cost)
        # compute average of generation and store it
        avg = 0
        for item in self.current_population:
            avg += item.cost
        self.average_of_generation.append(avg/len(self.current_population))
