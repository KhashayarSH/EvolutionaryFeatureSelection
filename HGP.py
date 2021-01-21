import random
from utils import SvmAccuracy
# class Particle for PSO part
class Particle:
    # each particle has a featureset, personal best, current cost, personal best cost, velocity
    def __init__(self, feature, dataset):
        self.feature = feature
        self.pbest = feature
        self.current_cost = self.Fitness(dataset)
        self.pbest_cost = self.Fitness(dataset)
        self.velocity = []

    def ClearVelocity(self):
        self.velocity.clear()

    def UpdateFitnessAndPbest(self, dataset):
        self.current_cost = self.Fitness(dataset)
        if self.current_cost < self.pbest_cost:
            self.pbest = self.feature
            self.pbest_cost = self.current_cost
    # cost is calculated by SvmAccuracy from utils
    def Fitness(self, dataset):
        return SvmAccuracy(self.feature, dataset)


    # Particle comparison functions
    def __lt__(self, other):
      return self.current_cost < other.current_cost
    def __le__(self, other):
      return self.current_cost <= other.current_cost
    def __eq__(self, other):
      return self.current_cost == other.current_cost
    def __ne__(self, other):
      return self.current_cost != other.current_cost
    def __gt__(self, other):
      return self.current_cost > other.current_cost
    def __ge__(self, other):
      return self.current_cost >= other.current_cost

# class State for GA
class State:
    # each state includes a featureset and cost
    def __init__(self, feature, dataset):
        self.feature = feature
        self.cost = self.Fitness( dataset)

    # cost is calculated by SvmAccuracy from utils
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



# Class hybrid GA and PSO algorithm feature selection
class HGP_FS:
    # HGP has number of generations, population_size, mutation_rate,ga_elite_size, pso_elite_size
    # gbest_probability : accelerate constant gbest, pbest_probability : accelerate constant pbest,
    # selection algorithm used for breeding pool in GA
     def __init__(self, generations, population_size, mutation_rate, ga_elite_size, pso_elite_size, gbest_probability=1.0, pbest_probability=1.0, selection = 'Roulette'):
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.ga_elite_size = ga_elite_size
        self.pso_elite_size = pso_elite_size
        self.selection = selection
        self.gbest = None
        self.gbest_of_generation = []
        self.current_population = []
        # probability bins for breeding pool selection
        self.pool_probability_bins = []
        self.best_of_generation = []
        self.average_of_generation = []
        self.worst_of_generation = []
        self.best = None
        self.particles = []
        self.gbest_probability = gbest_probability
        self.pbest_probability = pbest_probability


# creates random population for GA with given population size
    def RandomPopulationGA(self, dataset):
        result = []
        for i in range(self.population_size):
            s = []
            zero_flag = True
            # a random number is chosen so there is a higher chance for feature sets with different lengths
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
            if zero_flag:
                s[int(random.uniform(0,len(s)))] = 1
            # adding last column which is labels to feature set
            s.append(1)
            result.append(State(s, dataset))
        return result


# creates random population for GA with given population size
    def RandomPopulationPSO(self, dataset):
        result = []
        for i in range(self.population_size):
            s = []
            zero_flag = True
            # a random number is chosen so there is a higher chance for feature sets with different lengths
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
                s[int(random.uniform(0,len(s)-0.01))] = 1
            # adding last column which is labels to feature set
            s.append(1)
            result.append(Particle(s, dataset))
        return result

    # a function used by selection algorithm to find the index of selected parent
    # it uses a binary search to determine which bin the random number falls in
    # which determines index of selected parent
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


    # generates a breeding pool using rank based selection
    # pool_probability_bins are created to fit rank based selection
    def GenerateBreedingPoolRank(self):
        breeding_pool = []
        # select a breeding pool with size of current_population
        for i in range(len(self.current_population)):
            # a random number is selected in range 0 and max bin
            r = int(random.uniform(0,self.pool_probability_bins[-1]))
            # index of selected random number is found using FindIndex
            index = self.FindIndex(self.pool_probability_bins, r)
            breeding_pool.append(self.current_population[index])
        return breeding_pool


    # generates a breeding pool using Roulette based selection
    # pool_probability_bins are created to fit Roulette based selection
    def GenerateBreedingPoolRoulette(self):
        pool_probability_bins =[0]
        for item in self.current_population:
            pool_probability_bins.append(pool_probability_bins[-1] + item.cost)
        breeding_pool = []
        # select a breeding pool with size of current_population
        for i in range(len(self.current_population)):
            # a random number is selected in range 0 and max bin
            r = random.uniform(0,pool_probability_bins[-1])
            # index of selected random number is found using FindIndex
            index = self.FindIndex(pool_probability_bins, r)
            breeding_pool.append(self.current_population[index])
        return breeding_pool

    # a 2 point cross over function for genetic algorithm
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
        for i in s[:-1]:
            if i == 1:
                zero_flag = False
        if zero_flag:
            r = int(random.uniform(0,len(s)-1.01))
            s[r] = abs(s[r] - 1)
        return State(s, dataset)

    # a mutation function which inverts a feature in random
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

    # function to run HGP_FS
    def run(self, dataset):
        # initiate particles for PSO part
        self.particles = self.RandomPopulationPSO(dataset)
        # initiate population for GA part
        self.current_population = self.RandomPopulationGA(dataset)
        # calculate rank based selection probability pools
        self.pool_probability_bins = [0]
        for i in range(self.population_size):
            self.pool_probability_bins.append(self.pool_probability_bins[i]+self.population_size-i)

        # start g iterations
        for g in range(self.generations):
            # store gbest of generation
            self.gbest = max(self.particles, key=lambda p: p.pbest_cost)
            self.gbest_of_generation.append(self.gbest.pbest_cost)
            # execute PSO part of iteration
            for particle in self.particles:
                particle.ClearVelocity()
                temp_velocity = []
                gbest = self.gbest.pbest[:]
                new_feature = particle.feature[:]
                # computing distance for personal best
                # each different feature adds 1 to distance
                for i in range(len(particle.feature)):
                    if new_feature[i] != particle.pbest[i]:
                        invert = (i, self.pbest_probability)
                        temp_velocity.append(invert)
                # computing distance for global best
                # each different feature adds 1 to distance
                for i in range(len(particle.feature)):
                    if new_feature[i] != gbest[i]:
                        invert = (i, self.gbest_probability)
                        temp_velocity.append(invert)

                particle.velocity = temp_velocity
                # for invert in velocity the feature is inverted with provided
                # probability for gbest and pbest
                for invert in temp_velocity:
                    if random.random() <= invert[1]:
                        new_feature[invert[0]] = abs(new_feature[invert[0]]-1)
                # checking if selected feature set is empty if so invert a feature in random
                zero_flag = True
                for i in new_feature[:-1]:
                    if i == 1:
                        zero_flag = False
                if zero_flag:
                    r = int(random.uniform(0,len(new_feature)-1.01))
                    new_feature[r] = abs(new_feature[r] - 1)

                particle.feature = new_feature
                particle.UpdateFitnessAndPbest(dataset)
            # genetic start
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
            # add GA elites to next generation
            next_population = self.current_population[:self.ga_elite_size]
            # add PSO elites to next generation
            self.particles.sort(reverse = True)
            for i in range(self.pso_elite_size):
                next_population.append(State(self.particles[i].feature,dataset))
            # breed and generate rest of next generation
            for i in range(self.population_size-(self.ga_elite_size + self.pso_elite_size)):
                p1 = int(random.uniform(0,len(breeding_pool)-1))
                p2 = int(random.uniform(0,len(breeding_pool)-1))
                offspring = self.CrossOver(breeding_pool[p1], breeding_pool[p2], dataset)
                # apply mutation at given mutation rate
                r = random.random()
                if r < self.mutation_rate:
                    offspring = self.Mutate(offspring, dataset)
                next_population.append(offspring)
            # replace population with new population
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
