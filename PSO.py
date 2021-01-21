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


# Class PSO algorithm feature selection
class PSO_FS:

    def __init__(self, iterations, population_size, gbest_probability=1.0, pbest_probability=1.0):
        self.gbest = None
        self.gbest_of_generation = []
        self.best_of_generation = []
        self.worst_of_generation = []
        self.average_of_generation = []
        self.iterations = iterations
        self.population_size = population_size
        self.particles = []
        self.gbest_probability = gbest_probability
        self.pbest_probability = pbest_probability


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
            result.append(Particle(s, dataset))
        return result

    # function to run PSO_FS
    def run(self, dataset):
        # initiating particles with random population
        self.particles = self.RandomPopulation(dataset)
        # run for specified iteration
        for t in range(self.iterations):
            # store gbest,best,average and worst of each iteration
            self.gbest = max(self.particles, key=lambda p: p.pbest_cost)
            self.gbest_of_generation.append(self.gbest.pbest_cost)
            best = max(self.particles, key=lambda p: p.current_cost)
            self.best_of_generation.append(best.current_cost)
            worst = min(self.particles, key=lambda p: p.current_cost)
            self.worst_of_generation.append(worst.current_cost)
            average = 0
            for particle in self.particles:
                average += particle.current_cost
            average /= len(self.particles)
            self.average_of_generation.append(average)

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
