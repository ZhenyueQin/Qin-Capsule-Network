from deap import base, creator
import random
from QinCapsNet import QinCapsNet
import os
import glob

from deap import tools

import numpy as np


caps_net_performance_dict = {}
prev_inds = []
prev_inds_fit = []


def fill_caps_net_performance_dict():
    files = glob.glob('generated_files/' + '*.txt')
    files.sort(key=os.path.getmtime)

    for a_file in files:
        a_tmp_file_0 = a_file.replace('generated_files/', '')
        if a_tmp_file_0.endswith(".txt") and a_tmp_file_0[0] != 'g':
            tmp_filename = a_tmp_file_0.replace('.txt', '')
            a_f = open('generated_files' + os.sep + a_tmp_file_0, 'r')
            split = tmp_filename.split('_')
            caps_net_performance_dict[(int(split[0]), int(split[1]), int(split[2]))] = float(a_f.read())
            prev_inds.append([int(split[0]), int(split[1]), int(split[2])])
            prev_inds_fit.append(caps_net_performance_dict[(int(split[0]), int(split[1]), int(split[2]))])
        else:
            continue


fill_caps_net_performance_dict()


print(caps_net_performance_dict)
print(prev_inds)
print(prev_inds_fit)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)

stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

predefined_ind = 0


def initialize_attribute():
    global predefined_ind
    if predefined_ind < len(prev_inds_fit):
        tmp_predefined_ind = predefined_ind
        predefined_ind = predefined_ind + 1
        return prev_inds[tmp_predefined_ind]
    a_caps1_n_maps = random.randint(20, 39)
    a_caps1_n_dims = random.randint(5, 24)
    a_caps2_n_dims = random.randint(10, 29)
    return [a_caps1_n_maps, a_caps1_n_dims, a_caps2_n_dims]


def mutate_a_net(individual):
    a_rand_idx = random.randint(0, 2)
    if a_rand_idx == 0:
        individual[0][a_rand_idx] = random.randint(20, 39)
    elif a_rand_idx == 1:
        individual[0][a_rand_idx] = random.randint(5, 24)
    elif a_rand_idx == 2:
        individual[0][a_rand_idx] = random.randint(10, 29)


def random_mix(ind1, ind2):
    a_rand_idx = random.randint(0, 2)
    a_tmp_cp = ind1[0][a_rand_idx]
    ind1[0][a_rand_idx] = ind2[0][a_rand_idx]
    ind2[0][a_rand_idx] = a_tmp_cp


def evaluate(individual):
    a_capsnet = QinCapsNet(individual[0][0], individual[0][1], 10, individual[0][2], 8, 100, False)
    if (individual[0][0], individual[0][1], individual[0][2]) in caps_net_performance_dict:
        return caps_net_performance_dict[(individual[0][0], individual[0][1], individual[0][2])],
    else:
        caps_net_performance_dict[(individual[0][0], individual[0][1], individual[0][2])] = a_capsnet.create_a_net()
        return caps_net_performance_dict[(individual[0][0], individual[0][1], individual[0][2])],


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

from deap import tools

toolbox = base.Toolbox()
toolbox.register("attribute", initialize_attribute)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", random_mix)
toolbox.register("mutate", mutate_a_net)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=30)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 50

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    logbook = tools.Logbook()
    for g in range(NGEN):
        print("generation: ", g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        print("Crossovering")
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        print("Mutating")
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        record = stats.compile(pop)
        print(record)

        tmp_file_name_1 = './generated_files/' + 'generation_' + str(g) + '.txt'
        with open(tmp_file_name_1, 'a') as out:
            out.write(str(record))

        logbook.record(gen=g, **record)

        tmp_file_name_2 = './generated_files/' + 'generation_' + str(g) + '.pk'
        import pickle
        with open(tmp_file_name_2, 'wb') as handle:
            pickle.dump(logbook, handle)
    return pop


main()
