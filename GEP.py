import deap
import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
from sympy import lambdify,symbols


random.seed(5)

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

import operator


def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def exp(x):
    return np.exp(x)

def log(x):
    return np.log(x)


pset = gep.PrimitiveSet('Main', input_names=['x','y'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)



pset.add_ephemeral_terminal(name='enc', gen=lambda: random.randint(-5, 5)) # each ENC is a random integer within [-10, 10]


from deap import creator, base, tools

creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)


h=6
n_genes = 2

toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register('compile', gep.compile_, pset=pset)

toolbox.register('select', tools.selTournament, tournsize=3)

toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')
toolbox.pbs['mut_ephemeral'] = 1


stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)



def init_pop(n_pop):

    pop = toolbox.population(n=n_pop)
    print("individual start:")
    for i in range(n_pop):
        ind=gep.simplify(pop[i])
        print(ind)
    return pop


import ast

def array_diff1(a, b):
    return [x for x in a if x not in b]

array=['sin','cos','log']

def G_ADF(pop):
    sy = gep.simplify(pop)
    sym = str(sy)
    names = [i.id for i in ast.walk(ast.parse(sym)) if isinstance(i, ast.Name)]
    var_names = list(set(names))
    var_names = array_diff1(var_names, array)
    var = symbols(var_names)
    exp_ADF = lambdify(var, sy, 'numpy')

    return sy,exp_ADF,len(var)


global ADF
global ADF2


def _apply_modification(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
    return population


def _apply_crossover(population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population

# selection with elitism
def selection(population, n_elites):
    elites = deap.tools.selBest(population, k=n_elites)
    offspring = toolbox.select(population, len(population) - n_elites)
    return elites, offspring


# replication
def replication(offspring):
        offspring = [toolbox.clone(ind) for ind in offspring]
        return offspring


def mutation(offspring):
    # mutation
    for op in toolbox.pbs:
        if op.startswith('mut'):
            offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
    return offspring


def crossover(offspring):
    # crossover
    for op in toolbox.pbs:
        if op.startswith('cx'):
            offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])
    return offspring


def replace(offspring, elites):
    # replace the current population with the offsprings
    population = elites + offspring
    return population




