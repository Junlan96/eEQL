from cmath import inf
import geppy as gep
from GEP import selection
from GEP import replication
from GEP import mutation
from GEP import crossover
from GEP import replace
import GEP
from eql import eql
import os
import numpy as np

# parameters
var_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8","x9","x10",
             "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18","x19","x20",
             "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28","x29","x30",
             "x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38","x39","x40"]

func = lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21:  x1



func_name = "waveform-21"


n_gen = 2
n_pop = 10
eq_list = {}



def gep_simple(var_dim, func_name, population,n_generations, n_elites=1):

    parent = population

    for gen in range(n_generations):
        print("======************** ", gen+1, " **************************========")

        for ind in range(n_pop):
            Flag = True
            sy, ADF, Dim = GEP.G_ADF(population[ind])
            print('ADF的值：', sy)


            # **********Adaptively adjust the network structure (One dimensional or two dimensional)**********
            if (Dim == 1):
                GEP.ADF = ADF
            else:
               GEP.ADF2 = ADF


            # **********Control ADF cannot be constant**********
            if (sy.free_symbols == set()):
                population[ind].fitness.values = [float(inf)]
                continue

        # **********Evaluate the population incrementally**********
            #Intra-population same
            for k in range(ind):
                if (gep.simplify(population[k]) == gep.simplify(population[ind])):
                    population[ind].fitness.values = population[k].fitness.values
                    Flag=False
                    break

            # **********Old and new alike**********
            for j in range(n_pop):
                if (gen==0):
                    break
                if (gep.simplify(population[ind]) == gep.simplify(parent[j])):
                    population[ind].fitness.values=parent[j].fitness.values
                    Flag = False
                    break


            # **********evaluate**********
            if(Flag==True):
                eq_list[ind], population[ind].fitness.values = eql(var_dim, func_name, trials=1, Dim=Dim,Var_names=var_names)



            else:
                continue

        # **********save GEP fitness result**********
        parent=population
        min_fit=population[0].fitness.values
        for i in range(n_pop):
            if(population[i].fitness.values < min_fit):
               min_fit = population[i].fitness.values

        func_dir = os.path.join('results/benchmark/test', func_name)
        fi = open(os.path.join(func_dir, 'fit.txt'), 'a')
        fi.write("%d\t%f\t\n\n" % (gen+1,min_fit[0]))
        fi.close()


        if gen == n_generations:
            break

        # **********print fitness value**********
        for ind in range(n_pop):
            print(population[ind].fitness.values)

        elites, offspring = selection(population, n_elites)
        offspring = replication(offspring)
        offspring = mutation(offspring)
        offspring = crossover(offspring)
        population = replace(offspring, elites)

        print("individual middle:")
        for i in range(n_pop):
            ind = gep.simplify(population[i])
            print(ind)

    return  parent


if __name__ == "__main__":

    # **********The number of variables that read the training data from txt**********
    filepath = 'dataset\\{0}_{1}.txt'.format(func_name,'train')
    with open(filepath, "r") as f:
        lines = f.readlines()
        numberOfLines = len(lines)  # read first line
        firstline = (lines[0].strip()).split('\t')
        sample_num = int(firstline[0])
        var_dim = int(firstline[1])


    pop = GEP.init_pop(n_pop)
    pop_final = gep_simple(var_dim, func_name, pop,n_generations=n_gen, n_elites=1)

    print("individual final:")
    for i in range(n_pop):
        print(gep.simplify(pop_final[i]))
        print(pop_final[i].fitness.values)













