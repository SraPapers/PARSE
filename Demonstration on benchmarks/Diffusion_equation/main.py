import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import pickle
from fractions import Fraction
from functools import partial
import time
import torch

import Genetic_Operator as gp 
import Dimension_Verification as dv
import LossFunction as lf
import Calculation_Operator as co

s = 0
random.seed(s)
np.random.seed(s)
#%% ------------------------ Input data ------------------------
td  = np.loadtxt('../../Data/Diffusion_equation/Diffusion.dat',  skiprows=1)
# C_t,r,C,C_r,C_rr,C_rrr

C_t   = td[:, 0:1]
r1    = td[:, 1:2]
C     = td[:, 2:3]
C_r   = td[:, 3:4]
C_rr  = td[:, 4:5]
C_rrr = td[:, 5:6]

ga = 1.6667
D = 1.9e-5
Y = C_t
cons = np.ones(Y.shape)
D = np.ones(Y.shape) * D

noise = 0.1  
noise_level = noise * np.std(Y) * np.random.randn(*Y.shape)
Y = Y + noise_level


#------------------------ Dimension System ------------------------
# [M, L, T, K]
num_units = 4
target_dimension = [1, -2, -1, 0]  
 
dict_of_dimension = {
    'r1'    : [0,  1,  0,  0],
    'C'    : [1, -2,  0,  0],
    'C_r'  : [1, -3,  0,  0],
    'C_rr' : [1, -4,  0,  0],
    'C_rrr': [1, -5,  0,  0],
    'D'    : [0,  2, -1,  0],
    'cons' : [0,  0,  0,  0],    
}

Function_names =  ['r1', 'C', 'D']
Gradient_names =  ['C_r','C_rr','C_rrr','cons']

names = Function_names + Gradient_names
data_dict = {}
for name in names:
    data_dict[name] = locals()[name]    

Gradient = [data_dict[n] for n in Gradient_names]
pset = gep.PrimitiveSet('Main', input_names=Function_names)
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(co.protected_div, 2)
# pset.add_rnc_terminal() # if add RNC

#------------------------ Parameter input ------------------------

use_dim_verify = True  
use_Trim = False
n_pop = 500           # Number of individuals in a population
n_generations = 200  # Maximum Generation
tol = 1e-5            # Threshold to terminate the evolution
h = 5                 # head length
n_genes = np.random.randint(len(Gradient_names), 2 * len(Gradient_names))   # number of genes in a chromosome
r = 10           #  length of the RNC array
champs = 3 
n_elites = 1

lamda = 1e-5       # the regularization parameter 
alpha = 0.001      # the penalty weight
ridge_tol = 0.1    # the penalty weight
ridge_iters = 10   # the threshold parameter

isRestart = False
output_type = f'Diff_alpha_noise{noise}'

# Define the indiviudal class, a subclass of gep.Chromosome
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin) 

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10)  
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=co.linker_add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gep.compile_, pset=pset)
toolbox.register(
    'evaluate',
    partial(
        lf.LossFunc,
        data_dict = data_dict, Y = Y, names = names, Gradient_names = Gradient_names,
        use_dim_verify = use_dim_verify, dict_of_dimension = dict_of_dimension, target_dimension = target_dimension, num_units = num_units,
        use_Trim = use_Trim, lamda = lamda,alpha = alpha, ridge_tol = ridge_tol, ridge_iters = ridge_iters,
        device="cuda" ))


toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

# 2. Dc-specific operators
# toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
# toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
# toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
# toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
# toolbox.pbs['mut_rnc_array_dc'] = 1 

# Statistics to be inspected
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("min", np.min)

if isRestart:
    with open(f'pkl/{output_type}.pkl','rb') as file:
        pop  = pickle.loads(file.read())
        
pop = toolbox.population(n=n_pop) 
hall_of_fame = tools.HallOfFame(champs)   
start_time = time.time()
pop, log = gp.gep_simple( data_dict, Y, names, Gradient_names, dict_of_dimension,
                          pop, toolbox, n_generations, n_elites, hall_of_fame, 
                          stats, lamda, alpha, ridge_tol, ridge_iters,
                          verbose = True, tolerance = tol, GEP_type = output_type)