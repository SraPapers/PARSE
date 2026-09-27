import deap
import random
import warnings
import numpy as np
import pickle
import datetime
import geppy as gep
import time
import os
import sympy as sp
from collections import defaultdict
import LossFunction as lf


def _validate_basic_toolbox(tb):
    """
    Validate the operators in the toolbox *tb* according to our conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a 'select' operator."
    # whether the ops in .pbs are all registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Operators must start with 'mut' or 'cx' except selection."
        assert hasattr(tb, op), "Probability for a operator called '{}' is specified, but this operator is not " \
                                "registered in the toolbox.".format(op)
    # whether all the mut_ and cx_ operators have their probabilities assigned in .pbs
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('{0} is registered, but its probability is NOT assigned in Toolbox.pbs. '
                          'By default, the probability is ZERO and the operator {0} will NOT be applied.'.format(op),
                          category=UserWarning)

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

# When outputting, simplify the expressions
def my_simplify(individual, names, Gradient_names, coef_tol):
    sym_dict = {name: sp.Symbol(name) for name in names}
    def protected_div_sym(a, b):
        return a / b
    ns = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "protected_div": protected_div_sym,
    }
    ns.update(sym_dict) 
    def trim_small_terms(expr, coef_tol):
        expr = sp.expand(expr)
        terms = sp.Add.make_args(expr)

        kept_terms = []
        for term in terms:
            coeff, rest = term.as_coeff_Mul()

            try:
                coeff_value = float(coeff)
            except TypeError:
                kept_terms.append(term)
                continue

            if abs(coeff_value) >= coef_tol:
                kept_terms.append(term)

        if not kept_terms:
            return sp.Integer(0)

        return sp.simplify(sum(kept_terms))  
    base_expr = defaultdict(lambda: 0)
    for k, (gene_idx, base_idx) in enumerate(individual.local_ij):
        c = float(individual.coef[k])
        c = round(float(c), 3)
        if abs(c) < coef_tol:
            continue  
            
        gene_str = str(gep.simplify(individual[gene_idx])).replace("\n", "").strip()
        g_sym = eval(gene_str, {}, ns)

        base = Gradient_names[base_idx]
        base_expr[base] += sp.Float(c) * g_sym   
    parts = []
    for base in Gradient_names:
        if base in base_expr:
            coef_simpl = sp.simplify(base_expr[base])
            coef_simpl = trim_small_terms(coef_simpl, coef_tol)
            if coef_simpl != 0:
                parts.append(f"({coef_simpl})*{base}")
            
    return " + ".join(parts)

def gep_simple( data_dict, Y, names, Gradient_names, dict_of_dimension,
                population, toolbox, n_generations, n_elites, hall_of_fame, 
                stats, lamda, alpha, ridge_tol, ridge_iters,
                verbose = True, tolerance = 1e-10,GEP_type = ''):

    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    start_time = time.time()
    
    is_exists = os.path.exists('output')
    if not is_exists:
        os.mkdir('output')
        
    is_exists = os.path.exists('pkl')
    if not is_exists:
        os.mkdir('pkl')
        
    
    simplified_best_list = []
    

    for gen in range(n_generations + 1):
        # # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
            
         # record statistics and log   
        if hall_of_fame is not None:
            hall_of_fame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)
        if verbose:
            print(logbook.stream)
        if gen == n_generations:
            break
            
        # selection with elitism
        elites = deap.tools.selBest(population, k=n_elites)
        
        # output the real-time result
        # Every 5 generations, the current optimal individual is checked, and if a new optimal individual appears, it is output to file.
        if gen > 0 and gen % 2 == 0:
            elites_IR = elites[0]

            try:
                simplified_best = my_simplify(elites_IR,names,Gradient_names,ridge_tol)
                print(simplified_best)

                sb_str = str(simplified_best)
                if sb_str not in simplified_best_list:
                    simplified_best_list.append(sb_str)
                    
                    elapsed = time.time() - start_time
                    time_str = '%.2f' % (elapsed)
                    
                    key = (
                            f"In generation {gen}, with PARSE running {time_str}s, \n"
                            f"The best prediction is:"
                        )
                    is_invalid = np.any(elites_IR.coef == 1e18)
                    if not is_invalid:
                        tail = f"with loss = {elites_IR.fitness.values[0]}"
                    else:
                        tail = "which is invalid!"
                    with open(f"output/{GEP_type}.dat", "a") as f:
                        f.write("\n" + key + sb_str + "\n" + tail + "\n")
            except Exception as e:
                print(f"Making an error when simplifying the expression: {str(e)}")
        
        offspring = toolbox.select(population, len(population) - n_elites)

        # replication, mutation, crossover
        offspring = [toolbox.clone(ind) for ind in offspring]
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # replace the current population with the offsprings
        population = elites + offspring


        curr = float(elites[0].fitness.values[0])
        if gen == 0:
            error_min = curr
            last_improve_gen = gen
        else:
            if curr < error_min:
                error_min = curr
                last_improve_gen = gen
            
        # Early Termination Condition
        stop_min_tol = (curr < tolerance)
        stop_no_improve = (gen - last_improve_gen) >= 200
        if stop_min_tol or stop_no_improve:
            time_now = str(datetime.datetime.now()) 
            pklFileName = time_now[:16].replace(':', '_').replace(' ', '_') 
            output_hal = open(f'pkl/{GEP_type}.pkl', 'wb') 
            str_class = pickle.dumps(population) 
            output_hal.write(str_class) 
            output_hal.close() #
            break 
    
    time_now = str(datetime.datetime.now()) 
    pklFileName = time_now[:16].replace(':', '_').replace(' ', '_') 
    output_hal = open(f'pkl/{GEP_type}.pkl', 'wb') 
    str_class = pickle.dumps(population) 
    output_hal.write(str_class) 
    output_hal.close() 
    return population, logbook


__all__ = ['gep_simple']


