import numpy as np
import torch

import Calculation_Operator as co
import Dimension_Verification as dv
import Function_Optimization as fo



#------------------------ Data cache ------------------------
_gpu_cache = {}

# Move numpy arrays to the GPU
def get_gpu_data(data_dict, names, device="cuda", dtype=torch.float64):
    key = (id(data_dict), tuple(names), device, dtype)
    if key in _gpu_cache:
        return _gpu_cache[key]

    gpu_dict = {}
    for n in names:
        arr = data_dict[n]
        if torch.is_tensor(arr):
            gpu_dict[n] = arr.to(device=device, dtype=dtype)
        else:
            gpu_dict[n] = torch.tensor(arr, device=device, dtype=dtype)
    _gpu_cache[key] = gpu_dict
    return gpu_dict

# The expression "gene × base"
def build_combo(gene, base_name):
    gene_str = str(gene).strip().replace('\n', '') 
    mul_term = f"mul({gene_str}, {base_name})"                
    return  mul_term

def my_compile_(term, gpu_data ):
    ns = {
        "add": co.add_torch,
        "sub": co.sub_torch,
        "mul": co.mul_torch,
        "protected_div": co.protected_div_torch,
        "linker_add": co.linker_add,
    }
    ns.update(gpu_data)
    return eval(term, {}, ns)


def LossFunc(individual, data_dict, Y, names, Gradient_names, 
             use_dim_verify, dict_of_dimension, target_dimension, num_units,
             use_Trim, lamda, alpha, ridge_tol = 0.1, ridge_iters = 10,
             device = "cuda"):

    terms = []
    local_ij = []
    
    if use_dim_verify:  
    # Search for the target dimension，whether to satisfy dimensional homogeneity
        required_gene_dims = [] 
        for base_name in Gradient_names: 
            base_dim =  np.array( dict_of_dimension.get(base_name) )
            required_gene_dim = np.array(target_dimension) - base_dim
            required_gene_dims.append(required_gene_dim)    
        gene_dims = dv.dimensional_verification(individual, dict_of_dimension, num_units)
        gene_dims = [np.array(g) if not isinstance(g, bool) else None for g in gene_dims]
    
        for j, base in enumerate(Gradient_names):
            for i, ind in enumerate(individual):
                if np.array_equal(required_gene_dims[j], gene_dims[i]):
                    term = build_combo(ind, base)
                    terms.append(term)   
                    local_ij.append([i,j])
        # If not dimensional homogeneous, we would identify it as an invalid individual.           
        if len(terms) == 0:
            return 1.0e18,  
            
    else:
    # Dimensionless data processing
        for j, base in enumerate(Gradient_names):
            for i, ind in enumerate(individual):
                term = build_combo(ind, base)
                terms.append(term)   
                local_ij.append([i,j])        

    
    # Data deduplication to prevent linear correlation in the matrix
    unique_terms = []
    unique_local = []
    seen = set()
    for t, ij in zip(terms, local_ij):
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)
            unique_local.append(ij)
    terms = unique_terms
    local_ij = unique_local  
    
    gpu_data = get_gpu_data(data_dict, names, device=device)
    
    # Switch to GPU computing
    Y_t = torch.tensor(Y, device=device, dtype=torch.float64).reshape(-1, 1)
    n_samples = Y_t.shape[0]
    n_valid = len(terms)
    X_valid = torch.empty((n_samples, n_valid), device=device, dtype=torch.float64)
    
    for j, term in enumerate(terms):
        col = my_compile_(term, gpu_data)

        X_valid[:, j] = col.reshape(-1)    
    X_valid = torch.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

    coef_best = fo.STRidge(X_valid, Y_t, lamda, tol=ridge_tol, iters=ridge_iters)
    loss_best, _ = fo.compute_loss(X_valid, Y_t, coef_best, alpha=alpha, ridge_tol=ridge_tol)
    terms_best = terms
    local_best = local_ij   

    
    # Whether to use Trimming Strategy
    if use_Trim and loss_best < 0.4:
        
        keep, coef_best, loss_best = fo.Trim(X_valid, Y_t, loss_best, lamda, 
                                            alpha=alpha, min_terms=2,
                                            ridge_tol=ridge_tol, ridge_iters=ridge_iters
                                            )
        keep_cpu = keep.detach().cpu().tolist()
        terms_best = [t for t, k in zip(terms, keep_cpu) if k]
        local_best = [ij for ij, k in zip(local_ij, keep_cpu) if k]


    # 8) individual
    individual.coef = coef_best.detach().cpu().numpy().reshape(-1).copy()
    individual.terms = terms_best
    individual.local_ij = local_best

    return loss_best,



