import torch


#------------------------STRidge ------------------------
def STRidge(X, Y, lamda, tol=1e-1, iters=10): 
    n_features = X.shape[1]
    col_var = torch.var(X, dim=0)
    valid_cols = col_var > 1e-8
    X_clean = X[:, valid_cols]

    if X_clean.shape[1] == 0:
        return torch.zeros((n_features, 1), device=X.device, dtype=X.dtype)
    n_clean = X_clean.shape[1]
    I = torch.eye(n_clean, device=X.device, dtype=X.dtype)
    A = X_clean.T @ X_clean + lamda * I
    B = X_clean.T @ Y
    w_clean = torch.linalg.pinv(A) @ B  
    
    # Threshold iteration
    for _ in range(iters):
        small = torch.abs(w_clean) < tol
        if not torch.any(small):
            break
        
        w_clean = w_clean.clone()   
        w_clean[small] = 0
        
        big = ~small.squeeze()
        if big.sum() == 0:
            break
            
        Xb = X_clean[:, big]
        Ib = torch.eye(big.sum(), device=X.device, dtype=X.dtype)
        Ab = Xb.T @ Xb + lamda * Ib
        Bb = Xb.T @ Y
        
        w_big = torch.linalg.pinv(Ab) @ Bb
        w_clean[big] = w_big.reshape(-1, 1)
    w_full = torch.zeros((n_features, 1), device=X.device, dtype=X.dtype)
    w_full[valid_cols] = w_clean
    
    return w_full


#------------------------Loss calculation ------------------------
#  Mean Relative Error + Punishment weight
def compute_loss(X, Y, coef, alpha=1e-3, ridge_tol=1e-1):
    
    Yp = X @ coef
    mre = torch.abs(Yp - Y) / (torch.abs(Y) + 1e-10)
    w_valid_sum = int((torch.abs(coef) > ridge_tol).sum().item())
    loss = torch.mean(mre) + alpha * w_valid_sum
    return float(loss.detach().cpu().item()), w_valid_sum



def Trim(X_full, Y, best_err, lamda, alpha=1e-3, min_terms=1,
                             ridge_tol=1e-2, ridge_iters=10):
    """
    Trim deletion: Start from the entire column and attempt to delete column by column.
    As long as the loss does not worsen after deletion, accept the deletion.
    Continue until it is impossible to delete any more or the number of columns is less than or equal to min_terms.
    """
    device = X_full.device
    n_terms = X_full.shape[1]
    keep = torch.ones(n_terms, device=device, dtype=torch.bool)

    improved = True
    while improved and int(keep.sum().item()) > min_terms:
        improved = False
        round_best_err = best_err
        round_best_drop = None

        keep_ids = torch.where(keep)[0]  
        for idx in keep_ids:
            keep_try = keep.clone()
            keep_try[idx] = False

            if int(keep_try.sum().item()) < min_terms:
                continue

            X_del = X_full[:, keep_try]
            coef_del = STRidge(X_del, Y, lamda, tol=ridge_tol, iters=ridge_iters)
            err_del, _ = compute_loss(X_del, Y, coef_del, alpha=alpha, ridge_tol=ridge_tol)
            if err_del <= round_best_err:
                round_best_err = err_del
                round_best_drop = int(idx.item())
                
        if round_best_drop is not None:
            keep[round_best_drop] = False
            X_new = X_full[:, keep]
            coef_new = STRidge(X_new, Y, lamda, tol=ridge_tol, iters=ridge_iters)
            best_err, _ = compute_loss(X_new, Y, coef_new, alpha=alpha, ridge_tol=ridge_tol)
            improved = True

    # Output the final optimal solution
    X_best = X_full[:, keep]
    coef_best = STRidge(X_best, Y, lamda, tol=ridge_tol, iters=ridge_iters)
    best_err, _ = compute_loss(X_best, Y, coef_best, alpha=alpha, ridge_tol=ridge_tol)

    return keep, coef_best, best_err
