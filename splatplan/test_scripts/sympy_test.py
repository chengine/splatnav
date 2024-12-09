#%% 
import sympy as sym
import numpy as np
import time

def b_spline_term_derivs(pts, deg, d):
    # terms are (K choose k)(1-t)**(K-k) * t**k
    terms = []
    for i in range(deg + 1):
        scaling = scipy.special.comb(deg, i)
        t = sym.Symbol('t')
        term = []
        for pt in pts:
            term.append(scaling * sym.diff((1-t)**(deg - i) *t**i, t, d).subs(t, pt))
        terms.append(np.array(term))

    return np.array(terms).astype(np.float32)

deg = sym.Symbol('deg')
k = sym.Symbol('k')
t = sym.Symbol('t')
T = sym.Symbol('T')

tnow = time.time()
rows = []

for deriv in range(4):
    columns = []
    if deriv == 0:
        deriv_fx = (1- (t) )**(deg - k) * ( (t) **k)
    else:
        deriv_fx = sym.diff( deriv_fx , t, 1)
    for k_ in range(7):
        fx = sym.simplify(deriv_fx.subs(deg, 6).subs(k, k_))
        columns.append(fx)
    rows.append(columns)
function_matrix = sym.Matrix(rows)
print('Elapsed', time.time() - tnow)

mat_fx = sym.lambdify(t, function_matrix, 'numpy')
#%%
tnow = time.time()
output = mat_fx(np.linspace(0, 1, 10))
print('Elapsed', time.time() - tnow)
#(sym.diff( (1- t )**(deg - k) * ( t **k) , t, 3) / T**3).subs(t, 0.3 / 2).subs(deg, 6).subs(k, 3).subs(T, 2)

# sym.diff( (1- (t/T) )**(deg - k) * ( (t/T) **k) , t, 2)

# sym.diff( (1- (t/T) )**(deg - k) * ( (t/T) **k) , t, 3)

# sym.diff( (1- (t/T) )**(deg - k) * ( (t/T) **k) , t, 4)

#print('Elapsed', time.time() - tnow)
# %%
