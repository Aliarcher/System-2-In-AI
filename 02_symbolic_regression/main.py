"""Symbolic Regression (tiny): brute-force expressions of depth ≤2 from a small DSL
and pick the one with lowest MSE on y = x^2 + x for x in [-3..3]."""
import itertools, math


X = list(range(-3,4))
Y = [x*x + x for x in X]


consts = [-2,-1,0,1,2]
ops = {
'add': lambda a,b: (f'({a[0]}+{b[0]})', [x+y for x,y in zip(a[1], b[1])]),
'sub': lambda a,b: (f'({a[0]}-{b[0]})', [x-y for x,y in zip(a[1], b[1])]),
'mul': lambda a,b: (f'({a[0]}*{b[0]})', [x*y for x,y in zip(a[1], b[1])]),
}


atoms = [("x", X)] + [(str(c), [c]*len(X)) for c in consts]


candidates = list(atoms)
# depth-2 compositions
for (a,b) in itertools.product(atoms, repeat=2):
    for name,op in ops.items():
        candidates.append(op(a,b))


# Evaluate MSE
best = None
for expr, vals in candidates:
    mse = sum((y-v)**2 for y,v in zip(Y, vals))/len(X)
    if math.isfinite(mse):
        if not best or mse < best[0]:
            best = (mse, expr)


print("Target: y = x^2 + x")
print("Best found expression:", best[1])
print("MSE:", round(best[0], 4))