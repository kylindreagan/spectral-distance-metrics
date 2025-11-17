import numpy as np

# 1D example: discrete derivative d on a line graph
# f = values at vertices
f = np.array([0, 1, 2, 1, 0], dtype=float)

# df on edges (forward difference)
df = f[1:] - f[:-1]
print("df =", df)

# Discrete Hodge star * on edges: treat all edges as unit length
# *df = dual 0-form living on faces (1D: midpoints)
star_df = df  # identity for this simple example
print("star df =", star_df)

# Codifferential δ = - * d *
# d(star_df):
d_star_df = star_df[1:] - star_df[:-1]
delta_f = -d_star_df
print("delta df =", delta_f)