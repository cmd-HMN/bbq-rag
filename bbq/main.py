import maxsimd
import numpy as np

dim, q_len, doc_lengths = 4, 2, [3, 2]
q = np.random.rand(q_len * dim).astype(np.float32)
d = np.random.rand(sum(doc_lengths) * dim).astype(np.float32)

result = maxsimd.maxsim_vrlen(q, d, doc_lengths, q_len, dim)
print(f"Success!\n{result}\nType: {type(result)}")
