import maxsimd
import numpy as np

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

print(maxsimd.dot_f32(a, b))
