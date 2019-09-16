import pickle
import numpy as np

pickle_in = open("save_params.pickle","rb")
example_dict = pickle.load(pickle_in)

# print(example_dict)
a, b = example_dict[0]
c, d = example_dict[4]
# print(example_dict)
print(b.shape)
# print(a[31])
# example_dict = np.array(example_dict)
# example_dict = example_dict.shape
# print(example_dict)


