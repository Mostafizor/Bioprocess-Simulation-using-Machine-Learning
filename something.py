import torch
import numpy as np
vocab = torch.load("./Data2/Optimised_Networks/Models/k_foldT7 2_20-12_30_0.0008_40.pt")

print(np.array(vocab))