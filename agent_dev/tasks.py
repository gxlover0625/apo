import json
import random
import numpy as np

def load_logical7():
    data_path = '/root/shared-nvme/apo/opro/data/BIG-Bench-Hard-data/logical_deduction_seven_objects.json'
    with open(data_path, 'r') as f:
        data = json.load(f)['examples']
    random.seed(42)
    np.random.seed(42)
    random.shuffle(data)
    train_data = data[:50]
    eval_data = data[50:150]
    test_data = data[150:]
    return train_data, eval_data, test_data

# train_data, eval_data, test_data = load_logical7()