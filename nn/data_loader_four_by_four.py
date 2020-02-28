import numpy as np

def get_data_sets():

    examples = []
    for i in range(20):
        examples.append(np.random.rand(4,4))


    def training_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    def evaluation_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    return training_set, evaluation_set