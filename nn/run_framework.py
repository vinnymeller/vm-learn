import numpy as np
import rune_loader as dat
from core.model import ANN
from core.error_func import sqr
from core.activation import tanh
from core.layers.dense import Dense
from core.layers.difference import Difference
from core.layers.range_normalization import RangeNormalization
from core.initializers import He
from core.optimizers.Momentum import Momentum
from viz.autoencoder_viz import Printer

training_set, evaluation_set = dat.get_data_sets()

#training_set, evaluation_set = load_images.get_data_sets()

sample = next(training_set())
n_pixels = np.prod(sample.shape)
printer = Printer(input_shape=sample.shape)

N_NODES = [60, 30, 15, 30]
n_nodes = N_NODES + [n_pixels]
dropout_rates = [.1, .1, .1, 0, 0]
model = []

model.append(RangeNormalization(training_set))

for i_layer in range(len(n_nodes)):
    new_layer = Dense(
        n_nodes[i_layer],
        activation_func=tanh,
        previous_layer=model[-1],
        optimizer=Momentum(minibatch_size = 5),
        initializer=He(),
        dropout_rate=dropout_rates[i_layer],
    )
    #new_layer.add_regularizer(L1())
    # new_layer.add_regularizer(L2())
    #new_layer.add_regularizer(Limit(4.0))
    model.append(new_layer)

model.append(Difference(model[-1], model[0]))

autoencoder = ANN(
    model=model,
    error_func=sqr,
    printer=printer,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
