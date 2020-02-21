import 	rune_loader 					as 		rl
import	nn_framework.framework 			as 		framework
import 	nn_framework.layer     			as 		layer
import 	nn_framework.activation 		as 		activation
import 	nn_framework.error_func 		as 		error_func
from 	nn_framework.regularization 	import 	L1, L2
from 	nn_framework.autoencoder_viz 	import 	Printer

##
N_NODES = [20]

training_set, evaluation_set = rl.get_data_sets()
sample = next(training_set())
expected_input_range = (0, 1)
n_pixels = sample.shape[0] * sample.shape[1]
printer = Printer(input_shape=sample.shape)

n_nodes = [n_pixels] + N_NODES + [n_pixels]
dropout_rates = [0.2, 0.5]
model = []
model.append(layer.RangeNormalization(training_set))

for i in range(len(n_nodes)-1):
	new_layer = layer.Dense(
		n_nodes[i],
		activation.tanh,
		previous_layer=model[-1]
		#dropout_rate = dropout_rates[i]
	)
	new_layer.add_regularizer(L1())
	#new_layer.add_regularizer(L2())
	#new_layer.add_regularizer(Limit(1.0))
	model.append(new_layer)

model.append(layer.Difference(model[-1], model[0]))

autoencoder = framework.ANN(
    model=model,
    expected_input_range=expected_input_range,
    error_func=error_func.sqr,
    printer=printer
)

autoencoder.train(training_set=training_set)
autoencoder.evaluate(evaluation_set=evaluation_set)