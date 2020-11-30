import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import lazymaps_pyc as lm
from sklearn import preprocessing
import torch
import pyro

mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
plt.close('all')

pyd = torch.distributions
pyrod = pyro.distributions
type_double = torch.float64


# import data
train = pd.read_csv("paper_examples/pd_speech_features.csv")

# set up data set

# Using first 500 feautures
all_x = train.values[1:, 1:501].astype(np.float64)
all_x = preprocessing.scale(all_x)
all_y = train.values[1:, -1].astype(np.float64)

# 605 = 80% of observations
num_pats = 605
y_train = all_y[:num_pats]
x_train = torch.tensor(all_x[:num_pats,:]).double()


# get data dimensions
num_observations = x_train.shape[0]
num_features = x_train.shape[1]    # + 1 to account for constant # + 1 to account for constant

# define base distribution as standard normal
scale = torch.ones([num_features]).double()
base_dist = pyrod.torch.MultivariateNormal(loc=torch.zeros([num_features]).double(), scale_tril=torch.diag(scale))

# define prior as zero mean, with variance
prior_var = 100
prior_std = np.sqrt(prior_var)

# prior = tfd.MultivariateNormalDiag(
#     loc=tf.zeros([num_features], tf_double),
#     scale_diag=tf.cast(tf.fill([num_features],prior_var),tf_double)
# )

# log-likelihood
def log_l(sample):
    z = torch.matmul(x_train, torch.transpose(sample, 0, -1))

    theta_1 = torch.nn.Softplus()(-z)
    theta_2 = torch.nn.Softplus()(z)

    repeat_number = sample.shape[0]
    Y = np.array([y_train for _ in range(repeat_number)]).transpose()
    Y = torch.tensor(Y.transpose()).double()

    log_like_terms = -Y * theta_1 + (Y - 1) * theta_2
    return log_like_terms.sum()


# Un-normalized target density (whitened)
def unnormalized_log_prob(samples):
    result = base_dist.log_prob(samples)
    for sample in samples:
         result += log_l(sample*prior_std)
    return result

# IAF hyper-parameters
num_stages = 4
width = 200 #128
depth = 2


# training hyper-parameters
sample_size = 100
num_iters = 20000
optimizer = torch.optim.Adam

# example to run
example = 2

if example == 0:
    dim = num_features
    iaf_bij = lm.make_iaf_bij(dim=dim, num_stages=num_stages, width=width, depth=depth)
    bij = iaf_bij

    step_record, time_record, loss_record = lm.train(base_dist,
                                                     bij,
                                                     bij.trainable_variables,
                                                     unnormalized_log_prob,
                                                     optimizer,
                                                     num_iters,
                                                     sample_size)

# U-IAF
if example == 1:
    bij = lm.Identity()
    dim = num_features

    # iaf bij
    iaf_bij = lm.make_iaf_bij(dim=dim, num_stages=num_stages, width=width, depth=depth)

    # form lazy map
    new_bij = lm.make_lazy_bij(iaf_bij, num_features, dim)

    # try runnign with nonlazy iaf to see if it's much slower

    bij, step_record_layer, time_record_layer, loss_record_layer = lm.update_lazy_layer(bij,
                                                                                        new_bij,
                                                                                        base_dist,
                                                                                        unnormalized_log_prob,
                                                                                        optimizer,
                                                                                        num_iters,
                                                                                        sample_size)

if example == 2:
    num_layers = 3
    bij = lm.Identity()

    dim = 200

    # testing layers of lazy
    step_record = []
    time_record = []
    loss_record = []
    for i in range(num_layers):

        # iaf bij
        iaf_bij = lm.make_iaf_bij(dim=dim, num_stages=num_stages, width=width, depth=depth)

        # form lazy map
        new_bij = lm.make_lazy_bij(iaf_bij, num_features, dim)

        bij, step_record_layer, time_record_layer, loss_record_layer = lm.update_lazy_layer(bij,
                                                                                            new_bij,
                                                                                            base_dist,
                                                                                            unnormalized_log_prob,
                                                                                            optimizer,
                                                                                            num_iters//num_layers,
                                                                                            sample_size)

        step_record.append(step_record_layer)
        time_record.append(time_record_layer)
        loss_record.append(loss_record_layer)



var_diag = lm.compute_var_diagnostic_tlp(base_dist, bij, unnormalized_log_prob, int(1e5))
elbo = lm.compute_elbo_tlp(base_dist, bij, unnormalized_log_prob, int(1e5))
h_is, h_q0 = lm.compute_h_diagnostic_tlp(base_dist, bij, unnormalized_log_prob, int(1e5))
h_is_trace = np.trace(h_is)
h_q0_trace = np.trace(h_q0)
print('RAN EXAMPLE: '+str(example))
print('==DIAGNOSTICS==')
print('ELBO: ' + str(elbo.numpy()))
print('Var: ' + str(var_diag.numpy()))
print('trace (IS): ' + str(h_is_trace))
print('trace (q0): ' + str(h_q0_trace))
