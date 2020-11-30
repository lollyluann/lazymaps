import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt
import time
import iaf

pyrod = pyro.distributions
pyd = torch.distributions
pytd = pyd.transformed_distribution

DTYPE = torch.float64


# diagnostic functions to measure distance/divergence from the posterior to the variational approximation

def sample_kl(dist1, dist2, num_samples):
    """ Estimates the KL divergence D(dist1|dist2) using num_samples drawn from dist1"""

    samples = dist1.sample(num_samples)
    log_prob_1 = dist1.log_prob(samples)
    log_prob_2 = dist2.log_prob(samples)

    return np.sum(log_prob_1 - log_prob_2) / num_samples


def compute_h_diagnostic_tlp(base_dist, bijector, target_log_prob, sample_size):
    """Estimates H diagnostic matrix as defined in https://arxiv.org/pdf/1906.00031.pdf
    returns a biased (but possibly lower variance) estimate and an importance sampled estimate,
    The trace diagnostic is then 0.5*trace(h) for either matrix
    """

    dim = base_dist.event_shape[0]
    h_is = np.zeros([dim, dim])
    h_q0 = np.zeros([dim, dim])
    y_grad_holder = []
    y_holder = []
    # compute y = gradient log(\pi/\rho) for each sample
    for _ in range(sample_size):
        x = base_dist.sample()
        x = torch.reshape(x, [1, dim])
        x.requires_grad_(True)

        t_x = bijector(x)  # forward map of S.N. samples
        log_target_prob_term = target_log_prob(t_x)
        y = log_target_prob_term + bijector.log_abs_det_jacobian() - base_dist.log_prob(x) #x, event_ndims=1
        y.backward()

        y_grad = x.grad
        # with tf.GradientTape(persistent=True) as tape:
        #     tape.watch(x)
        #     t_x = bijector(x)  # forward map of S.N. samples
        #     log_target_prob_term = target_log_prob(t_x)
        #     y = log_target_prob_term + bijector.log_abs_det_jacobian(x, event_ndims=1) \
        #         - base_dist.log_prob(x)

        # y_grad = tape.gradient(y, x).numpy()
        y_grad_holder.append(y_grad)
        y_holder.append(y.detach().numpy())

    y_np = np.array(y_holder).flatten()
    weights = np.exp(y_np - np.max(y_np))
    sum_weights = np.sum(weights)

    # computed sum_y( outer product (y))
    for i in range(sample_size):
        h_is += weights[i] * np.outer(y_grad_holder[i], y_grad_holder[i])
        h_q0 += np.outer(y_grad_holder[i], y_grad_holder[i])

    # normalize
    h_is = h_is / sum_weights
    h_q0 = h_q0 / sample_size
    return h_is, h_q0


def compute_elbo_tlp(base_dist, bijector, target_log_prob, sample_size):
    """Estimate evidence lower bound (ELBO) between posterior and variational approximation """

    # Draw samples and transform them
    samples = base_dist.sample(sample_size)
    t_x = bijector(samples)  # forward map of S.N. samples

    # Log density term
    log_prob_term = target_log_prob(t_x)

    # Jacobian term
    jacobian_term = bijector.log_abs_det_jacobian(samples, event_ndims=1)

    # Add up all terms:
    objective = log_prob_term + jacobian_term - base_dist.log_prob(samples)
    return -torch.mean(objective)


def compute_var_diagnostic_tlp(base_dist, bijector, target_log_prob, sample_size):
    """Estimate variance diagnostic between posterior and variational approximation
    (T. Moselhy and Y. Marzouk. Bayesian inference with optimal maps, 2012).
     """

    # Draw samples and transform them
    samples = base_dist.sample(sample_size)
    t_x = bijector(samples)  # forward map of S.N. samples

    # Log density term
    log_prob_term = target_log_prob(t_x)

    # Jacobian term
    jacobian_term = bijector.log_abs_det_jacobian(samples)

    # Add up all terms:
    objective = log_prob_term + jacobian_term - base_dist.log_prob(samples)
    return 0.5 * torch.var(objective)


class LinearOperatorWithDetOne():
    """LinearOperator for rotations (U such that U U^T = U^T U = Id)"""
    def __init__(self, U):
        super().__init__()
        self.scale = torch.tensor(self.compute_scale(U)).double()

    # given U and you want V=SU such that VV' = V'V = I
    def compute_scale(self, U):
        eigenval, eigenvec = np.linalg.eig(U@np.transpose(U))
        div = 1 / np.sqrt(eigenval)
        scale = np.diag(div)
        return scale @ np.linalg.inv(eigenvec)

    def determinant(self):
        return torch.tensor(1).double()

    def log_abs_det_jacobian(self):
        return torch.tensor(0).double()


class LinearOperatorScale(pyrod.torch_transform.TransformModule):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.trainable_variables = set()

    def __call__(self, x):
        return torch.matmul(self.scale, x)

    def update_variables(self, vars):
        self.trainable_variables.update(vars)

    def log_abs_det_jacobian(self):
        return self.scale.log_abs_det_jacobian()


class LinearOperatorShift(pyrod.torch_transform.TransformModule):
    def __init__(self, shift):
        super().__init__()
        self.scale = shift

    def __call__(self, x):
        return x + self.shift

    def log_abs_det_jacobian(self):
        print("LinearOperatorShift has no log_abs_ function")
        pass


class Blockwise(pyrod.torch_transform.TransformModule):
    def __init__(self, bij1, bij2, dims):
        super().__init__()
        self.bij1 = bij1
        self.bij2 = bij2
        self.dims = dims
        self.trainable_variables = set()

    def __call__(self, input, x, dims):
        x1, x2 = torch.split(x, dims)
        y1 = self.bij1(x1)
        y2 = self.bij2(x2)
        return torch.cat([y1, y2])

    def update_variables(self, vars):
        self.trainable_variables.update(vars)

    def log_abs_det_jacobian(self):
        print("Blockwise has no log_abs_ function")
        pass


class Identity(pyrod.torch_transform.TransformModule):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x

    def __invert__(self, y):
        return y

    def log_abs_det_jacobian(self):
        return torch.tensor(0).double()


def make_shift_bij(dim):
    """forms a trainable shift bijector [x -> x + shift]"""
    shift = torch.zeros(dim, dtype=DTYPE, requires_grad=True)
    return LinearOperatorShift(shift=shift)


def make_diag_scale_bij(dim):
    """forms a trainable diagonal scaling bijector [x -> Dx: D is diagonal]"""
    # initialize
    w = torch.empty(dim)
    scale_init = torch.nn.init.xavier_uniform(w)
    scale_init.requires_grad_()

    # build linear operator and reutrn bijector
    scale_lin_oper = torch.diag(scale_init)
    return LinearOperatorScale(scale=scale_lin_oper)


def make_lower_tri_scale_bij(dim):
    """forms a trainable lower triangular scaling bijector [x -> Lx: L is lower tri]"""

    # initialize
    w = torch.empty(dim, dim)
    scale_init = torch.nn.init.xavier_uniform(w).numpy()

    # zero out above diagonal
    for j in range(dim):
        for i in range(j):
            scale_init[i, j] = 0

    # build linear operator and return bijector
    input_mat = torch.as_tensor(scale_init)
    input_mat.requires_grad_()
    # input_mat = torch.autograd.Variable(scale_init, name='lower_tri_scale')
    scale_lin_oper = torch.tril(input_mat)
    return LinearOperatorScale(scale=scale_lin_oper)


def make_rotation_bij(U):
    """forms a constant rotation bijector from U [x -> Ux: U such that U U^T = U^T U = Id]"""
    operator = LinearOperatorWithDetOne(U)
    return LinearOperatorScale(scale=operator.scale)


def make_iaf_bij(dim, num_stages, width, depth):
    perm = [i for i in range(dim)]
    perm = torch.tensor(perm[::-1]).double()
    hidden_dim = list(np.repeat(width, depth))

    iaf_bij = iaf.InverseAutoregressiveFlow(dim, hidden_dim, permutation=perm)
    print("come back to line 223 in make_iaf_bij")
    iaf_bij(torch.zeros((dim,)).double())
    return iaf_bij


def make_lazy_bij(bij, full_dim, active_dim):
    """returns a lazy bijector [bij(active_dim) -> lazy_bij(full_dim) = [bij(active_dim) ; id(lazy_dim)]"""
    lazy_dim = full_dim - active_dim
    # return tfb.Blockwise([bij, torch.nn.Identity()], [active_dim, lazy_dim])
    return Blockwise(bij, Identity(), [active_dim, lazy_dim])
    # when you call blockwise.forward, it splits the inputs (torch.split) and applying the two passed in bijectors


def train(base_dist, bijector, training_vars, target_log_prob, optimizer, num_iters, sample_size):
    """trains training_vars to minimize negative ELBO between base_dist and the pull-back bijector^# target

    Arguments:
        base_dist:          Starting distribution to be pushed forward to the target [\rho in \pi \approx bij_# \rho]
        bijector:           bijector defining the variational approximation to the target through (bij above)
        training_vars:      variables to be trained (could be bijector.trainable_variables, or a subset)
        target_log_prob:    (unnormalized) log probability of target
        optimizer:          tf.optimizers (e.g Adam(step_size) or SGD(step_size)
        num_iters:          number of iterations
        sample_size:        number of samples used to approximate expectation for the ELBO
    """
    loss_record = []
    time_record = []
    step_record = []
    t_start = time.time()

    def loss_fn(base_dist, bijector, target_log_prob, sample_size):
        return compute_elbo_tlp(base_dist, bijector, target_log_prob, sample_size)

    for i in range(num_iters):
        optimizer = optimizer(training_vars, lr=1e-3)
        loss = loss_fn(base_dist, bijector, target_log_prob, sample_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optimizer.minimize(loss=loss, var_list=training_vars)

        if i % 10 == 0:
            step_record.append(i)
            loss_record.append(loss().numpy())
            time_record.append(time.time() - t_start)

            update_str = 'Step: ' + str(step_record[-1]) + \
                         ' time(s): ' + str(time_record[-1]) + \
                         ' loss: ' + str(loss_record[-1])
            print(update_str)

    t_end = time.time()

    print('Training time (s):' + str(t_end - t_start))

    return step_record, time_record, loss_record


def update_lazy_layer(bij, new_bij, base_dist, target_log_prob, optimizer, num_iters, sample_size):
    """Adds a lazy layer to bij
    In notation of https://arxiv.org/pdf/1906.00031.pdf

    bij = \mathfrak{T}_k, new_bij = T_{k+1}, lazy_bij = \mathfrak{T}_{k+1} = \mathfrak{T}_k o T_{k+1}
    """

    h_is_new, h_q0_new = compute_h_diagnostic_tlp(base_dist, bij, target_log_prob, 1000)
    vals_new, vecs_new = np.linalg.eigh(h_q0_new)

    vecs_new = vecs_new[:, ::-1]
    rotation_bij = make_rotation_bij(vecs_new)

    bijs = [bij, rotation_bij, new_bij]
    lazy_bij = pyrod.torch_transform.ComposeTransformModule(bijs)

    step_record_layer, time_record_layer, loss_record_layer = train(base_dist,
                                                                    lazy_bij,
                                                                    new_bij.trainable_variables,
                                                                    target_log_prob, optimizer, num_iters, sample_size)

    return lazy_bij, step_record_layer, time_record_layer, loss_record_layer
