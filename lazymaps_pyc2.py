import numpy as np
import torch
import pyro
from scipy.sparse.linalg import LinearOperator
import time

pyrod = pyro.distributions
pyd = torch.distributions
pytd = pyd.transformed_distribution

DTYPE = torch.float32 # pytorch operates internally in float32


# diagnostic functions to measure distance/divergence from the posterior to the variational approximation

def sample_kl(dist1, dist2, num_samples):
    """ Estimates the KL divergence D(dist1|dist2) using num_samples drawn from dist1"""

    samples = dist1.sample(torch.Size([num_samples])).float()
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
    prev_time = time.time()

    # compute y = gradient log(\pi/\rho) for each sample
    for i in range(sample_size):
        x = base_dist.sample()
        #x = torch.tensor(np.load("x.npy")).float()
        x = torch.reshape(x, [1, dim])
        x.requires_grad_(True)

        bijector.event_dim = 1
        t_x = bijector(x)  # forward map of S.N. samples
        log_target_prob_term = target_log_prob(t_x)
        y = log_target_prob_term + bijector.log_abs_det_jacobian(x, t_x) - base_dist.log_prob(x) #x, event_ndims=1
        y.backward() #torch.ones(1))

        y_grad_holder.append(x.grad) #x.grad is the gradient of y with respect to x
        y_holder.append(y.detach().numpy())

        if i%250==0:
            print("Step {i}, Time: {time}".format(i=i, time=time.time()-prev_time))
            prev_time = time.time()

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
    samples = base_dist.sample(torch.Size([sample_size])).float()
    #samples = torch.load("../samples.pt")
    #print("Samples", samples)
    bijector.event_dim = 1
    t_x = bijector(samples)  # forward map of S.N. samples
    #print("forward map", t_x)

    # Log density term
    log_prob_term = target_log_prob(t_x)

    # Jacobian term
    jacobian_term = bijector.log_abs_det_jacobian(samples, t_x)

    # Add up all terms:
    objective = log_prob_term + jacobian_term - base_dist.log_prob(samples)
    return -torch.mean(objective)


def compute_var_diagnostic_tlp(base_dist, bijector, target_log_prob, sample_size):
    """Estimate variance diagnostic between posterior and variational approximation
    (T. Moselhy and Y. Marzouk. Bayesian inference with optimal maps, 2012).
     """

    # Draw samples and transform them
    samples = base_dist.sample(torch.Size([sample_size])).float()
    t_x = bijector(samples)  # forward map of S.N. samples

    # Log density term
    log_prob_term = target_log_prob(t_x)

    # Jacobian term
    jacobian_term = bijector.log_abs_det_jacobian(samples, t_x)

    # Add up all terms:
    objective = log_prob_term + jacobian_term - base_dist.log_prob(samples)
    return 0.5 * torch.var(objective)


'''
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
'''

class LinearOperatorWithDetOne(LinearOperator):
    """tf LinearOperator for rotations (U such that U U^T = U^T U = Id)"""
    def __init__(self, mat):
        super().__init__(shape=mat.shape, dtype=np.float32)
        #self.scale = torch.from_numpy(np.flip(mat, axis=0).copy()).float()
        if type(mat) is np.ndarray:
            self.scale = torch.from_numpy(mat.copy()).float()
        else:
            self.scale = mat.float()

    def _matmat(self, X):
        if self.scale.shape[1]==X.shape[0]:
            return torch.matmul(self.scale, X) #self.scale, torch.transpose(X, 0, 1))
        else:
            expanded = torch.unsqueeze(X, -1)
            return torch.squeeze(torch.matmul(self.scale, expanded), -1)

    def determinant(self):
        return torch.tensor(1).double()

    def _determinant(self):
        return torch.tensor(1).double()

    def log_abs_det_jacobian(self, x=None, y=None):
        return torch.tensor(0).double()

    def _log_abs_det_jacobian(self, x=None):
        return torch.tensor(0).double()

    def inverse(self):
        return self.adjoint()

    def get_trainable_variables(self):
        return []


class LinearOperatorScale(pyrod.torch_transform.TransformModule):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.trainable_variables = []

    def __call__(self, x):
        return self.scale._matmat(x)

    def get_trainable_variables(self):
        return []

    def log_abs_det_jacobian(self, x=None, y=None):
        return self.scale.log_abs_det_jacobian()


class LinearOperatorShift(pyrod.torch_transform.TransformModule):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def __call__(self, x):
        return x + self.shift

    def log_abs_det_jacobian(self, x=None):
        return torch.tensor(0).double()


class Blockwise(pyrod.torch_transform.TransformModule):
    def __init__(self, bij1, bij2, dims):
        super().__init__()
        self.bij1 = bij1
        self.bij2 = bij2
        self.dims = dims
        self.trainable_variables = set()

    def __call__(self, x):
        x1, x2 = torch.split(x, self.dims, dim=1)
        y1 = self.bij1(x1)
        y2 = self.bij2(x2)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y):
        y1, y2 = torch.split(y, self.dims, dim=1)
        a = self.bij1._inverse(y1)
        b = self.bij2._inverse(y2)
        return torch.cat([a, b], dim=-1)

    def _inverse(self, y):
        return self.inverse(y)

    def get_trainable_variables(self):
        self.trainable_variables.update(self.bij1.get_trainable_variables())
        self.trainable_variables.update(self.bij2.get_trainable_variables())
        return self.trainable_variables

    def log_abs_det_jacobian(self, x, y=None):
        x1, x2 = torch.split(x, self.dims, dim=1)
        ldj_1 = self.bij1.log_abs_det_jacobian(x1, self.bij1(x1))
        ldj_2 = self.bij2.log_abs_det_jacobian(x2, self.bij2(x2))
        return ldj_1+ldj_2


class ComposeTransformWithJacobian(pyrod.torch_transform.ComposeTransformModule):
    def __init__(self, parts):
        super().__init__(parts)
        self.trainable_variables = set()

    def get_trainable_variables(self):
        for part in self.parts:
            self.trainable_variables.update(part.get_trainable_variables())
        return self.trainable_variables

    '''def log_abs_det_jacobian(self, x):
        from torch.distributions.utils import _sum_rightmost
        if not self.parts:
            return torch.zeros_like(x)
        result = 0
        for part in self.parts[:-1]:
            #result = result + _sum_rightmost(part.log_abs_det_jacobian(x),
            #                                 self.event_dim - part.event_dim)
            result = result + part.log_abs_det_jacobian(x)
            x = part(x)
        part = self.parts[-1]
        #result = result + _sum_rightmost(part.log_abs_det_jacobian(x),
        #                                 self.event_dim - part.event_dim)
        result = result + part.log_abs_det_jacobian(x)
        return result'''


class AffineFlowWithInverse(pyrod.transforms.AffineAutoregressive):
    def __init__(self, input):
        super().__init__(input)

    '''def log_abs_det_jacobian(self, x):
        #_, logit_scale = self.arn(x)
        #log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        #return log_scale[:, 1] #log_scale.sum(-1)
        #return super().log_abs_det_jacobian(x, self.arn(x))
        from pyro.distributions.transforms.utils import clamp_preserve_gradients
        x_old, y_old = self._cached_x_y
        if x is not x_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        elif not self.stable:
            _, log_scale = self.arn(x)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        else:
            _, logit_scale = self.arn(x)
            log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        return log_scale.sum(-1)'''

    '''def inverse_log_abs_det_jacobian(self, y):
        return -self.log_abs_det_jacobian(self._inverse(y))'''

    def get_trainable_variables(self):
        return list(self.arn.parameters())


class Identity(pyrod.torch_transform.TransformModule):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x

    def inverse(self, y):
        return y

    def _inverse(self, y):
        return self.inverse(y)

    def log_abs_det_jacobian(self, x=None, y=None):
        return torch.tensor(0).double()

    def inverse_log_abs_det_jacobian(self, x=None):
        return torch.tensor(0).double()

    def get_trainable_variables(self):
        return []


class Invert(pyrod.torch_transform.TransformModule):
    def __init__(self, bijector):
        super().__init__()
        self.bijector = bijector
        self.trainable_variables = set()

    def __call__(self, x):
        return self.bijector._inverse(x)

    def inverse(self, y):
        return self.bijector(y)

    def _inverse(self, y):
        return self.bijector(y)

    def inverse_log_abs_det_jacobian(self, y):
        return self.bijector.log_abs_det_jacobian(y)

    def log_abs_det_jacobian(self, x):
        return self.bijector.inverse_log_abs_det_jacobian(x)

    def get_trainable_variables(self):
        self.trainable_variables.update(self.bijector.get_trainable_variables())
        return self.trainable_variables


class Flip(pyrod.torch_transform.TransformModule):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, x):
        return torch.flip(x, [self.axis])

    def log_abs_det_jacobian(self, x=None, y=None):
        return torch.tensor(0).double()

    def get_trainable_variables(self):
        return []


def make_shift_bij(dim):
    """forms a trainable shift bijector [x -> x + shift]"""
    shift = torch.zeros(dim, dtype=DTYPE)
    return LinearOperatorShift(shift=shift)


def make_diag_scale_bij(dim):
    """forms a trainable diagonal scaling bijector [x -> Dx: D is diagonal]"""
    # initialize
    w = torch.empty(dim)
    scale_init = torch.nn.init.xavier_uniform(w)
    #scale_init.requires_grad_()

    # build linear operator and return bijector
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
    #input_mat.requires_grad_()
    # input_mat = torch.autograd.Variable(scale_init, name='lower_tri_scale')
    scale_lin_oper = torch.tril(input_mat)
    return LinearOperatorScale(scale=scale_lin_oper)


def make_rotation_bij(U):
    """forms a constant rotation bijector from U [x -> Ux: U such that U U^T = U^T U = Id]"""
    operator = LinearOperatorWithDetOne(U)
    return LinearOperatorScale(scale=operator)


def make_iaf_bij(dim, num_stages, width, depth):
    """forms a trainable iaf bijector [x -> iaf(x)]"""
    bijectors = []
    hidden_dim = list(np.repeat(width, depth))
    permutation = [i for i in range(dim)]
    permutation = torch.tensor(permutation[::-1])

    for i in range(num_stages):
        made = pyro.nn.AutoRegressiveNN(dim,
                                        hidden_dim,
                                        #permutation=permutation,
                                        param_dims=[1, 1],
                                        nonlinearity=torch.nn.ELU())
        bijectors.append(AffineFlowWithInverse(made))#Invert(AffineFlowWithInverse(made)))
        bijectors.append(Flip(axis=-1))

    iaf_bij = ComposeTransformWithJacobian(list(reversed(bijectors)))
    iaf_bij(torch.zeros((dim,)).float())
    return iaf_bij


def make_lazy_bij(bij, full_dim, active_dim):
    """returns a lazy bijector [bij(active_dim) -> lazy_bij(full_dim) = [bij(active_dim) ; id(lazy_dim)]"""
    lazy_dim = full_dim - active_dim
    tmp_bij = Blockwise(bij, Identity(), [active_dim, lazy_dim])
    #tmp_bij.event_dim = 1
    return tmp_bij

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
    training_vars = training_vars.get_trainable_variables()
    opt = optimizer(list(training_vars), lr=1e-4)

    def loss_fn(base_dist, bijector, target_log_prob, sample_size):
        return compute_elbo_tlp(base_dist, bijector, target_log_prob, sample_size)

    for i in range(num_iters):
        opt.zero_grad()
        loss = loss_fn(base_dist, bijector, target_log_prob, sample_size)
        loss.backward()
        opt.step()

        if i % 10 == 0:
            step_record.append(i)
            loss_record.append(loss.item())
            time_record.append(time.time() - t_start)

            update_str = 'Step: ' + str(step_record[-1]) + \
                         ' time(s): ' + str(time_record[-1]) + \
                         ' loss: ' + str(loss_record[-1])
            print(update_str)
        #break

    t_end = time.time()

    print('Training time (s):' + str(t_end - t_start))

    return step_record, time_record, loss_record


def update_lazy_layer(bij, new_bij, base_dist, target_log_prob, optimizer, num_iters, sample_size):
    """Adds a lazy layer to bij
    In notation of https://arxiv.org/pdf/1906.00031.pdf
    bij = \mathfrak{T}_k, new_bij = T_{k+1}, lazy_bij = \mathfrak{T}_{k+1} = \mathfrak{T}_k o T_{k+1}
    """

    print("Computing H diagnostic")

    h_is_new, h_q0_new = compute_h_diagnostic_tlp(base_dist, bij, target_log_prob, 1000)
    vals_new, vecs_new = np.linalg.eigh(h_q0_new)

    vecs_new = vecs_new[:, ::-1]
    rotation_bij = make_rotation_bij(vecs_new)

    bijs = [bij, new_bij] #rotation_bij, new_bij]
    lazy_bij = ComposeTransformWithJacobian(bijs)

    #weights = torch.ones((500,500), requires_grad=True)

    #lazy_bij = LinearOperatorScale(LinearOperatorWithDetOne(weights))

    print("Beginning training")
    print(len(new_bij.get_trainable_variables()))

    step_record_layer, time_record_layer, loss_record_layer = train(base_dist,
                                                                    lazy_bij,
                                                                    new_bij, #.get_trainable_variables(),
                                                                    target_log_prob, optimizer, num_iters, sample_size)

    return lazy_bij, step_record_layer, time_record_layer, loss_record_layer