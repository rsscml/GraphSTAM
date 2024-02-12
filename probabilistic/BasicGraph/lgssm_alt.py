import torch as t

def lgssmlogprob(x: t.Tensor,
                 z_prior_mean: t.Tensor,
                 z_prior_covariance: t.Tensor,
                 transition_matrix: t.Tensor,
                 transition_covariance: t.Tensor,
                 observation_matrix: t.Tensor,
                 observation_covariance: t.Tensor):
    '''
     :param x: (bs, 1)
     :param z_prior_mean: (bs, z_size, 1)
     :param z_prior_covariance: (bs, z_size, z_size) # should be positive definite

     :param transition_matrix: (bs, z_size, z_size)
     :param transition_covariance: (bs, z_size, z_size) # should be positive definite

     :param observation_matrix: (bs, z_size, 1)
     :param observation_covariance: (bs, 1, 1) # should be positive definite
    '''

    (bs, z_size, x_size) = observation_matrix.shape
    (bs, _) = x.shape

    # this version: https://github.com/rasmusbergpalm/pytorch-lgssm/blob/main/lgssm/lgssm.py
    x_mean = t.matmul(t.transpose(z_prior_mean, 1, 2), observation_matrix) #z_mean @ self.observation_matrix  # (bs, 1, 1)
    x_covariance = observation_covariance + t.transpose(observation_matrix, 1, 2) @ z_prior_covariance @ observation_matrix  # (bs, 1, 1)
    logpx = -t.distributions.MultivariateNormal(x_mean.squeeze(-1), x_covariance).log_prob(x).unsqueeze(-1)  # (bs,1)

    return logpx  # (bs,1)


def lgssmsample(num_samples,
                device,
                z_prior_mean: t.Tensor,
                z_prior_covariance: t.Tensor,
                transition_matrix: t.Tensor,
                transition_covariance: t.Tensor,
                observation_matrix: t.Tensor,
                observation_covariance: t.Tensor):
    '''
     :param num_samples: integer
     :param z_prior_mean: (bs, z_size, 1)
     :param z_prior_covariance: (bs, z_size, z_size) # should be positive definite

     :param transition_matrix: (bs, z_size, z_size)
     :param transition_covariance: (bs, z_size, z_size) # should be positive definite

     :param observation_matrix: (bs, z_size, 1)
     :param observation_covariance: (bs, 1, 1) # should be positive definite

    '''
    (bs, z_size, x_size) = observation_matrix.shape
    samples = []

    for _ in range(num_samples):
        state = t.distributions.MultivariateNormal(z_prior_mean.squeeze(-1), z_prior_covariance).sample()  # (bs, z_size)
        observation = t.distributions.MultivariateNormal(state @ observation_matrix, observation_covariance).sample().unsqueeze(-1) # (bs, x_size)
        samples.append(observation)

        #state = t.matmul(transition_matrix, init_state.unsqueeze(-1)).squeeze(-1) + t.distributions.MultivariateNormal(t.zeros(z_size).to(device=device), transition_covariance).sample()
        #state = t.matmul(transition_matrix, z_prior_mean).squeeze(-1) + t.distributions.MultivariateNormal(t.zeros(z_size).to(device=device), transition_covariance).sample()
        #observation = t.matmul(t.transpose(observation_matrix, 1, 2), state.unsqueeze(-1)).squeeze(-1) + t.distributions.MultivariateNormal(t.zeros(x_size).to(device=device), observation_covariance).sample()
        #samples.append(observation)

    return t.cat(samples, dim=1)

