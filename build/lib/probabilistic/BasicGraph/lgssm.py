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

     :param observation_matrix: (bs, z_size, x_size)
     :param observation_covariance: (bs, x_size, x_size) # should be positive definite
    '''

    (bs, z_size, x_size) = observation_matrix.shape
    (bs, _) = x.shape

    # transition
    # ((bs, z_size, z_size), (bs, z_size, 1)) -> (bs, z_size, 1)
    z_mu_posterior = t.matmul(transition_matrix, z_prior_mean)
    # (bs, z_size, z_size)
    z_cov_posterior = t.matmul(t.matmul(transition_matrix, z_prior_covariance), t.transpose(transition_matrix, 1, 2)) + transition_covariance
    z = t.distributions.MultivariateNormal(z_mu_posterior.squeeze(-1), z_cov_posterior).sample().unsqueeze(-1) # (bs,z_size,1)

    # Compute the likelihood of the observation
    observation_dist = t.distributions.MultivariateNormal(t.matmul(t.transpose(observation_matrix, 1, 2), z).squeeze(-1), observation_covariance)
    logprob = -observation_dist.log_prob(x).unsqueeze(-1)

    return logprob  # (bs,1)

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

     :param observation_matrix: (bs, z_size, x_size)
     :param observation_covariance: (bs, x_size, x_size) # should be positive definite

    '''
    (bs, z_size, x_size) = observation_matrix.shape
    samples = []

    for _ in range(num_samples):
        # ((bs, z_size, z_size), (bs, z_size, 1)) -> (bs, z_size, 1)
        z_mu_posterior = t.matmul(transition_matrix, z_prior_mean)
        # (bs, z_size, z_size)
        z_cov_posterior = t.matmul(t.matmul(transition_matrix, z_prior_covariance), t.transpose(transition_matrix, 1, 2)) + transition_covariance
        z = t.distributions.MultivariateNormal(z_mu_posterior.squeeze(-1), z_cov_posterior).sample()  # (bs,z_size)

        #init_state = t.distributions.MultivariateNormal(z_prior_mean.squeeze(-1), z_prior_covariance).sample()  # (bs, z_size)
        #state = t.matmul(transition_matrix, init_state.unsqueeze(-1)).squeeze(-1) + t.distributions.MultivariateNormal(t.zeros(z_size).to(device=device), transition_covariance).sample()
        #state = t.matmul(transition_matrix, z_prior_mean).squeeze(-1) + t.distributions.MultivariateNormal(t.zeros(z_size).to(device=device), transition_covariance).sample()
        observation = t.matmul(t.transpose(observation_matrix, 1, 2), z.unsqueeze(-1)).squeeze(-1) + t.distributions.MultivariateNormal(t.zeros(x_size).to(device=device), observation_covariance).sample()
        samples.append(observation)

    return t.cat(samples, dim=1)

