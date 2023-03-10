import numpy as np
from p1 import Q1_solution


class Q2_solution(Q1_solution):

  @staticmethod
  def observation(x):
    """ Implement Q2A. Observation function without noise.
    Input:
      x: (6,) numpy array representing the state.
    Output:
      obs: (3,) numpy array representing the observation (u,v,d).
    Note:
      we define disparity to be possitive.
    """
    # Hint: this should be similar to your implemention in Q1, but with two cameras
    # stereo camera baseline
    baseline = 0.2
    
    # camera intrinsics
    K = np.array([[500., 0.,   320.],
                  [0.,   500., 240.],
                  [0.,   0.,   1.]], dtype=np.float64)

    
    # Get image projection and convert to euclidean
    uv_obs = np.dot(K, x[:3])
    uv_obs = (uv_obs / uv_obs[-1])[:2]
    
    # Calculate the disparity
    disp = K[0,0] * baseline / x[2]
    
    obs = np.array([uv_obs[0], uv_obs[1], disp], dtype=np.float64)

    return obs

  @staticmethod
  def observation_state_jacobian(x):
    """ Implement Q2B. The jacobian of observation function w.r.t state.
    Input:
      x: (6,) numpy array, the state to take jacobian with.
    Output:
      H: (3,6) numpy array, the jacobian H.
    """
    H = np.zeros((3,6))
    
    # stereo camera baseline
    baseline = 0.2
    
    # camera intrinsics
    K = np.array([[500., 0.,   320.],
                  [0.,   500., 240.],
                  [0.,   0.,   1.]], dtype=np.float64)
    
    H[0:] = [K[0,0]/x[2],  0., -K[0,0]*x[0]/(x[2]**2), 0., 0., 0.]
    H[1:] = [0.,  K[1,1]/x[2], -K[1,1]*x[1]/(x[2]**2), 0., 0., 0.]
    H[2:] = [0.,  0.,      -K[0,0]*baseline/(x[2]**2), 0., 0., 0.]
    return H

  @staticmethod
  def observation_noise_covariance():
    """ Implement Q2C here.
    Output:
      R: (3,3) numpy array, the covariance matrix for observation noise.
    """
    sig_u = 5.0
    sig_v = 5.0
    R = np.array([[sig_u, 0.0, np.sqrt(sig_u)*np.sqrt(sig_u)],
                  [0.0, sig_v, 0.0],
                  [np.sqrt(sig_u)*np.sqrt(sig_v), 0.0, sig_u+sig_v]])
    return R


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d

    np.random.seed(315)
    solution = Q2_solution()
    states, observations = solution.simulation()
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states[:,0], states[:,1], states[:,2], c=np.arange(states.shape[0]))
    plt.show()

    fig = plt.figure()
    plt.scatter(observations[:,0], observations[:,1], c=np.arange(states.shape[0]), s=4)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()

    observations = np.load('./data/Q2D_measurement.npy')
    filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.EKF(observations)
    # plotting
    true_states = np.load('./data/Q2D_state.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], c='C0')
    for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
        draw_3d(ax, cov[:3,:3], mean[:3])
    ax.view_init(elev=10., azim=45)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0], observations[:,1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        draw_2d(ax, cov[:2,:2], mean[:2])
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0]-observations[:,2], observations[:,1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        # TODO find out the mean and convariance for (u^R, v^R).
        # raise NotImplementedError()
        disp = mean[-1]
        right_cov = cov[:2, :2]
        right_mean = mean[:2] - disp
        draw_2d(ax, right_cov, right_mean)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()




