import numpy as np

# TODO fill out the following class, reusing your implementation from the 
# prior problems
class Q3_solution(object):

  @staticmethod
  def system_matrix():
    """ 
    Output:
      A: 6x6 numpy array for the system matrix.
    """
    # Hint: this is the same as in part A
    A = np.eye(6)
    # TODO fill in values for A below
    # Hint: only 9 entries should be non-zero, with values 1, 0.1, and 0.8
    A[0][3] = 0.1
    A[1][4] = 0.1
    A[2][5] = 0.1
    A[3][3] = 0.8
    A[4][4] = 0.8
    A[5][5] = 0.8
    
    return A

  @staticmethod
  def process_noise_covariance():
    """ 
    Output:
      Q: 6x6 numpy array for the covariance matrix.
    """
    Q = np.eye(6)
    # TODO fill in values for Q below
    # Hint: only 3 entries should be non-zero, and all should equal 0.05
    Q = np.diag([0,0,0,0.05, 0.05, 0.05])
    return Q

  @staticmethod
  def observation_noise_covariance():
    """ 
    Output:
      R: 2x2 numpy array for the covariance matrix.
    """
    sigma = np.diag([0.005, 0.005, 0.01])
    return sigma

  def KF(self, observations, mu_0, sigma_0, remove_outliers):
    """ Implement Kalman filtering 
    Input:
      observations: (N,3) numpy array, the sequence of observations. From T=1.
      mu_0: (6,) numpy array, the mean of state belief after T=0
      sigma_0: (6,6) numpy array, the covariance matrix for state belief after T=0.
      remove_outliers: bool, whether to remove outliers
    Output:
      state_mean: (N+1,6) numpy array, the filtered mean state at each time step, starting at T=0 
      state_sigma: (N+1,6,6) numpy array, the filtered state covariance at each time step, starting at T=0 
      predicted_observation_mean: (N,2) numpy array, the mean of predicted observations. Start from T=1
      predicted_observation_sigma: (N,2,2) numpy array, the covariance matrix of predicted observations. Start from T=1
    """
    A = self.system_matrix()
    Q = self.process_noise_covariance()
    R = self.observation_noise_covariance()
    state_mean = [mu_0]
    state_sigma = [sigma_0]
    predicted_observation_mean = []
    predicted_observation_sigma = []
    for ob in observations:
        mu_bar_next = np.dot(A, state_mean[-1])  # TODO fill this in
        sigma_bar_next = np.dot(A, np.dot(state_sigma[-1], A.T)) + Q # TODO fill this in
        H =np.array([[1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0.]], dtype=np.float64)# TODO fill this in
        kalman_gain_numerator = np.dot(sigma_bar_next, H.T) # TODO fill this in
        kalman_gain_denominator = np.dot(H, np.dot(sigma_bar_next, H.T)) + R # TODO fill this in
        kalman_gain = np.dot(kalman_gain_numerator, np.linalg.inv(kalman_gain_denominator)) # TODO fill this in
        expected_observation = np.dot(H, mu_bar_next) # TODO fill this in
        # for D outlier
        mu_next = mu_bar_next + np.dot(kalman_gain, (ob - expected_observation).T) # TODO fill this in
        sigma_next = np.dot((np.eye(6, dtype=np.float64) - np.dot(kalman_gain, H)), sigma_bar_next) # TODO fill this in
        deviation = np.sqrt(np.dot((ob - expected_observation).T, np.dot(np.linalg.inv(kalman_gain_denominator), (ob - expected_observation)))) # TODO fill this in
        if not filter_outliers or deviation <= 25:# part D
            mu_next = mu_bar_next + np.dot(kalman_gain, (ob - expected_observation).T) # TODO fill this in
            sigma_next = np.dot((np.eye(6, dtype=np.float64) - np.dot(kalman_gain, H)), sigma_bar_next) # TODO fill this in
        else:
            mu_next = mu_bar_next # TODO fill this in
            sigma_next = sigma_bar_next # TODO fill this in
        state_mean.append(mu_next)
        state_sigma.append(sigma_next)
        predicted_observation_mean.append(expected_observation)
        predicted_observation_sigma.append(kalman_gain_denominator)
    return state_mean, state_sigma, predicted_observation_mean, predicted_observation_sigma


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d
    part_b_data = False
    filter_outliers = True
    
    solution = Q3_solution()
    
    if part_b_data:
        state_positions = np.load('data/Q3B_data/Q3B_positions_gt.npy')
        observations = np.load('./data/Q3B_predictions.npy')
    else:
        state_positions = np.load('data/Q3D_data/Q3D_positions_gt.npy')
        observations = np.load('./data/Q3D_predictions.npy')
        
    state_0 = np.concatenate([state_positions[0], np.zeros(3)])
    sigma_0 = np.eye(6)*0.01
    sigma_0[3:,3:] = 0.0
    state_mean, state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.KF(observations[1:], state_0, sigma_0, filter_outliers)
    state_mean = np.array(state_mean)
    dev = np.linalg.norm(state_positions-state_mean[:,:3], axis=1)
    err = np.linalg.norm(state_positions-observations[:,:3], axis=1)
    
    # 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(state_mean[:,0], state_mean[:,1], state_mean[:,2], c='C1')
    ax.scatter(state_positions[:,0], state_positions[:,1], state_positions[:,2], c='C0')
    plt.show()
    
    if not part_b_data:
      # 2d plotting
      fig = plt.figure()
      plt.scatter(state_positions[:,0], state_positions[:,2], c='C0')
      state_mean = np.array(state_mean)
      plt.scatter(state_mean[:,0], state_mean[:,2], c='C1')
      plt.show()

      fig = plt.figure()
      plt.scatter(state_positions[:,1], state_positions[:,2], c='C0')
      state_mean = np.array(state_mean)
      plt.scatter(state_mean[:,1], state_mean[:,2], c='C1')
      plt.show()
      
      if filter_outliers:
          def projection(points_3d):
              intrinsic = np.array([[500,0,320],[0,500,240],[0,0,1]])
              points_proj = np.dot(points_3d, intrinsic.transpose())
              points_proj = points_proj[:,:2]/points_proj[:,2:3]
              return points_proj

          gt_state_proj = projection(state_positions)
          detection_proj = projection(observations)
          filtered_state_proj = projection(np.array(state_mean)[:,:3])

          fig = plt.figure()
          ax = fig.add_subplot(111)
          ax.scatter(gt_state_proj[:,0], gt_state_proj[:,1], s=4)
          ax.scatter(detection_proj[:,0], detection_proj[:,1], s=4)
          ax.scatter(filtered_state_proj[:,0], filtered_state_proj[:,1], s=4)
          plt.xlim([0,640])
          plt.ylim([0,480])
          plt.gca().invert_yaxis()
          plt.show()



