import sys
sys.path.append('..')
import numpy as np
from radar_sim.config import RadarParams
from sklearn.cluster import DBSCAN

class PostProcessing:
  def __init__(self,theta_res, cfg):
    self.theta_res = theta_res
    self.N = cfg.N
    self.M = cfg.M
    self.lam = cfg.lam
    self.num_rx = cfg.num_rx
    self.fs = cfg.fs
    self.S = cfg.S
    self.Tchirp = cfg.Tchirp
    self.c0 = cfg.c0



  def compute_RD_RA(self, x):
      theta = np.linspace(-np.pi/3, +np.pi/3, self.theta_res)  # -60..+60 deg
      A = self.ula_steering_matrix(theta=theta, d=self.lam/2, sign=-1)
      w_fast = np.hanning(self.N)[None, None, :]
      w_slow = np.hanning(self.M)[None, :, None]
      det_list = []
      # (8, M, N) * (1,M,1) * (1,1,N)
      xw = x * w_slow * w_fast

      # Range FFT with proper shift
      Xr = np.fft.fftshift(np.fft.fft(xw, axis=2), axes=2)
      f_b = np.fft.fftshift(np.fft.fftfreq(self.N, d=1/self.fs))
      ranges = self.c0 * f_b / (2*self.S)
      #r_guard = 1.0  # meters, tune this
      #Xr[:, :, np.abs(ranges) < r_guard] = 0
      # Stationary clutter removal in slow time:
      # subtract mean over chirps for each RX and range bin
      Xr_clutter_removed = self.clutter_fir_2tap_method(Xr)
      # Doppler FFT
      Xrd = np.fft.fftshift(np.fft.fft(Xr_clutter_removed, axis=1), axes=1)
      f_d = np.fft.fftshift(np.fft.fftfreq(self.M, d=self.Tchirp))
      vel = (self.lam/2.0) * f_d

      RD = 10*np.log10(np.sum(np.abs(Xrd)**2, axis=0) + 1e-12)
      RD_lin = np.sum(np.abs(Xrd)**2, axis=0)
      detections = self.go_cfar_2d(RD_lin)
      # Range Angle Map
      '''
      m0 = pick_m0_from_rd_peak(RD, ranges, rmin=5.0, rmax=ranges[-1])
      X_slice = Xrd[:, m0, :]              # (8, K), complex
      Y = A.conj().T @ X_slice             # (Ntheta, K)
      P = np.abs(Y)**2                     # (Ntheta, K)
      '''
      clusters = self.cluster_detections(detections, RD_lin, ranges, vel)

      for c in clusters:
        m, k = c["index"]
        x = Xrd[:,m, k]
        y = A.conj().T @ x
        p = np.abs(y)**2

        theta_hat = theta[np.argmax(p)]
        det_list.append({
            "range": c["range"],
            "velocity": c["velocity"],
            "angle_deg": np.degrees(theta_hat),
            "index": (m, k)
        })

      # Complete RA map: noncoherent integration over Doppler
      # For each Doppler m: beamform Xrd[:, m, :] -> (Ntheta, N)
      P_ra = np.zeros((len(theta), Xrd.shape[2]), dtype=float)
      for m in range(Xrd.shape[1]):
          Y_m = A.conj().T @ Xrd[:, m, :]     # (Ntheta, N)
          P_ra += np.abs(Y_m)**2

      RA = 10 * np.log10(P_ra + 1e-12)

      # point cloud
      point_cloud, point_list = self.detections_to_point_cloud(
          clusters, Xrd, theta, A, ranges, vel, RD_lin
      )

      return RD, RA, ranges, vel, detections, clusters, point_cloud, point_list

  @staticmethod
  def clutter_mean_method(Xr):
    return Xr - np.mean(Xr, axis=1,keepdims=True)
  @staticmethod
  def clutter_ema_method(Xr, alpha=0.5):
      """
      Recursive EMA clutter suppression along slow time.

      Parameters
      ----------
      Xr : ndarray, shape (num_rx, M, N)
          Range-FFT data
      alpha : float
          EMA coefficient, typically 0.9 to 0.99

      Returns
      -------
      Y : ndarray, shape (num_rx, M, N)
          Clutter-suppressed output
      """

      Y = np.zeros_like(Xr, dtype=Xr.dtype)

      # Initialize clutter estimate with first chirp
      c_hat = Xr[:, 0, :].copy()
      Y[:, 0, :] = Xr[:, 0, :] - c_hat

      for m in range(1, Xr.shape[1]):
          x_m = Xr[:, m, :]
          c_hat = alpha * c_hat + (1.0 - alpha) * x_m
          Y[:, m, :] = x_m - c_hat

      return Y

  @staticmethod
  def clutter_fir_2tap_method(Xr):
      """
      2-tap FIR MTI clutter suppression:
          y[m] = x[m] - x[m-1]

      Parameters
      ----------
      Xr : ndarray, shape (num_rx, M, N)

      Returns
      -------
      Y : ndarray, shape (num_rx, M, N)
      """
      Y = np.zeros_like(Xr, dtype=Xr.dtype)
      Y[:, 1:, :] = Xr[:, 1:, :] - Xr[:, :-1, :]
      return Y

  def ula_steering_matrix(self, theta, d=None, sign=-1):
    """
    Returns A of shape (num_rx, Ntheta), where each column is a(θ).
    sign = -1 is the common receive steering convention.
    """
    num_rx = self.num_rx
    lam = self.lam

    if d is None:
        d = lam/2
    n = np.arange(num_rx)[:, None]          # (num_rx, 1)
    th = theta[None, :]                     # (1, Ntheta)
    phase = sign * 2*np.pi * (d/lam) * n * np.sin(th)
    A = np.exp(1j * phase)                  # (num_rx, Ntheta)
    A = A / np.sqrt(num_rx)                 # normalize
    return A


  @staticmethod
  def ca_cfar(RD_lin, Pfa = 1e-10, Tr=16, Td=12, Gr=10, Gd=6):

    M, N = RD_lin.shape        #M: slow, N: fast
    detections = np.zeros_like(RD_lin, dtype=bool)

    N_train = (2*Tr + 2*Gr + 1)*(2*Td + 2*Gd + 1) - (2*Gr + 1)*(2*Gd + 1)
    alpha = N_train * (Pfa**(-1/N_train) - 1)



    for i in range(Tr+Gr, N-(Tr+Gr)):
        for j in range(Td+Gd, M-(Td+Gd)):

          window = RD_lin[i-(Tr+Gr):i+(Tr+Gr)+1,
                          j-(Td+Gd):j+(Td+Gd)+1
          ]

          guard = RD_lin[i-Gr:i+Gr+1,
                         j-Gd:j+Gd+1
          ]

          noise = (np.sum(window) - np.sum(guard)) / N_train

          if RD_lin[i,j] > noise * alpha:
            detections[i,j] = True

    return detections
  
  @staticmethod
  def go_cfar_2d(RD_lin, Pfa=1e-5, Tr=12, Td=8, Gr=4, Gd=2):

    M, K = RD_lin.shape
    detections = np.zeros_like(RD_lin, dtype=bool)

    # Number of training cells per side (range split)
    N_half = Tr * (2*Td + 2*Gd + 1)

    # Threshold scaling
    alpha = N_half * (Pfa**(-1/N_half) - 1)

    for i in range(Tr+Gr, M-(Tr+Gr)):
        for j in range(Td+Gd, K-(Td+Gd)):

            window = RD_lin[
                i-(Tr+Gr):i+(Tr+Gr)+1,
                j-(Td+Gd):j+(Td+Gd)+1
            ]

            # Remove guard region
            mask = np.ones_like(window, dtype=bool)

            r0 = Tr
            r1 = Tr + 2*Gr + 1
            c0 = Td
            c1 = Td + 2*Gd + 1

            mask[r0:r1, c0:c1] = False

            training = window[mask]

            # Split into left and right halves (range direction)
            mid = training.size // 2

            left = training[:mid]
            right = training[mid:]

            if left.size == 0 or right.size == 0:
                continue

            noise_L = np.mean(left)
            noise_R = np.mean(right)

            noise_est = max(noise_L, noise_R)

            threshold = alpha * noise_est

            if RD_lin[i, j] > threshold:
                detections[i, j] = True

    return detections


  @staticmethod
  def cluster_detections(det_map, RD_lin, ranges, vel,
                      eps_r=0.5, eps_v=0.5, min_samples=3):

    det_indices = np.argwhere(det_map)
    if len(det_indices) == 0:
        return []

    points = []
    strengths = []

    for (m, k) in det_indices:
        points.append([ranges[k], vel[m]])
        strengths.append(RD_lin[m, k])

    points = np.array(points)
    strengths = np.array(strengths)

    # normalize dimensions
    norm_points = np.column_stack([
        points[:, 0] / eps_r,
        points[:, 1] / eps_v
    ])

    clustering = DBSCAN(eps=1.0,
                        min_samples=min_samples).fit(norm_points)

    labels = clustering.labels_

    clusters = []
    for lbl in set(labels):
        if lbl == -1:              # Noisy samples are given the label -1
            continue

        idx = np.where(labels == lbl)[0]
        strongest = idx[np.argmax(strengths[idx])]

        clusters.append({
            "range": points[strongest, 0],
            "velocity": points[strongest, 1],
            "index": det_indices[strongest]
        })

    return clusters

  def detections_to_point_cloud(self, clusters, Xrd, theta, A, ranges, vel, RD_lin):
      """
      Convert clustered RD detections into a radar point cloud.

      Returns
      -------
      point_cloud : ndarray, shape (Npts, 6)
          Columns:
          [x, y, velocity, range, angle_deg, power]
      point_list : list of dict
          More descriptive version of the same output
      """
      points = []
      point_list = []

      for c in clusters:
          m, k = c["index"]

          # antenna snapshot for one RD cell
          x_mk = Xrd[:, m, k]                    # (num_rx,)
          y = A.conj().T @ x_mk                  # (Ntheta,)
          p = np.abs(y)**2

          idx_max = np.argmax(p)
          theta_hat = theta[idx_max]             # radians
          theta_deg = np.degrees(theta_hat)

          r = ranges[k]
          v = vel[m]
          power = p[idx_max]

          # polar -> Cartesian
          x = r * np.cos(theta_hat)
          y_cart = r * np.sin(theta_hat)

          points.append([x, y_cart, v, r, theta_deg, power])

          point_list.append({
              "x": x,
              "y": y_cart,
              "velocity": v,
              "range": r,
              "angle_deg": theta_deg,
              "power": power,
              "doppler_bin": int(m),
              "range_bin": int(k)
          })

      if len(points) == 0:
          point_cloud = np.zeros((0, 6), dtype=float)
      else:
          point_cloud = np.array(points, dtype=float)

      return point_cloud, point_list