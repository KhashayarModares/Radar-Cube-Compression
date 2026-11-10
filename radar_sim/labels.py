import numpy as np

class LabelProcessor:

    @staticmethod
    def compute_ground_truth_labels(traj, velocity_hist, tx_pos):
        tx_pos = np.asarray(tx_pos, dtype=float)
        traj = np.asarray(traj, dtype=float)
        velocity_hist = np.asarray(velocity_hist, dtype=float)

        n = len(traj)
        velocity_hist = velocity_hist[:n]   # match trajectory length

        delta = traj - tx_pos[None, :]
        true_range = np.linalg.norm(delta, axis=1)
        true_angle_deg = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))

        eps = 1e-12
        r_hat = delta / np.maximum(true_range[:, None], eps)

        radial_velocity = np.sum(r_hat * velocity_hist, axis=1)

        return true_range, true_angle_deg, -radial_velocity
    
    @staticmethod
    def build_frame_labels(cars_data, tx_pos, T):
        tx_pos = np.asarray(tx_pos, dtype=float)
        frame_labels = []

        for i in range(T):
            targets_i = []

            for c in cars_data:
                if i >= len(c["traj"]):
                    continue

                pos = np.asarray(c["traj"][i], dtype=float)
                vel_i = np.asarray(c["velocity_hist"][i], dtype=float)

                delta = pos - tx_pos
                rng = np.linalg.norm(delta)

                angle_deg = np.degrees(np.arctan2(delta[1], delta[0]))

                if rng > 1e-12:
                    r_hat = delta / rng
                    vr = np.dot(r_hat, vel_i)
                else:
                    vr = 0.0

                targets_i.append({
                    "name": c["obj"].name,
                    "range": float(rng),
                    "angle_deg": float(angle_deg),
                    "radial_velocity": float(vr),
                    "position": pos.copy(),
                    "velocity": vel_i.copy(),
                })

            frame_labels.append(targets_i)

        return frame_labels