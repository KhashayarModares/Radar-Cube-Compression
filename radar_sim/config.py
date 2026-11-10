from dataclasses import dataclass

@dataclass

class RadarParams:
    def __init__(self, N, M, fc, lam, num_rx, fs, S, Tchirp, snr_db, c0=299_792_458.0):
        self.N = N
        self.M = M
        self.fc = fc
        self.lam = lam
        self.num_rx = num_rx
        self.fs = fs
        self.S = S
        self.Tchirp = Tchirp
        self.c0 = c0
        self.snr_db = snr_db

