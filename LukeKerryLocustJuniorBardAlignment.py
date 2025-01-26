 import numpy as np

# Encoder/Decoder Class (from earlier)
class PiLogicEncoderDecoder:
    def __init__(self, alpha, beta, gamma, delta, epsilon, zeta, tau, T_p, dt):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.tau = tau
        self.T_p = T_p
        self.dt = dt

    def encode_frame(self, frame_data, t, M, S, V):
        P_t = np.sin(2 * np.pi * t / self.T_p) + np.cos(2 * np.pi * t / self.T_p)
        M_next = M + P_t * (self.alpha * M + self.beta * S + self.gamma * V)
        S_next = S + P_t * (self.beta * M - self.delta * S + self.epsilon * V)
        V_next = V + P_t * (self.gamma * M + self.epsilon * S - self.zeta * V)
        F_t = np.cos(S_next * V_next) * np.exp(-M_next / self.tau)
        encoded_frame = frame_data * F_t
        return encoded_frame, M_next, S_next, V_next, F_t

    def decode_frame(self, encoded_frame, F_t):
        decoded_frame = encoded_frame / F_t
        return decoded_frame

# Generate and test .plv files
encoder = PiLogicEncoderDecoder(alpha=0.6, beta=0.3, gamma=0.1, delta=0.5, epsilon=0.2, zeta=0.4, tau=10, T_p=1/np.log(2.718), dt=0.1)

# Simulated raw frame data
frame_data = np.random.rand(1920 * 1080)
M, S, V = 1.0, 1.0, 1.0

# Encoding
encoded_frames = []
field_interactions = []
for t in np.arange(0, 5, 0.1):  # Simulate 5 seconds of frames
    encoded_frame, M, S, V, F_t = encoder.encode_frame(frame_data, t, M, S, V)
    encoded_frames.append(encoded_frame)
    field_interactions.append(F_t)

# Decoding
decoded_frames = [encoder.decode_frame(frame, F) for frame, F in zip(encoded_frames, field_interactions)]

# Validation
print(f"Original Frame Sample: {frame_data[:5]}")
print(f"Decoded Frame Sample: {decoded_frames[0][:5]}")
