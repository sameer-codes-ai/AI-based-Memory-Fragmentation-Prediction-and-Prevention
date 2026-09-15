"""
PREFRAG: Deep Temporal Sequence Forecasters
Implements M1 (LSTM), M2 (GRU), and M3 (Temporal Convolutional Network) for multi-horizon forecasting.
All implementations use pure NumPy with Adam optimizer — no PyTorch/TensorFlow dependency.
"""

import sys
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler


def _render_progress_bar(
    epoch: int,
    total_epochs: int,
    loss: float,
    model_name: str = "Model",
    width: int = 25,
    prefix: str = "  -> "
) -> None:
    """Renders a responsive console progress bar with current training loss."""
    progress = min(1.0, max(0.0, epoch / total_epochs))
    filled = int(width * progress)
    bar = "=" * filled + (">" if filled < width else "") + "." * (width - filled - (1 if filled < width else 0))
    pct = int(progress * 100)
    sys.stdout.write(f"\r{prefix}{model_name:<20} | Epoch [{epoch:>{len(str(total_epochs))}}/{total_epochs}] [{bar}] {pct:>3}% | Loss: {loss:.5f}")
    sys.stdout.flush()
    if epoch >= total_epochs:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ============================================================================
# Activation Functions
# ============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clip))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _glorot_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

def _he_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    return (rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)).astype(np.float32)


# ============================================================================
# Adam Optimizer Helper
# ============================================================================

class AdamState:
    """Tracks first and second moment estimates for a set of parameters."""
    def __init__(self):
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray],
               lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.t += 1
        for key in grads:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ============================================================================
# M1: Real LSTM Forecaster
# ============================================================================

class TemporalForecasterLSTM:
    """
    M1: Long Short-Term Memory (LSTM) sequence forecaster.
    Implements real LSTM cells with forget gate, input gate, output gate, and cell state.
    Processes feature sequences through recurrent connections before a linear output layer.
    """

    def __init__(self, input_dim: int = 28, hidden_dim: int = 48, output_dim: int = 4,
                 seq_len: int = 10, learning_rate: float = 0.001, epochs: int = 80, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.lr = learning_rate
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.rng = np.random.RandomState(seed)
        self.model_name = "M1: Temporal LSTM"

        # LSTM gate weights: [input; hidden] -> gates
        # Combined weight matrix for all 4 gates: forget, input, cell_candidate, output
        combined_dim = input_dim + hidden_dim
        self.params = {
            # LSTM gates (combined: f, i, g, o)
            'Wf': _glorot_init(combined_dim, hidden_dim, self.rng),
            'bf': np.zeros(hidden_dim, dtype=np.float32) + 1.0,  # Forget gate bias init to 1
            'Wi': _glorot_init(combined_dim, hidden_dim, self.rng),
            'bi': np.zeros(hidden_dim, dtype=np.float32),
            'Wg': _glorot_init(combined_dim, hidden_dim, self.rng),
            'bg': np.zeros(hidden_dim, dtype=np.float32),
            'Wo': _glorot_init(combined_dim, hidden_dim, self.rng),
            'bo': np.zeros(hidden_dim, dtype=np.float32),
            # Output projection
            'Wy': _he_init(hidden_dim, output_dim, self.rng),
            'by': np.zeros(output_dim, dtype=np.float32),
        }
        self.adam = AdamState()

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix (N, D) into causal sequences of shape (N, seq_len, D)."""
        N, D = X.shape
        pad = np.zeros((self.seq_len - 1, D), dtype=np.float32)
        X_padded = np.vstack([pad, X])
        seqs = np.zeros((N, self.seq_len, D), dtype=np.float32)
        for i in range(N):
            seqs[i] = X_padded[i:i + self.seq_len]
        return seqs

    def _lstm_forward(self, X_seq: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through LSTM for a batch of sequences.
        X_seq: (batch, seq_len, input_dim)
        Returns: final hidden state (batch, hidden_dim) and cache for backprop
        """
        batch_size, seq_len, _ = X_seq.shape
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        cache = {'h_states': [], 'c_states': [], 'f_gates': [], 'i_gates': [],
                 'g_gates': [], 'o_gates': [], 'concat': [], 'c_prev': []}
        cache['h_states'].append(h.copy())
        cache['c_states'].append(c.copy())

        for t in range(seq_len):
            x_t = X_seq[:, t, :]  # (batch, input_dim)
            concat = np.hstack([x_t, h])  # (batch, input_dim + hidden_dim)

            # Gate computations
            f_t = _sigmoid(concat @ self.params['Wf'] + self.params['bf'])  # Forget gate
            i_t = _sigmoid(concat @ self.params['Wi'] + self.params['bi'])  # Input gate
            g_t = _tanh(concat @ self.params['Wg'] + self.params['bg'])     # Cell candidate
            o_t = _sigmoid(concat @ self.params['Wo'] + self.params['bo'])  # Output gate

            c_prev = c.copy()
            c = f_t * c + i_t * g_t  # Cell state update
            h = o_t * _tanh(c)        # Hidden state

            cache['concat'].append(concat)
            cache['f_gates'].append(f_t)
            cache['i_gates'].append(i_t)
            cache['g_gates'].append(g_t)
            cache['o_gates'].append(o_t)
            cache['c_prev'].append(c_prev)
            cache['h_states'].append(h.copy())
            cache['c_states'].append(c.copy())

        return h, cache

    def _lstm_backward(self, dh_final: np.ndarray, X_seq: np.ndarray, cache: Dict) -> Dict[str, np.ndarray]:
        """Backpropagation through time (BPTT) for LSTM."""
        batch_size, seq_len, _ = X_seq.shape
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        dh_next = dh_final.copy()
        dc_next = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(seq_len)):
            concat = cache['concat'][t]
            f_t = cache['f_gates'][t]
            i_t = cache['i_gates'][t]
            g_t = cache['g_gates'][t]
            o_t = cache['o_gates'][t]
            c_prev = cache['c_prev'][t]
            c_t = cache['c_states'][t + 1]

            # Gradients through hidden state
            dh = dh_next
            tanh_c = _tanh(c_t)

            # Output gate gradient
            do = dh * tanh_c
            do_raw = do * o_t * (1 - o_t)

            # Cell state gradient
            dc = dh * o_t * (1 - tanh_c ** 2) + dc_next

            # Forget gate gradient
            df = dc * c_prev
            df_raw = df * f_t * (1 - f_t)

            # Input gate gradient
            di = dc * g_t
            di_raw = di * i_t * (1 - i_t)

            # Cell candidate gradient
            dg = dc * i_t
            dg_raw = dg * (1 - g_t ** 2)

            # Weight gradients
            grads['Wf'] += concat.T @ df_raw / batch_size
            grads['bf'] += np.mean(df_raw, axis=0)
            grads['Wi'] += concat.T @ di_raw / batch_size
            grads['bi'] += np.mean(di_raw, axis=0)
            grads['Wg'] += concat.T @ dg_raw / batch_size
            grads['bg'] += np.mean(dg_raw, axis=0)
            grads['Wo'] += concat.T @ do_raw / batch_size
            grads['bo'] += np.mean(do_raw, axis=0)

            # Gradient to previous hidden state
            d_concat = (df_raw @ self.params['Wf'].T +
                        di_raw @ self.params['Wi'].T +
                        dg_raw @ self.params['Wg'].T +
                        do_raw @ self.params['Wo'].T)
            dh_next = d_concat[:, self.input_dim:]
            dc_next = dc * f_t

        # Clip gradients to prevent explosion
        for k in grads:
            np.clip(grads[k], -5.0, 5.0, out=grads[k])

        return grads

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> "TemporalForecasterLSTM":
        X_scaled = self.scaler.fit_transform(X)
        sequences = self._build_sequences(X_scaled)
        Y_aligned = Y
        N = len(sequences)

        if N >= 2000:
            batch_size = min(128, N)
        elif N >= 500:
            batch_size = min(64, N)
        else:
            batch_size = min(32, N)

        indices = np.arange(N)

        for epoch in range(1, self.epochs + 1):
            self.rng.shuffle(indices)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                X_batch = sequences[idx]
                Y_batch = Y_aligned[idx]

                # Forward through LSTM
                h_final, cache = self._lstm_forward(X_batch)

                # Output projection
                y_pred = h_final @ self.params['Wy'] + self.params['by']
                err_diff = y_pred - Y_batch
                batch_loss = float(np.mean(err_diff ** 2))
                epoch_loss += batch_loss
                num_batches += 1

                error = err_diff / len(idx)

                # Output layer gradients
                grads = {'Wy': h_final.T @ error, 'by': np.mean(error, axis=0)}

                # BPTT through LSTM
                dh_final = error @ self.params['Wy'].T
                lstm_grads = self._lstm_backward(dh_final, X_batch, cache)
                grads.update(lstm_grads)

                self.adam.update(self.params, grads, lr=self.lr)

            if verbose:
                avg_loss = epoch_loss / max(1, num_batches)
                _render_progress_bar(epoch, self.epochs, avg_loss, model_name=self.model_name)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        X_scaled = self.scaler.transform(X)
        sequences = self._build_sequences(X_scaled)
        h_final, _ = self._lstm_forward(sequences)
        out = h_final @ self.params['Wy'] + self.params['by']
        return np.clip(out, 0.0, 1.0)


# ============================================================================
# M2: Real GRU Forecaster
# ============================================================================

class TemporalForecasterGRU:
    """
    M2: Gated Recurrent Unit (GRU) sequence forecaster.
    Implements real GRU cells with reset gate and update gate.
    Lighter than LSTM (no separate cell state), often comparable performance.
    """

    def __init__(self, input_dim: int = 28, hidden_dim: int = 40, output_dim: int = 4,
                 seq_len: int = 10, learning_rate: float = 0.001, epochs: int = 80, seed: int = 101):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.lr = learning_rate
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.rng = np.random.RandomState(seed)
        self.model_name = "M2: Temporal GRU"

        combined_dim = input_dim + hidden_dim
        self.params = {
            # GRU gates
            'Wz': _glorot_init(combined_dim, hidden_dim, self.rng),  # Update gate
            'bz': np.zeros(hidden_dim, dtype=np.float32),
            'Wr': _glorot_init(combined_dim, hidden_dim, self.rng),  # Reset gate
            'br': np.zeros(hidden_dim, dtype=np.float32),
            'Wh': _glorot_init(combined_dim, hidden_dim, self.rng),  # Candidate hidden
            'bh': np.zeros(hidden_dim, dtype=np.float32),
            # Output projection
            'Wy': _he_init(hidden_dim, output_dim, self.rng),
            'by': np.zeros(output_dim, dtype=np.float32),
        }
        self.adam = AdamState()

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix (N, D) into causal sequences of shape (N, seq_len, D)."""
        N, D = X.shape
        pad = np.zeros((self.seq_len - 1, D), dtype=np.float32)
        X_padded = np.vstack([pad, X])
        seqs = np.zeros((N, self.seq_len, D), dtype=np.float32)
        for i in range(N):
            seqs[i] = X_padded[i:i + self.seq_len]
        return seqs

    def _gru_forward(self, X_seq: np.ndarray) -> Tuple[np.ndarray, Dict]:
        batch_size, seq_len, _ = X_seq.shape
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        cache = {'h_states': [h.copy()], 'z_gates': [], 'r_gates': [],
                 'h_candidates': [], 'concat_zr': [], 'concat_h': []}

        for t in range(seq_len):
            x_t = X_seq[:, t, :]
            concat_zr = np.hstack([x_t, h])

            z_t = _sigmoid(concat_zr @ self.params['Wz'] + self.params['bz'])  # Update gate
            r_t = _sigmoid(concat_zr @ self.params['Wr'] + self.params['br'])  # Reset gate

            concat_h = np.hstack([x_t, r_t * h])
            h_candidate = _tanh(concat_h @ self.params['Wh'] + self.params['bh'])

            h = (1 - z_t) * h + z_t * h_candidate

            cache['concat_zr'].append(concat_zr)
            cache['concat_h'].append(concat_h)
            cache['z_gates'].append(z_t)
            cache['r_gates'].append(r_t)
            cache['h_candidates'].append(h_candidate)
            cache['h_states'].append(h.copy())

        return h, cache

    def _gru_backward(self, dh_final: np.ndarray, X_seq: np.ndarray, cache: Dict) -> Dict[str, np.ndarray]:
        batch_size, seq_len, _ = X_seq.shape
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dh_next = dh_final.copy()

        for t in reversed(range(seq_len)):
            z_t = cache['z_gates'][t]
            r_t = cache['r_gates'][t]
            h_cand = cache['h_candidates'][t]
            h_prev = cache['h_states'][t]
            concat_zr = cache['concat_zr'][t]
            concat_h = cache['concat_h'][t]

            dh = dh_next

            # Candidate gradient
            dh_cand = dh * z_t
            dh_cand_raw = dh_cand * (1 - h_cand ** 2)

            grads['Wh'] += concat_h.T @ dh_cand_raw / batch_size
            grads['bh'] += np.mean(dh_cand_raw, axis=0)

            d_concat_h = dh_cand_raw @ self.params['Wh'].T
            d_rh = d_concat_h[:, self.input_dim:]

            # Update gate gradient
            dz = dh * (h_cand - h_prev)
            dz_raw = dz * z_t * (1 - z_t)

            grads['Wz'] += concat_zr.T @ dz_raw / batch_size
            grads['bz'] += np.mean(dz_raw, axis=0)

            # Reset gate gradient
            dr = d_rh * h_prev
            dr_raw = dr * r_t * (1 - r_t)

            grads['Wr'] += concat_zr.T @ dr_raw / batch_size
            grads['br'] += np.mean(dr_raw, axis=0)

            # Hidden state gradient for previous timestep
            dh_from_z = dz_raw @ self.params['Wz'].T
            dh_from_r = dr_raw @ self.params['Wr'].T
            dh_from_h = d_rh * r_t

            dh_next = dh * (1 - z_t) + dh_from_z[:, self.input_dim:] + \
                      dh_from_r[:, self.input_dim:] + dh_from_h

        for k in grads:
            np.clip(grads[k], -5.0, 5.0, out=grads[k])
        return grads

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> "TemporalForecasterGRU":
        X_scaled = self.scaler.fit_transform(X)
        sequences = self._build_sequences(X_scaled)
        Y_aligned = Y
        N = len(sequences)

        if N >= 2000:
            batch_size = min(128, N)
        elif N >= 500:
            batch_size = min(64, N)
        else:
            batch_size = min(32, N)

        indices = np.arange(N)

        for epoch in range(1, self.epochs + 1):
            self.rng.shuffle(indices)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                X_batch, Y_batch = sequences[idx], Y_aligned[idx]

                h_final, cache = self._gru_forward(X_batch)
                y_pred = h_final @ self.params['Wy'] + self.params['by']
                err_diff = y_pred - Y_batch
                batch_loss = float(np.mean(err_diff ** 2))
                epoch_loss += batch_loss
                num_batches += 1

                error = err_diff / len(idx)

                grads = {'Wy': h_final.T @ error, 'by': np.mean(error, axis=0)}
                dh_final = error @ self.params['Wy'].T
                gru_grads = self._gru_backward(dh_final, X_batch, cache)
                grads.update(gru_grads)

                self.adam.update(self.params, grads, lr=self.lr)

            if verbose:
                avg_loss = epoch_loss / max(1, num_batches)
                _render_progress_bar(epoch, self.epochs, avg_loss, model_name=self.model_name)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        X_scaled = self.scaler.transform(X)
        sequences = self._build_sequences(X_scaled)
        h_final, _ = self._gru_forward(sequences)
        out = h_final @ self.params['Wy'] + self.params['by']
        return np.clip(out, 0.0, 1.0)


# ============================================================================
# M3: Real TCN (Temporal Convolutional Network) Forecaster
# ============================================================================

class TemporalForecasterTCN:
    """
    M3: Temporal Convolutional Network (TCN) multi-horizon forecaster.
    Implements dilated causal 1D convolutions with residual connections.
    Uses kernel_size=3, dilation rates [1, 2, 4] for exponentially growing receptive field.
    """

    def __init__(self, input_dim: int = 28, hidden_dim: int = 32, output_dim: int = 4,
                 seq_len: int = 10, kernel_size: int = 3, num_layers: int = 3,
                 learning_rate: float = 0.001, epochs: int = 80, seed: int = 202):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.rng = np.random.RandomState(seed)
        self.model_name = "M3: Temporal TCN"

        # Dilated causal conv layers
        self.params = {}

        # First conv layer: input_dim -> hidden_dim
        self.params['W_conv0'] = _he_init(input_dim * kernel_size, hidden_dim, self.rng)
        self.params['b_conv0'] = np.zeros(hidden_dim, dtype=np.float32)

        # Subsequent conv layers: hidden_dim -> hidden_dim
        for l in range(1, num_layers):
            self.params[f'W_conv{l}'] = _he_init(hidden_dim * kernel_size, hidden_dim, self.rng)
            self.params[f'b_conv{l}'] = np.zeros(hidden_dim, dtype=np.float32)

        # Residual projection for first layer (input_dim != hidden_dim)
        self.params['W_res0'] = _he_init(input_dim, hidden_dim, self.rng)

        # Output projection (from last timestep)
        self.params['Wy'] = _he_init(hidden_dim, output_dim, self.rng)
        self.params['by'] = np.zeros(output_dim, dtype=np.float32)

        self.adam = AdamState()

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix (N, D) into causal sequences of shape (N, seq_len, D)."""
        N, D = X.shape
        pad = np.zeros((self.seq_len - 1, D), dtype=np.float32)
        X_padded = np.vstack([pad, X])
        seqs = np.zeros((N, self.seq_len, D), dtype=np.float32)
        for i in range(N):
            seqs[i] = X_padded[i:i + self.seq_len]
        return seqs

    def _causal_conv1d(self, X: np.ndarray, W: np.ndarray, b: np.ndarray,
                       dilation: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dilated causal 1D convolution.
        X: (batch, seq_len, channels_in)
        W: (channels_in * kernel_size, channels_out)
        Returns: (batch, seq_len, channels_out), padded_input for backprop
        """
        batch, seq_len, C_in = X.shape
        C_out = W.shape[1]
        K = self.kernel_size

        # Causal padding: pad the beginning so output length == input length
        pad_size = (K - 1) * dilation
        X_padded = np.concatenate([np.zeros((batch, pad_size, C_in), dtype=np.float32), X], axis=1)

        out = np.zeros((batch, seq_len, C_out), dtype=np.float32)
        for t in range(seq_len):
            # Gather K samples at dilated positions
            patches = []
            for k in range(K):
                idx = t + pad_size - k * dilation
                patches.append(X_padded[:, idx, :])  # (batch, C_in)
            patch = np.concatenate(patches, axis=1)  # (batch, C_in * K)
            out[:, t, :] = patch @ W + b

        return out, X_padded

    def _tcn_forward(self, X_seq: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass through all TCN layers with residual connections."""
        cache = {'inputs': [], 'pre_relu': [], 'padded': []}
        h = X_seq  # (batch, seq_len, input_dim)

        for l in range(self.num_layers):
            dilation = 2 ** l
            cache['inputs'].append(h.copy())

            conv_out, padded = self._causal_conv1d(
                h, self.params[f'W_conv{l}'], self.params[f'b_conv{l}'], dilation)
            cache['padded'].append(padded)
            cache['pre_relu'].append(conv_out.copy())

            # Residual connection
            if l == 0:
                # Project residual from input_dim to hidden_dim
                residual = h @ self.params['W_res0']
            else:
                residual = h

            h = _relu(conv_out) + residual

        # Take last timestep
        h_final = h[:, -1, :]  # (batch, hidden_dim)
        cache['final_seq'] = h
        return h_final, cache

    def _tcn_backward(self, dh_final: np.ndarray, X_seq: np.ndarray, cache: Dict) -> Dict[str, np.ndarray]:
        """Backpropagation through TCN layers."""
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        batch, seq_len, _ = X_seq.shape

        # Gradient flows from last timestep
        dh_seq = np.zeros_like(cache['final_seq'])
        dh_seq[:, -1, :] = dh_final

        for l in reversed(range(self.num_layers)):
            dilation = 2 ** l
            h_in = cache['inputs'][l]
            pre_relu = cache['pre_relu'][l]

            # Through residual connection
            if l == 0:
                d_residual = dh_seq.copy()
                # Gradient through residual projection
                for t in range(seq_len):
                    grads['W_res0'] += h_in[:, t, :].T @ d_residual[:, t, :] / batch
            else:
                d_residual = dh_seq.copy()

            # Through ReLU
            d_conv = dh_seq * (pre_relu > 0)

            # Through causal conv (simplified gradient accumulation)
            C_in = h_in.shape[2]
            K = self.kernel_size
            pad_size = (K - 1) * dilation
            padded = cache['padded'][l]

            dW = np.zeros_like(self.params[f'W_conv{l}'])
            db = np.zeros_like(self.params[f'b_conv{l}'])
            d_padded = np.zeros_like(padded)

            for t in range(seq_len):
                patches = []
                patch_indices = []
                for k in range(K):
                    idx = t + pad_size - k * dilation
                    patches.append(padded[:, idx, :])
                    patch_indices.append(idx)
                patch = np.concatenate(patches, axis=1)

                d_out_t = d_conv[:, t, :]  # (batch, C_out)
                dW += patch.T @ d_out_t / batch
                db += np.mean(d_out_t, axis=0)

                d_patch = d_out_t @ self.params[f'W_conv{l}'].T  # (batch, C_in * K)
                for k_idx, pidx in enumerate(patch_indices):
                    d_padded[:, pidx, :] += d_patch[:, k_idx * C_in:(k_idx + 1) * C_in]

            grads[f'W_conv{l}'] = dW
            grads[f'b_conv{l}'] = db

            # Pass gradient to previous layer (strip padding)
            if l > 0:
                dh_seq = d_padded[:, pad_size:, :] + d_residual

        for k in grads:
            np.clip(grads[k], -5.0, 5.0, out=grads[k])
        return grads

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> "TemporalForecasterTCN":
        X_scaled = self.scaler.fit_transform(X)
        sequences = self._build_sequences(X_scaled)
        Y_aligned = Y
        N = len(sequences)

        if N >= 2000:
            batch_size = min(128, N)
        elif N >= 500:
            batch_size = min(64, N)
        else:
            batch_size = min(32, N)

        indices = np.arange(N)

        for epoch in range(1, self.epochs + 1):
            self.rng.shuffle(indices)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                X_batch, Y_batch = sequences[idx], Y_aligned[idx]

                h_final, cache = self._tcn_forward(X_batch)
                y_pred = h_final @ self.params['Wy'] + self.params['by']
                err_diff = y_pred - Y_batch
                batch_loss = float(np.mean(err_diff ** 2))
                epoch_loss += batch_loss
                num_batches += 1

                error = err_diff / len(idx)

                grads = {'Wy': h_final.T @ error, 'by': np.mean(error, axis=0)}
                dh_final = error @ self.params['Wy'].T
                tcn_grads = self._tcn_backward(dh_final, X_batch, cache)
                grads.update(tcn_grads)

                self.adam.update(self.params, grads, lr=self.lr)

            if verbose:
                avg_loss = epoch_loss / max(1, num_batches)
                _render_progress_bar(epoch, self.epochs, avg_loss, model_name=self.model_name)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        X_scaled = self.scaler.transform(X)
        sequences = self._build_sequences(X_scaled)
        h_final, _ = self._tcn_forward(sequences)
        out = h_final @ self.params['Wy'] + self.params['by']
        return np.clip(out, 0.0, 1.0)
