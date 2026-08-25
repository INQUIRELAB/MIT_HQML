"""
Post-hoc classifier heads for frozen CGCNN (or other) graph embeddings.

These modules map a fixed-size embedding vector to a single binary logit
(e.g. for `BCEWithLogitsLoss`). They are intentionally **not** wired into
training scripts yet.

Dependencies:
    - ``torch`` (always)
    - ``pennylane`` (for :class:`QuantumHead` and :class:`REUPHead`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor, nn

__all__ = [
    "LinearHead",
    "MLPHead",
    "QuantumHead",
    "REUPHead",
    "LatentMLPMatchedHead",
    "RandomFourierLogisticHead",
    "NoisyBackendUnavailableError",
    "count_parameters",
    "count_trainable_parameters",
    "reference_quantum_trainable_count",
    "reference_reup_trainable_count",
    "probe_noisy_quantum_support",
    "probe_noise_model_support",
    "probe_lightning_qubit_support",
    "build_head",
]


class NoisyBackendUnavailableError(RuntimeError):
    """Raised when ``quantum_backend='noisy'`` but PennyLane mixed simulation is unavailable."""


QuantumBackendType = Literal["exact", "shots", "noisy"]
NoiseModelType = Literal["none", "depolarizing", "readout", "amplitude_damping"]


@dataclass(frozen=True)
class QuantumSimulationConfig:
    """PennyLane device / noise settings for VQC and REUP circuits."""

    quantum_backend: str = "exact"
    shots: int | None = None
    noise_model: str = "none"
    noise_prob: float = 0.0
    pennylane_device: str | None = None
    available: bool = True
    availability_note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantum_backend": self.quantum_backend,
            "shots": self.shots,
            "noise_model": self.noise_model,
            "noise_prob": float(self.noise_prob),
            "pennylane_device": self.pennylane_device,
            "available": self.available,
            "availability_note": self.availability_note,
            **self.extra,
        }


def probe_noisy_quantum_support() -> tuple[bool, str]:
    """Return whether ``default.mixed`` can be constructed for noisy simulation."""
    try:
        import pennylane as qml
    except ImportError as e:
        return False, f"pennylane not installed: {e}"
    try:
        qml.device("default.mixed", wires=2)
        return True, "default.mixed"
    except Exception as e:
        return False, str(e)


def probe_lightning_qubit_support() -> tuple[bool, str]:
    """Return whether ``lightning.qubit`` is installed (preferred for finite shots)."""
    try:
        import pennylane as qml
    except ImportError as e:
        return False, f"pennylane not installed: {e}"
    try:
        qml.device("lightning.qubit", wires=2)
        return True, "lightning.qubit"
    except Exception as e:
        return False, str(e)


def probe_noise_model_support(noise_model: str) -> tuple[bool, str]:
    """Check that a noise channel can be executed on ``default.mixed``."""
    nm = str(noise_model).lower().strip()
    if nm in ("none", ""):
        return True, "none"
    try:
        import pennylane as qml
    except ImportError as e:
        return False, f"pennylane not installed: {e}"
    try:
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def _probe() -> Any:
            qml.Hadamard(wires=0)
            if nm == "readout":
                _apply_readout_noise(qml, [0], nm, 0.01)
            else:
                _apply_layer_noise(qml, [0], nm, 0.01)
            return qml.expval(qml.PauliZ(0))

        _probe()
        return True, nm
    except Exception as e:
        return False, str(e)


def _apply_layer_noise(qml: Any, wires: list[int], noise_model: str, noise_prob: float) -> None:
    """Mid-circuit noise after variational layers (depolarizing / amplitude damping)."""
    nm = str(noise_model).lower().strip()
    p = float(noise_prob)
    if nm in ("none", "", "readout") or p <= 0.0:
        return
    if nm == "depolarizing":
        for w in wires:
            qml.DepolarizingChannel(p, wires=w)
        return
    if nm == "amplitude_damping":
        for w in wires:
            qml.AmplitudeDamping(p, wires=w)
        return
    raise ValueError(f"Unknown layer noise_model {noise_model!r}")


def _apply_readout_noise(qml: Any, wires: list[int], noise_model: str, noise_prob: float) -> None:
    """Readout error model applied immediately before measurement."""
    nm = str(noise_model).lower().strip()
    p = float(noise_prob)
    if nm != "readout" or p <= 0.0:
        return
    for w in wires:
        qml.BitFlip(p, wires=w)


def _resolve_quantum_simulation(
    *,
    n_qubits: int,
    quantum_backend: str = "exact",
    shots: int | None = None,
    noise_model: str = "none",
    noise_prob: float = 0.0,
) -> QuantumSimulationConfig:
    """Build PennyLane device metadata; raises if noisy backend is unavailable."""
    try:
        import pennylane as qml
    except ImportError as e:
        raise ImportError(
            "Quantum heads require the `pennylane` package. Install with: pip install pennylane"
        ) from e

    backend = str(quantum_backend).lower().strip()
    nm = str(noise_model).lower().strip()
    p = float(noise_prob)

    if backend not in ("exact", "shots", "noisy"):
        raise ValueError(f"quantum_backend must be exact|shots|noisy, got {quantum_backend!r}")

    if backend == "exact":
        return QuantumSimulationConfig(
            quantum_backend=backend,
            shots=None,
            noise_model="none",
            noise_prob=0.0,
            pennylane_device="default.qubit",
            extra={"shots_on_device": None},
        )

    if backend == "shots":
        if shots is None or int(shots) < 1:
            raise ValueError(f"shots backend requires positive --shots, got {shots!r}")
        n_shots = int(shots)
        ln_ok, ln_note = probe_lightning_qubit_support()
        device_name = "lightning.qubit" if ln_ok else "default.qubit"
        return QuantumSimulationConfig(
            quantum_backend=backend,
            shots=n_shots,
            noise_model="none",
            noise_prob=0.0,
            pennylane_device=device_name,
            extra={
                "shots_on_device": n_shots,
                "shots_via": "qml.set_shots",
                "lightning_available": ln_ok,
                "lightning_note": ln_note if not ln_ok else None,
                "batching": "serial_per_sample",
            },
        )

    ok, note = probe_noisy_quantum_support()
    if not ok:
        raise NoisyBackendUnavailableError(note or "noisy backend not available")
    ch_ok, ch_note = probe_noise_model_support(nm if nm != "none" else "depolarizing")
    if nm not in ("none", "") and not ch_ok:
        raise NoisyBackendUnavailableError(f"noise_model {nm!r} not available: {ch_note}")

    return QuantumSimulationConfig(
        quantum_backend=backend,
        shots=None,
        noise_model=nm,
        noise_prob=p,
        pennylane_device="default.mixed",
        extra={"noisy_device_note": note},
    )


def _make_pennylane_device(n_qubits: int, sim: QuantumSimulationConfig) -> Any:
    """Instantiate PennyLane device from resolved simulation config."""
    import pennylane as qml

    backend = sim.quantum_backend
    if backend == "exact":
        return qml.device("default.qubit", wires=n_qubits)
    if backend == "shots":
        if sim.pennylane_device == "lightning.qubit":
            return qml.device("lightning.qubit", wires=n_qubits)
        return qml.device("default.qubit", wires=n_qubits)
    if backend == "noisy":
        return qml.device("default.mixed", wires=n_qubits)
    raise ValueError(f"Unhandled quantum_backend {backend!r}")


def _finalize_qnode(circuit: Any, sim: QuantumSimulationConfig) -> Any:
    """Apply ``qml.set_shots`` for finite-shot simulation (PennyLane >= 0.44)."""
    if sim.quantum_backend == "shots" and sim.shots is not None:
        import pennylane as qml

        return qml.set_shots(circuit, shots=int(sim.shots))
    return circuit


class _ShotsBatchSafeTorchLayer(nn.Module):
    """Run PennyLane ``TorchLayer`` one sample at a time (required for shot gradients)."""

    def __init__(self, torch_layer: nn.Module) -> None:
        super().__init__()
        self.torch_layer = torch_layer

    def forward(self, x: Tensor) -> Tensor:
        if x.size(0) <= 1:
            return self.torch_layer(x)
        chunks = [self.torch_layer(x[i : i + 1]) for i in range(x.size(0))]
        return torch.cat(chunks, dim=0)


class _ShotsSTETorchLayer(nn.Module):
    """
    Straight-through estimator for finite-shot QNodes.

    Forward uses sampled ``shots`` expectations; backward uses exact statevector
    gradients (PennyLane cannot differentiate shot estimates w.r.t. classical
    inputs in batched training — see PL #4462).
    """

    def __init__(self, exact_layer: nn.Module, shots_layer: nn.Module) -> None:
        super().__init__()
        self.exact_layer = exact_layer
        self.shots_layer = shots_layer

    def _sync_weights(self) -> None:
        with torch.no_grad():
            for p_exact, p_shots in zip(
                self.exact_layer.parameters(), self.shots_layer.parameters(), strict=True
            ):
                p_shots.copy_(p_exact)

    def forward(self, x: Tensor) -> Tensor:
        self._sync_weights()
        if self.training:
            out_exact = self.exact_layer(x)
            with torch.no_grad():
                out_shots = self.shots_layer(x)
            return out_exact + (out_shots - out_exact).detach()
        return self.shots_layer(x)

    def parameters(self, recurse: bool = True):  # noqa: ANN001
        return self.exact_layer.parameters(recurse=recurse)


def _wrap_torch_layer_for_simulation(
    torch_layer: nn.Module, sim: QuantumSimulationConfig, *, exact_layer: nn.Module | None = None
) -> nn.Module:
    if sim.quantum_backend == "shots" and exact_layer is not None:
        return _ShotsSTETorchLayer(exact_layer, torch_layer)
    if sim.quantum_backend == "shots":
        return _ShotsBatchSafeTorchLayer(torch_layer)
    return torch_layer


def _assert_batch_embed_shape(x: Tensor, input_dim: int, name: str = "x") -> None:
    """Validate ``(batch, input_dim)`` layout for embedding tensors."""
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D (batch, input_dim); got shape {tuple(x.shape)}")
    if x.size(-1) != input_dim:
        raise ValueError(
            f"{name} last dim must equal input_dim={input_dim}; got {x.size(-1)}"
        )


class LinearHead(nn.Module):
    """
    Single linear layer: ``input_dim -> 1`` logit per sample.

    Parameters
    ----------
    input_dim
        Dimension of the frozen embedding (e.g. 64 for mean-pooled CGCNN).

    Notes
    -----
    Output shape is ``(batch, 1)`` so it stacks cleanly with batch losses
    that expect a trailing class dimension.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        out = self.fc(x)
        assert out.shape == (x.size(0), 1), f"expected (batch, 1), got {tuple(out.shape)}"
        return out


class MLPHead(nn.Module):
    """
    Two-layer MLP: ``input_dim -> hidden_dim -> 1`` with ReLU and dropout.

    Parameters
    ----------
    input_dim
        Embedding size.
    hidden_dim
        Hidden width (default 32).
    dropout
        Dropout probability after ReLU (default 0.1).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        out = self.net(x)
        assert out.shape == (x.size(0), 1), f"expected (batch, 1), got {tuple(out.shape)}"
        return out


def _build_quantum_torch_layer(
    n_qubits: int,
    n_q_layers: int,
):
    """
    PennyLane QNode wrapped as ``TorchLayer``.

    ``qnode(inputs, weights)`` applies ``RY`` angle embedding (``rotation="Y"``)
    and trainable :class:`~pennylane.BasicEntanglerLayers`, then returns one
    ``PauliZ`` expectation per wire.
    """
    try:
        import pennylane as qml
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "QuantumHead requires the `pennylane` package. Install with: pip install pennylane"
        ) from e

    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if n_q_layers < 1:
        raise ValueError(f"n_q_layers must be >= 1, got {n_q_layers}")

    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: Tensor, weights: Tensor) -> list:
        # inputs: one rotation angle per qubit (scaled to ~[-pi, pi] before the call)
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=wires)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_q_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


EntanglementType = Literal["none", "linear", "ring", "full"]
EncodingType = Literal["ry", "rx_ry", "rx_ry_rz"]
MeasurementType = Literal["single_z", "all_z", "z_with_classical_readout"]


def _apply_entanglement(qml, wires: list[int], entanglement_type: str) -> int:
    """Apply entanglement and return an approximate 2q gate count."""
    et = str(entanglement_type).lower().strip()
    n = len(wires)
    if et == "none" or n <= 1:
        return 0
    if et == "linear":
        for i in range(n - 1):
            qml.CZ(wires=[wires[i], wires[i + 1]])
        return max(0, n - 1)
    if et == "ring":
        for i in range(n - 1):
            qml.CZ(wires=[wires[i], wires[i + 1]])
        if n > 2:
            qml.CZ(wires=[wires[-1], wires[0]])
            return n
        return max(0, n - 1)
    if et == "full":
        c = 0
        for i in range(n):
            for j in range(i + 1, n):
                qml.CZ(wires=[wires[i], wires[j]])
                c += 1
        return c
    raise ValueError(f"Unknown entanglement_type {entanglement_type!r}")


def _apply_encoding(qml, inputs: Tensor, wire: int, encoding_type: str, feature: Tensor) -> int:
    """Apply encoding gates on a single wire. Returns gate count for depth estimate."""
    et = str(encoding_type).lower().strip()
    if et == "ry":
        qml.RY(feature, wires=wire)
        return 1
    if et == "rx_ry":
        qml.RX(feature, wires=wire)
        qml.RY(feature, wires=wire)
        return 2
    if et == "rx_ry_rz":
        qml.RX(feature, wires=wire)
        qml.RY(feature, wires=wire)
        qml.RZ(feature, wires=wire)
        return 3
    raise ValueError(f"Unknown encoding_type {encoding_type!r}")


def _estimated_vqc_depth(
    *,
    n_qubits: int,
    n_q_layers: int,
    entanglement_type: str,
    encoding_type: str,
    measurement_type: str,
) -> int:
    """Heuristic circuit depth estimate for VQC-style circuit."""
    per_wire_enc = {"ry": 1, "rx_ry": 2, "rx_ry_rz": 3}[str(encoding_type).lower().strip()]
    enc = per_wire_enc  # per wire, assumed parallel → depth contribution is per-wire count
    ent = 0
    et = str(entanglement_type).lower().strip()
    if et == "none" or n_qubits <= 1:
        ent = 0
    elif et in ("linear", "ring"):
        ent = 1  # assume one entangling layer depth
    elif et == "full":
        ent = max(1, n_qubits - 1)
    meas = 1
    mt = str(measurement_type).lower().strip()
    if mt == "single_z":
        meas = 1
    elif mt in ("all_z", "z_with_classical_readout"):
        meas = 1
    return int(enc + n_q_layers * (3 + ent) + meas)


def _estimated_reup_depth(
    *,
    n_qubits: int,
    n_reup_layers: int,
    latent_dim: int,
    entanglement_type: str,
    encoding_type: str,
    measurement_type: str,
) -> int:
    per_wire_enc = {"ry": 1, "rx_ry": 2, "rx_ry_rz": 3}[str(encoding_type).lower().strip()]
    # REUP encodes latent_dim features per wire; assume serial on each wire
    enc = latent_dim * per_wire_enc
    et = str(entanglement_type).lower().strip()
    if et == "none" or n_qubits <= 1:
        ent = 0
    elif et in ("linear", "ring"):
        ent = 1
    else:  # full
        ent = max(1, n_qubits - 1)
    meas = 1
    return int(enc + n_reup_layers * (3 + ent) + meas)


def _build_vqc_configurable_torch_layer(
    *,
    n_qubits: int,
    n_q_layers: int,
    entanglement_type: str,
    encoding_type: str,
    measurement_type: str,
    quantum_backend: str = "exact",
    shots: int | None = None,
    noise_model: str = "none",
    noise_prob: float = 0.0,
) -> tuple[Any, QuantumSimulationConfig]:
    """Configurable VQC QNode wrapped as TorchLayer."""
    try:
        import pennylane as qml
    except ImportError as e:  # pragma: no cover
        raise ImportError("QuantumHead requires the `pennylane` package. Install with: pip install pennylane") from e

    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if n_q_layers < 1:
        raise ValueError(f"n_q_layers must be >= 1, got {n_q_layers}")

    mt = str(measurement_type).lower().strip()
    if mt not in ("single_z", "all_z", "z_with_classical_readout"):
        raise ValueError(f"Unknown measurement_type {measurement_type!r}")

    sim = _resolve_quantum_simulation(
        n_qubits=n_qubits,
        quantum_backend=quantum_backend,
        shots=shots,
        noise_model=noise_model,
        noise_prob=noise_prob,
    )
    dev = _make_pennylane_device(n_qubits, sim)
    wires = list(range(n_qubits))
    use_noise = sim.quantum_backend == "noisy"

    def _circuit_body(inputs: Tensor, weights: Tensor) -> list:
        for w in wires:
            _apply_encoding(qml, inputs, w, encoding_type, inputs[..., w])
        for ell in range(n_q_layers):
            for w in wires:
                qml.RX(weights[ell, 0, w], wires=w)
                qml.RY(weights[ell, 1, w], wires=w)
                qml.RZ(weights[ell, 2, w], wires=w)
            _apply_entanglement(qml, wires, entanglement_type)
            if use_noise:
                _apply_layer_noise(qml, wires, sim.noise_model, sim.noise_prob)
        if use_noise:
            _apply_readout_noise(qml, wires, sim.noise_model, sim.noise_prob)
        if mt == "single_z":
            return [qml.expval(qml.PauliZ(wires[0]))]
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    weight_shapes = {"weights": (n_q_layers, 3, n_qubits)}

    if sim.quantum_backend == "shots":
        sim_exact = _resolve_quantum_simulation(
            n_qubits=n_qubits,
            quantum_backend="exact",
            shots=None,
            noise_model="none",
            noise_prob=0.0,
        )
        dev_exact = _make_pennylane_device(n_qubits, sim_exact)

        @qml.qnode(dev_exact, interface="torch")
        def circuit_exact(inputs: Tensor, weights: Tensor) -> list:
            return _circuit_body(inputs, weights)

        @qml.qnode(dev, interface="torch")
        def circuit_shots(inputs: Tensor, weights: Tensor) -> list:
            return _circuit_body(inputs, weights)

        circuit_shots = _finalize_qnode(circuit_shots, sim)
        exact_layer = qml.qnn.TorchLayer(circuit_exact, weight_shapes)
        shots_layer = qml.qnn.TorchLayer(circuit_shots, weight_shapes)
        sim = QuantumSimulationConfig(
            quantum_backend=sim.quantum_backend,
            shots=sim.shots,
            noise_model=sim.noise_model,
            noise_prob=sim.noise_prob,
            pennylane_device=sim.pennylane_device,
            available=sim.available,
            availability_note=sim.availability_note,
            extra={**sim.extra, "gradient_mode": "straight_through_exact"},
        )
        return _ShotsSTETorchLayer(exact_layer, shots_layer), sim

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: Tensor, weights: Tensor) -> list:
        return _circuit_body(inputs, weights)

    circuit = _finalize_qnode(circuit, sim)
    return qml.qnn.TorchLayer(circuit, weight_shapes), sim


def _build_reup_configurable_torch_layer(
    *,
    n_qubits: int,
    n_reup_layers: int,
    latent_dim: int,
    reup_method: str,
    entanglement_type: str,
    encoding_type: str,
    measurement_type: str,
    quantum_backend: str = "exact",
    shots: int | None = None,
    noise_model: str = "none",
    noise_prob: float = 0.0,
) -> tuple[Any, QuantumSimulationConfig]:
    """Configurable REUP QNode wrapped as TorchLayer."""
    try:
        import pennylane as qml
    except ImportError as e:  # pragma: no cover
        raise ImportError("REUPHead requires the `pennylane` package. Install with: pip install pennylane") from e

    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if n_reup_layers < 1:
        raise ValueError(f"n_reup_layers must be >= 1, got {n_reup_layers}")
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
    method = reup_method.lower().strip()
    if method not in ("symmetrical", "asymmetrical"):
        raise ValueError(f"reup_method must be 'symmetrical' or 'asymmetrical', got {reup_method!r}")
    mt = str(measurement_type).lower().strip()
    if mt not in ("single_z", "all_z", "z_with_classical_readout"):
        raise ValueError(f"Unknown measurement_type {measurement_type!r}")

    sim = _resolve_quantum_simulation(
        n_qubits=n_qubits,
        quantum_backend=quantum_backend,
        shots=shots,
        noise_model=noise_model,
        noise_prob=noise_prob,
    )
    dev = _make_pennylane_device(n_qubits, sim)
    wires = list(range(n_qubits))
    use_noise = sim.quantum_backend == "noisy"

    def _circuit_body(inputs: Tensor, weights: Tensor) -> list:
        L = latent_dim
        for ell in range(n_reup_layers):
            for w in wires:
                for j in range(L):
                    idx = j if method == "symmetrical" else (w + j) % L
                    _apply_encoding(qml, inputs, w, encoding_type, inputs[..., idx])
            for w in wires:
                qml.RX(weights[ell, 0, w], wires=w)
                qml.RY(weights[ell, 1, w], wires=w)
                qml.RZ(weights[ell, 2, w], wires=w)
            if ell < n_reup_layers - 1:
                _apply_entanglement(qml, wires, entanglement_type)
            if use_noise:
                _apply_layer_noise(qml, wires, sim.noise_model, sim.noise_prob)
        if use_noise:
            _apply_readout_noise(qml, wires, sim.noise_model, sim.noise_prob)
        if mt == "single_z":
            return [qml.expval(qml.PauliZ(wires[0]))]
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    weight_shapes = {"weights": (n_reup_layers, 3, n_qubits)}

    if sim.quantum_backend == "shots":
        sim_exact = _resolve_quantum_simulation(
            n_qubits=n_qubits,
            quantum_backend="exact",
            shots=None,
            noise_model="none",
            noise_prob=0.0,
        )
        dev_exact = _make_pennylane_device(n_qubits, sim_exact)

        @qml.qnode(dev_exact, interface="torch")
        def circuit_exact(inputs: Tensor, weights: Tensor) -> list:
            return _circuit_body(inputs, weights)

        @qml.qnode(dev, interface="torch")
        def circuit_shots(inputs: Tensor, weights: Tensor) -> list:
            return _circuit_body(inputs, weights)

        circuit_shots = _finalize_qnode(circuit_shots, sim)
        exact_layer = qml.qnn.TorchLayer(circuit_exact, weight_shapes)
        shots_layer = qml.qnn.TorchLayer(circuit_shots, weight_shapes)
        sim = QuantumSimulationConfig(
            quantum_backend=sim.quantum_backend,
            shots=sim.shots,
            noise_model=sim.noise_model,
            noise_prob=sim.noise_prob,
            pennylane_device=sim.pennylane_device,
            available=sim.available,
            availability_note=sim.availability_note,
            extra={**sim.extra, "gradient_mode": "straight_through_exact"},
        )
        return _ShotsSTETorchLayer(exact_layer, shots_layer), sim

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: Tensor, weights: Tensor) -> list:
        return _circuit_body(inputs, weights)

    circuit = _finalize_qnode(circuit, sim)
    return qml.qnn.TorchLayer(circuit, weight_shapes), sim


def _init_quantum_head_weights(
    proj1: nn.Linear,
    proj2: nn.Linear,
    readout: nn.Module,
    quantum: nn.Module,
) -> None:
    """
    Mild, small-n-friendly inits: bounded classical features and small readout /
    entangler weights so logits do not explode when batch statistics are noisy.
    """
    nn.init.xavier_uniform_(proj1.weight, gain=nn.init.calculate_gain("relu"))
    nn.init.zeros_(proj1.bias)
    # Tanh path: keep features in a moderate range before * pi
    nn.init.xavier_uniform_(proj2.weight, gain=0.5)
    nn.init.zeros_(proj2.bias)
    if isinstance(readout, nn.Linear):
        nn.init.normal_(readout.weight, std=0.02)
        nn.init.zeros_(readout.bias)
    with torch.no_grad():
        for p in quantum.parameters():
            # Small rotation angles for BasicEntanglerLayers
            p.uniform_(-0.05, 0.05)


class QuantumHead(nn.Module):
    """
    Classical MLP front-end, variational QNode (``default.qubit``), then linear readout.

    Architecture
    ------------
    - ``Linear(input_dim -> proj_dim)`` → ReLU → Dropout
    - ``Linear(proj_dim -> n_qubits)`` → tanh → multiply by ``pi`` (circuit inputs)
    - QNode: ``AngleEmbedding(..., rotation="Y")`` + ``BasicEntanglerLayers``
    - Expectations: ``PauliZ`` on each wire
    - ``Linear(n_qubits -> 1)`` logit

    Forward
    -------
    Input ``[batch, input_dim]``, output ``[batch, 1]``.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 16,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        entanglement_type: str = "ring",
        encoding_type: str = "ry",
        measurement_type: str = "all_z",
        measurement_readout_hidden: int = 8,
        dropout: float = 0.1,
        quantum_backend: str = "exact",
        shots: int | None = None,
        noise_model: str = "none",
        noise_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if proj_dim < 1:
            raise ValueError(f"proj_dim must be >= 1, got {proj_dim}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_q_layers < 1:
            raise ValueError(f"n_q_layers must be >= 1, got {n_q_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.entanglement_type = str(entanglement_type).lower().strip()
        self.encoding_type = str(encoding_type).lower().strip()
        self.measurement_type = str(measurement_type).lower().strip()
        self.measurement_readout_hidden = int(measurement_readout_hidden)
        self.dropout_p = dropout
        self.quantum_backend = str(quantum_backend).lower().strip()
        self.shots = int(shots) if shots is not None else None
        self.noise_model = str(noise_model).lower().strip()
        self.noise_prob = float(noise_prob)

        self.proj1 = nn.Linear(input_dim, proj_dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=dropout)
        self.proj2 = nn.Linear(proj_dim, n_qubits)
        self.quantum, self.quantum_simulation = _build_vqc_configurable_torch_layer(
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            entanglement_type=self.entanglement_type,
            encoding_type=self.encoding_type,
            measurement_type=self.measurement_type,
            quantum_backend=self.quantum_backend,
            shots=self.shots,
            noise_model=self.noise_model,
            noise_prob=self.noise_prob,
        )
        exp_dim = 1 if self.measurement_type == "single_z" else n_qubits
        if self.measurement_type == "z_with_classical_readout":
            if self.measurement_readout_hidden < 1:
                raise ValueError("measurement_readout_hidden must be >= 1 for z_with_classical_readout.")
            self.readout = nn.Sequential(
                nn.Linear(exp_dim, self.measurement_readout_hidden),
                nn.ReLU(inplace=False),
                nn.Linear(self.measurement_readout_hidden, 1),
            )
            # Initialize last layer similarly to prior linear readout
            nn.init.normal_(self.readout[-1].weight, std=0.02)
            nn.init.zeros_(self.readout[-1].bias)
        else:
            self.readout = nn.Linear(exp_dim, 1)
            nn.init.normal_(self.readout.weight, std=0.02)
            nn.init.zeros_(self.readout.bias)

        _init_quantum_head_weights(self.proj1, self.proj2, nn.Linear(1, 1), self.quantum)
        self.estimated_circuit_depth = _estimated_vqc_depth(
            n_qubits=n_qubits,
            n_q_layers=n_q_layers,
            entanglement_type=self.entanglement_type,
            encoding_type=self.encoding_type,
            measurement_type=self.measurement_type,
        )

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)

        h = self.proj1(x)
        assert h.shape == (x.size(0), self.proj_dim), (
            f"after proj1 expected (batch, proj_dim)={(x.size(0), self.proj_dim)}, got {tuple(h.shape)}"
        )
        h = self.act(h)
        h = self.drop(h)

        z = torch.tanh(self.proj2(h))
        assert z.shape == (x.size(0), self.n_qubits), (
            f"after proj2+tanh expected (batch, n_qubits)={(x.size(0), self.n_qubits)}, got {tuple(z.shape)}"
        )
        # Scale to [-pi, pi] for RY angles (AngleEmbedding with rotation="Y")
        angles = z * math.pi

        q_out = self.quantum(angles)
        exp_dim = 1 if self.measurement_type == "single_z" else self.n_qubits
        assert q_out.shape == (x.size(0), exp_dim), (
            f"quantum output expected (batch, exp_dim)={(x.size(0), exp_dim)}, got {tuple(q_out.shape)}"
        )

        logit = self.readout(q_out)
        assert logit.shape == (x.size(0), 1), f"expected (batch, 1), got {tuple(logit.shape)}"
        return logit


def _build_reup_torch_layer(
    n_qubits: int,
    n_reup_layers: int,
    latent_dim: int,
    reup_method: str,
    entanglement: bool,
):
    """
    PennyLane QNode (data re-upload + trainable RX/RY/RZ) wrapped as ``TorchLayer``.

    **Symmetrical** re-upload: on every wire, apply ``RY(inputs[j])`` for each
    ``j in range(latent_dim)`` (same latent sequence on all qubits).

    **Asymmetrical** re-upload: wire ``w`` uses ``RY(inputs[(w + j) % latent_dim])``
    for ``j in range(latent_dim)`` (cycled latent indices per qubit).

    If ``latent_dim != n_qubits``, only the encoding uses ``latent_dim``; the
    circuit still has ``n_qubits`` wires. Indices always use ``% latent_dim``
    (asymmetrical) or the full ``inputs[0:L]`` (symmetrical).

    Trainable ``weights`` have shape ``(n_reup_layers, 3, n_qubits)`` for
    RX, RY, RZ per layer and qubit.
    """
    try:
        import pennylane as qml
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "REUPHead requires the `pennylane` package. Install with: pip install pennylane"
        ) from e

    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if n_reup_layers < 1:
        raise ValueError(f"n_reup_layers must be >= 1, got {n_reup_layers}")
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
    method = reup_method.lower().strip()
    if method not in ("symmetrical", "asymmetrical"):
        raise ValueError(f"reup_method must be 'symmetrical' or 'asymmetrical', got {reup_method!r}")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: Tensor, weights: Tensor) -> list:
        L = latent_dim
        for ell in range(n_reup_layers):
            if method == "symmetrical":
                for w in range(n_qubits):
                    for j in range(L):
                        # TorchLayer passes ``inputs`` as (batch, L); index the feature dim.
                        qml.RY(inputs[..., j], wires=w)
            else:
                for w in range(n_qubits):
                    for j in range(L):
                        qml.RY(inputs[..., (w + j) % L], wires=w)
            for w in range(n_qubits):
                qml.RX(weights[ell, 0, w], wires=w)
                qml.RY(weights[ell, 1, w], wires=w)
                qml.RZ(weights[ell, 2, w], wires=w)
            if entanglement and n_qubits > 1 and ell < n_reup_layers - 1:
                for w in range(n_qubits - 1):
                    qml.CZ(wires=[w, w + 1])
                if n_qubits > 2:
                    qml.CZ(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_reup_layers, 3, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def _init_reup_head_weights(
    proj1: nn.Linear,
    proj2: nn.Linear,
    readout: nn.Module,
    quantum: nn.Module,
) -> None:
    nn.init.xavier_uniform_(proj1.weight, gain=nn.init.calculate_gain("relu"))
    nn.init.zeros_(proj1.bias)
    nn.init.xavier_uniform_(proj2.weight, gain=0.5)
    nn.init.zeros_(proj2.bias)
    if isinstance(readout, nn.Linear):
        nn.init.normal_(readout.weight, std=0.02)
        nn.init.zeros_(readout.bias)
    with torch.no_grad():
        for p in quantum.parameters():
            p.uniform_(-0.05, 0.05)


class REUPHead(nn.Module):
    """
    Classical projection + data **re-uploading** variational circuit + linear readout.

    Compared to :class:`QuantumHead`, this head **re-encodes** the latent vector
    on every re-upload layer (symmetrical or asymmetrical RY chains) before
    trainable Euler rotations, optional ring **CZ** entanglement, then
    ``PauliZ`` expectations and a final linear logit.

    Forward: ``[batch, input_dim]`` → ``[batch, 1]``.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 16,
        n_qubits: int = 4,
        n_reup_layers: int = 4,
        reup_method: str = "symmetrical",
        entanglement: bool = True,
        entanglement_type: str | None = None,
        encoding_type: str = "ry",
        measurement_type: str = "all_z",
        measurement_readout_hidden: int = 8,
        latent_dim: int | None = None,
        dropout: float = 0.1,
        quantum_backend: str = "exact",
        shots: int | None = None,
        noise_model: str = "none",
        noise_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if proj_dim < 1:
            raise ValueError(f"proj_dim must be >= 1, got {proj_dim}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_reup_layers < 1:
            raise ValueError(f"n_reup_layers must be >= 1, got {n_reup_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        L = int(latent_dim) if latent_dim is not None else int(n_qubits)
        if L < 1:
            raise ValueError(f"latent_dim must be >= 1, got {L}")

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.n_qubits = n_qubits
        self.n_reup_layers = n_reup_layers
        self.reup_method = reup_method.lower().strip()
        self.entanglement = bool(entanglement)
        self.entanglement_type = (
            str(entanglement_type).lower().strip()
            if entanglement_type is not None
            else ("ring" if self.entanglement else "none")
        )
        self.encoding_type = str(encoding_type).lower().strip()
        self.measurement_type = str(measurement_type).lower().strip()
        self.measurement_readout_hidden = int(measurement_readout_hidden)
        self.latent_dim = L
        self.dropout_p = dropout
        self.quantum_backend = str(quantum_backend).lower().strip()
        self.shots = int(shots) if shots is not None else None
        self.noise_model = str(noise_model).lower().strip()
        self.noise_prob = float(noise_prob)

        self.proj1 = nn.Linear(input_dim, proj_dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=dropout)
        self.proj2 = nn.Linear(proj_dim, L)
        self.quantum, self.quantum_simulation = _build_reup_configurable_torch_layer(
            n_qubits=n_qubits,
            n_reup_layers=n_reup_layers,
            latent_dim=L,
            reup_method=self.reup_method,
            entanglement_type=self.entanglement_type,
            encoding_type=self.encoding_type,
            measurement_type=self.measurement_type,
            quantum_backend=self.quantum_backend,
            shots=self.shots,
            noise_model=self.noise_model,
            noise_prob=self.noise_prob,
        )
        exp_dim = 1 if self.measurement_type == "single_z" else n_qubits
        if self.measurement_type == "z_with_classical_readout":
            if self.measurement_readout_hidden < 1:
                raise ValueError("measurement_readout_hidden must be >= 1 for z_with_classical_readout.")
            self.readout = nn.Sequential(
                nn.Linear(exp_dim, self.measurement_readout_hidden),
                nn.ReLU(inplace=False),
                nn.Linear(self.measurement_readout_hidden, 1),
            )
            nn.init.normal_(self.readout[-1].weight, std=0.02)
            nn.init.zeros_(self.readout[-1].bias)
        else:
            self.readout = nn.Linear(exp_dim, 1)

        _init_reup_head_weights(self.proj1, self.proj2, nn.Linear(1, 1), self.quantum)
        self.estimated_circuit_depth = _estimated_reup_depth(
            n_qubits=n_qubits,
            n_reup_layers=n_reup_layers,
            latent_dim=L,
            entanglement_type=self.entanglement_type,
            encoding_type=self.encoding_type,
            measurement_type=self.measurement_type,
        )

    def _latent_angles(self, x: Tensor) -> Tensor:
        h = self.proj1(x)
        assert h.shape == (x.size(0), self.proj_dim), (
            f"after proj1 expected (batch, proj_dim)={(x.size(0), self.proj_dim)}, got {tuple(h.shape)}"
        )
        h = self.act(h)
        h = self.drop(h)
        z = torch.tanh(self.proj2(h))
        assert z.shape == (x.size(0), self.latent_dim), (
            f"after proj2+tanh expected (batch, latent_dim)={(x.size(0), self.latent_dim)}, got {tuple(z.shape)}"
        )
        return z * math.pi

    def get_pre_readout(self, x: Tensor) -> Tensor:
        """PauliZ expectations ``(batch, n_qubits)`` before the final linear layer."""
        _assert_batch_embed_shape(x, self.input_dim)
        angles = self._latent_angles(x)
        q_out = self.quantum(angles)
        exp_dim = 1 if self.measurement_type == "single_z" else self.n_qubits
        assert q_out.shape == (x.size(0), exp_dim), (
            f"quantum output expected (batch, exp_dim)={(x.size(0), exp_dim)}, got {tuple(q_out.shape)}"
        )
        return q_out

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        q_out = self.get_pre_readout(x)
        logit = self.readout(q_out)
        assert logit.shape == (x.size(0), 1), f"expected (batch, 1), got {tuple(logit.shape)}"
        return logit


def count_parameters(model: nn.Module) -> int:
    """Total parameter count (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reference_quantum_trainable_count(
    input_dim: int,
    proj_dim: int = 16,
    n_qubits: int = 4,
    n_q_layers: int = 2,
) -> int:
    """Trainable parameter count for :class:`QuantumHead` (classical + circuit weights)."""
    proj1 = input_dim * proj_dim + proj_dim
    proj2 = proj_dim * n_qubits + n_qubits
    readout = n_qubits + 1
    circuit = n_q_layers * n_qubits
    return int(proj1 + proj2 + readout + circuit)


def reference_reup_trainable_count(
    input_dim: int,
    proj_dim: int = 16,
    n_qubits: int = 4,
    n_reup_layers: int = 4,
    latent_dim: int | None = None,
) -> int:
    """Trainable parameter count for :class:`REUPHead` (classical + circuit weights)."""
    L = int(latent_dim) if latent_dim is not None else int(n_qubits)
    proj1 = input_dim * proj_dim + proj_dim
    proj2 = proj_dim * L + L
    readout = n_qubits + 1
    circuit = n_reup_layers * 3 * n_qubits
    return int(proj1 + proj2 + readout + circuit)


def _mlp_block_param_count(in_features: int, hidden: int, out_features: int) -> int:
    return in_features * hidden + hidden + hidden * out_features + out_features


def _pick_hidden_for_param_budget(
    in_features: int,
    out_features: int,
    target_params: int,
    *,
    max_hidden: int = 256,
) -> int:
    """Single hidden-layer width whose Linear-ReLU-Linear block has ~``target_params`` weights."""
    best_h = 1
    best_diff = float("inf")
    for h in range(1, max_hidden + 1):
        p = _mlp_block_param_count(in_features, h, out_features)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h
    return best_h


class LatentMLPMatchedHead(nn.Module):
    """
    Classical head with the same latent projection as a QML head and an MLP block
  sized to approximate the variational circuit parameter count.

    Replaces the PennyLane circuit with ``Linear → ReLU → Dropout → Linear`` on the
    latent tensor (same shape as circuit inputs / PauliZ readout width).
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        latent_dim: int,
        readout_dim: int,
        target_circuit_params: int,
        dropout: float = 0.1,
        reference_head: str = "vqc",
    ) -> None:
        super().__init__()
        if input_dim < 1 or proj_dim < 1 or latent_dim < 1 or readout_dim < 1:
            raise ValueError("input_dim, proj_dim, latent_dim, readout_dim must be >= 1")
        if target_circuit_params < 0:
            raise ValueError(f"target_circuit_params must be >= 0, got {target_circuit_params}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.latent_dim = latent_dim
        self.readout_dim = readout_dim
        self.target_circuit_params = int(target_circuit_params)
        self.reference_head = reference_head.lower().strip()
        self.dropout_p = dropout

        self.proj1 = nn.Linear(input_dim, proj_dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=dropout)
        self.proj2 = nn.Linear(proj_dim, latent_dim)

        h = _pick_hidden_for_param_budget(latent_dim, readout_dim, target_circuit_params)
        self.classical_hidden_dim = h
        self.classical = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(h, readout_dim),
        )
        self.readout = nn.Linear(readout_dim, 1)

        nn.init.xavier_uniform_(self.proj1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.5)
        nn.init.zeros_(self.proj2.bias)
        nn.init.normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def _latent(self, x: Tensor) -> Tensor:
        h = self.proj1(x)
        h = self.act(h)
        h = self.drop(h)
        z = torch.tanh(self.proj2(h)) * math.pi
        return z

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        z = self._latent(x)
        h = self.classical(z)
        logit = self.readout(h)
        assert logit.shape == (x.size(0), 1)
        return logit


class ProjBottleneckMLPHead(nn.Module):
    """Classical bottleneck matched to quantum ``proj_dim`` (no circuit-param budget).

    Architecture: ``Linear(input→proj_dim) → ReLU → Dropout → Linear(proj_dim→1)``.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 16,
        dropout: float = 0.1,
        reference_head: str = "reup",
    ) -> None:
        super().__init__()
        if input_dim < 1 or proj_dim < 1:
            raise ValueError("input_dim and proj_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.reference_head = reference_head.lower().strip()
        self.dropout_p = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(proj_dim, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        logit = self.net(x)
        assert logit.shape == (x.size(0), 1)
        return logit


class LatentBottleneckMLPHead(nn.Module):
    """Classical bottleneck matched to quantum latent / qubit width (no circuit budget).

    Architecture:
      ``Linear(input→proj_dim) → ReLU → Dropout → Linear(proj_dim→latent_dim)
        → tanh·π → Linear(latent_dim→1)``.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 16,
        latent_dim: int = 2,
        dropout: float = 0.1,
        reference_head: str = "reup",
    ) -> None:
        super().__init__()
        if input_dim < 1 or proj_dim < 1 or latent_dim < 1:
            raise ValueError("input_dim, proj_dim, latent_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.latent_dim = latent_dim
        self.reference_head = reference_head.lower().strip()
        self.dropout_p = dropout
        self.proj1 = nn.Linear(input_dim, proj_dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(p=dropout)
        self.proj2 = nn.Linear(proj_dim, latent_dim)
        self.readout = nn.Linear(latent_dim, 1)
        nn.init.xavier_uniform_(self.proj1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight, gain=0.5)
        nn.init.zeros_(self.proj2.bias)
        nn.init.normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        h = self.drop(self.act(self.proj1(x)))
        z = torch.tanh(self.proj2(h)) * math.pi
        logit = self.readout(z)
        assert logit.shape == (x.size(0), 1)
        return logit


class RandomFourierFeatures(nn.Module):
    """Fixed random Fourier feature map (cos/sin); weights are not trained."""

    def __init__(
        self,
        input_dim: int,
        fourier_dim: int,
        seed: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or fourier_dim < 1:
            raise ValueError("input_dim and fourier_dim must be >= 1")
        g = torch.Generator()
        g.manual_seed(int(seed))
        W = torch.randn(fourier_dim, input_dim, generator=g) * float(scale)
        b = 2.0 * math.pi * torch.rand(fourier_dim, generator=g)
        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.seed = int(seed)
        self.scale = float(scale)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        proj = x @ self.W.T + self.b
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class RandomFourierLogisticHead(nn.Module):
    """
    Random Fourier features (frozen) + single linear logit layer.

    Trainable parameters: ``2 * fourier_dim + 1``.
    """

    def __init__(
        self,
        input_dim: int,
        fourier_dim: int,
        seed: int,
        rff_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.fourier_dim = int(fourier_dim)
        self.seed = int(seed)
        self.rff_scale = float(rff_scale)
        self.rff = RandomFourierFeatures(input_dim, self.fourier_dim, seed=self.seed, scale=self.rff_scale)
        self.fc = nn.Linear(2 * self.fourier_dim, 1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        _assert_batch_embed_shape(x, self.input_dim)
        feats = self.rff(x)
        return self.fc(feats)


def default_fourier_dim_for_vqc_match(
    input_dim: int = 64,
    proj_dim: int = 16,
    n_qubits: int = 4,
    n_q_layers: int = 2,
) -> int:
    """``fourier_dim`` so RFF+logit trainable count ~ :func:`reference_quantum_trainable_count`."""
    target = reference_quantum_trainable_count(input_dim, proj_dim, n_qubits, n_q_layers)
    return max(1, (target - 1) // 2)


HeadType = Literal[
    "linear",
    "mlp",
    "quantum",
    "reup",
    "latent_mlp_vqc_matched",
    "latent_mlp_reup_matched",
    "proj_bottleneck_mlp",
    "latent_bottleneck_mlp",
    "random_fourier_logistic",
]


def build_head(
    head_type: str | HeadType,
    input_dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory for post-hoc heads.

    Parameters
    ----------
    head_type
        One of ``"linear"``, ``"mlp"``, ``"quantum"``, ``"reup"`` (case-insensitive).
    input_dim
        Embedding dimension.
    **kwargs
        Passed to the concrete head:

        - ``MLPHead``: ``hidden_dim`` (default 32), ``dropout`` (default 0.1).
        - ``QuantumHead``: ``proj_dim`` (default 16), ``n_qubits`` (default 4),
          ``n_q_layers`` (default 2), ``dropout`` (default 0.1).
        - ``REUPHead``: ``proj_dim`` (default 16), ``n_qubits`` (default 4),
          ``n_reup_layers`` (default 4), ``reup_method`` (``"symmetrical"`` | ``"asymmetrical"``),
          ``entanglement`` (bool), ``latent_dim`` (optional; default ``n_qubits``), ``dropout``.

    Returns
    -------
    torch.nn.Module
        A head module with ``forward(x) -> (batch, 1)``.

    Examples
    --------
    >>> build_head("linear", 64)
    >>> build_head("mlp", 64, hidden_dim=64, dropout=0.2)
    >>> build_head("quantum", 64, n_qubits=4, n_q_layers=2)
    >>> build_head("reup", 64, n_qubits=4, n_reup_layers=4, reup_method="symmetrical")
    """
    key = head_type.lower().strip()
    if key == "linear":
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='linear': {sorted(kwargs)}")
        return LinearHead(input_dim)
    if key == "mlp":
        head = MLPHead(
            input_dim,
            hidden_dim=int(kwargs.pop("hidden_dim", 32)),
            dropout=float(kwargs.pop("dropout", 0.1)),
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='mlp': {sorted(kwargs)}")
        return head
    if key == "quantum":
        shots_kw = kwargs.pop("shots", None)
        head = QuantumHead(
            input_dim,
            proj_dim=int(kwargs.pop("proj_dim", 16)),
            n_qubits=int(kwargs.pop("n_qubits", 4)),
            n_q_layers=int(kwargs.pop("n_q_layers", 2)),
            entanglement_type=str(kwargs.pop("entanglement_type", "ring")),
            encoding_type=str(kwargs.pop("encoding_type", "ry")),
            measurement_type=str(kwargs.pop("measurement_type", "all_z")),
            measurement_readout_hidden=int(kwargs.pop("measurement_readout_hidden", 8)),
            dropout=float(kwargs.pop("dropout", 0.1)),
            quantum_backend=str(kwargs.pop("quantum_backend", "exact")),
            shots=int(shots_kw) if shots_kw is not None else None,
            noise_model=str(kwargs.pop("noise_model", "none")),
            noise_prob=float(kwargs.pop("noise_prob", 0.0)),
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='quantum': {sorted(kwargs)}")
        return head
    if key == "reup":
        n_q = int(kwargs.pop("n_qubits", 4))
        latent = kwargs.pop("latent_dim", None)
        shots_kw = kwargs.pop("shots", None)
        head = REUPHead(
            input_dim,
            proj_dim=int(kwargs.pop("proj_dim", 16)),
            n_qubits=n_q,
            n_reup_layers=int(kwargs.pop("n_reup_layers", 4)),
            reup_method=str(kwargs.pop("reup_method", "symmetrical")),
            entanglement=bool(kwargs.pop("entanglement", True)),
            entanglement_type=kwargs.pop("entanglement_type", None),
            encoding_type=str(kwargs.pop("encoding_type", "ry")),
            measurement_type=str(kwargs.pop("measurement_type", "all_z")),
            measurement_readout_hidden=int(kwargs.pop("measurement_readout_hidden", 8)),
            latent_dim=int(latent) if latent is not None else None,
            dropout=float(kwargs.pop("dropout", 0.1)),
            quantum_backend=str(kwargs.pop("quantum_backend", "exact")),
            shots=int(shots_kw) if shots_kw is not None else None,
            noise_model=str(kwargs.pop("noise_model", "none")),
            noise_prob=float(kwargs.pop("noise_prob", 0.0)),
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='reup': {sorted(kwargs)}")
        return head
    if key == "latent_mlp_vqc_matched":
        proj_dim = int(kwargs.pop("proj_dim", 16))
        n_qubits = int(kwargs.pop("n_qubits", 4))
        n_q_layers = int(kwargs.pop("n_q_layers", 2))
        dropout = float(kwargs.pop("dropout", 0.1))
        target = int(kwargs.pop("target_circuit_params", n_q_layers * n_qubits))
        head = LatentMLPMatchedHead(
            input_dim,
            proj_dim=proj_dim,
            latent_dim=n_qubits,
            readout_dim=n_qubits,
            target_circuit_params=target,
            dropout=dropout,
            reference_head="vqc",
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='latent_mlp_vqc_matched': {sorted(kwargs)}")
        return head
    if key == "latent_mlp_reup_matched":
        proj_dim = int(kwargs.pop("proj_dim", 16))
        n_qubits = int(kwargs.pop("n_qubits", 4))
        n_reup_layers = int(kwargs.pop("n_reup_layers", 4))
        latent = kwargs.pop("latent_dim", None)
        L = int(latent) if latent is not None else n_qubits
        dropout = float(kwargs.pop("dropout", 0.1))
        target = int(kwargs.pop("target_circuit_params", n_reup_layers * 3 * n_qubits))
        head = LatentMLPMatchedHead(
            input_dim,
            proj_dim=proj_dim,
            latent_dim=L,
            readout_dim=n_qubits,
            target_circuit_params=target,
            dropout=dropout,
            reference_head="reup",
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='latent_mlp_reup_matched': {sorted(kwargs)}")
        return head
    if key == "proj_bottleneck_mlp":
        head = ProjBottleneckMLPHead(
            input_dim,
            proj_dim=int(kwargs.pop("proj_dim", 16)),
            dropout=float(kwargs.pop("dropout", 0.1)),
            reference_head=str(kwargs.pop("reference_head", "reup")),
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='proj_bottleneck_mlp': {sorted(kwargs)}")
        return head
    if key == "latent_bottleneck_mlp":
        latent = kwargs.pop("latent_dim", None)
        n_qubits = int(kwargs.pop("n_qubits", 2))
        latent_dim = int(latent) if latent is not None else n_qubits
        head = LatentBottleneckMLPHead(
            input_dim,
            proj_dim=int(kwargs.pop("proj_dim", 16)),
            latent_dim=latent_dim,
            dropout=float(kwargs.pop("dropout", 0.1)),
            reference_head=str(kwargs.pop("reference_head", "reup")),
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='latent_bottleneck_mlp': {sorted(kwargs)}")
        return head
    if key == "random_fourier_logistic":
        fourier_dim = kwargs.pop("fourier_dim", None)
        if fourier_dim is None:
            fourier_dim = default_fourier_dim_for_vqc_match(input_dim)
        seed = int(kwargs.pop("seed", 42))
        rff_scale = float(kwargs.pop("rff_scale", 1.0))
        head = RandomFourierLogisticHead(
            input_dim,
            fourier_dim=int(fourier_dim),
            seed=seed,
            rff_scale=rff_scale,
        )
        if kwargs:
            raise TypeError(f"Unexpected kwargs for head_type='random_fourier_logistic': {sorted(kwargs)}")
        return head
    raise ValueError(
        f"Unknown head_type {head_type!r}; expected one of "
        f"{HeadType.__args__ if hasattr(HeadType, '__args__') else HeadType}"
    )


def _self_test_reup_head() -> None:
    torch.manual_seed(0)
    m = REUPHead(
        input_dim=8,
        proj_dim=16,
        n_qubits=4,
        n_reup_layers=2,
        reup_method="symmetrical",
        entanglement=False,
        dropout=0.1,
    )
    m.train()
    x = torch.randn(5, 8, requires_grad=True)
    y = m(x)
    assert y.shape == (5, 1), y.shape
    loss = y.sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape
    n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.requires_grad)
    assert n_grad > 0
    print("REUPHead self-test ok:", y.shape, "n_params_with_grad", n_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    try:
        m = QuantumHead(input_dim=8, proj_dim=16, n_qubits=4, n_q_layers=2, dropout=0.1)
        m.train()
        x = torch.randn(4, 8, requires_grad=True)
        y = m(x)
        assert y.shape == (4, 1), y.shape
        loss = y.sum()
        loss.backward()
        assert x.grad is not None and x.grad.shape == x.shape
        n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.requires_grad)
        assert n_grad > 0
        print("QuantumHead self-test ok:", y.shape, "n_params_with_grad", n_grad)
    except ImportError as e:
        print(f"skip QuantumHead self-test: {e}")

    try:
        _self_test_reup_head()
    except ImportError as e:
        print(f"skip REUPHead self-test: {e}")
