"""
Poema ROM Generator — Compile Discovered Laws into Executable Poems
===================================================================

Converts ROM models discovered by the Autopoietic Scientist into
executable Poema AST representations that can be compiled and run
by the Gideon engine.

A "Poem" in this context is a mathematical program that encodes:
  1. The ROM dynamics: ȧ = L·a + Q(a,a) + c + ν_t·D·a
  2. The Koopman basis: φ_k(x) = cos(k·arccos(x)) (Chebyshev)
  3. The closure model: ν_t from ERGON diagnostics
  4. Verification conditions: ROM-1, ROM-2, ROM-3 certificates

The generated Poem can be:
  - Executed directly via RK4 integration
  - Compiled to Gideon IR for hardware-optimized execution
  - Exported to symbolic form for Lean 4 verification
  - Serialized to JSON for persistence

ARCHITECTURE:
  ROMModel (from rom_synthesizer) → PoemROM → GideonIR → execution

This module bridges the discovery phase (SINDy/Galerkin) with the
execution phase (Gideon), completing the autopoietic loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PoemROMNode:
    """A node in the Poem ROM AST."""
    kind: str                    # "linear", "quadratic", "constant", "closure", "rk4", "output"
    operands: List[str]          # Input variable names
    result: str                  # Output variable name
    parameters: Dict[str, Any]   # Numerical parameters (matrices, vectors)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoemROM:
    """
    Complete Poem representation of a ROM.

    This is the "law" compiled into an executable mathematical program.
    """
    name: str
    n_modes: int
    dt: float
    n_steps: int
    nodes: List[PoemROMNode]
    L: np.ndarray                     # Linear operator
    Q: np.ndarray                     # Quadratic tensor
    c: np.ndarray                     # Constant forcing
    D: np.ndarray                     # Dissipation operator
    nu_t: float                       # Eddy viscosity
    certificates: Dict[str, float]    # Verification certificates
    discovery_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "n_modes": self.n_modes,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "L": self.L.tolist(),
            "Q": self.Q.tolist(),
            "c": self.c.tolist(),
            "D": self.D.tolist(),
            "nu_t": self.nu_t,
            "certificates": self.certificates,
            "discovery_metadata": self.discovery_metadata,
            "nodes": [
                {
                    "kind": n.kind,
                    "operands": n.operands,
                    "result": n.result,
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PoemROM":
        """Deserialize from dict."""
        nodes = [
            PoemROMNode(
                kind=n["kind"],
                operands=n["operands"],
                result=n["result"],
                parameters={},
                metadata=n.get("metadata", {}),
            )
            for n in d.get("nodes", [])
        ]
        return cls(
            name=d["name"],
            n_modes=d["n_modes"],
            dt=d["dt"],
            n_steps=d["n_steps"],
            nodes=nodes,
            L=np.array(d["L"]),
            Q=np.array(d["Q"]),
            c=np.array(d["c"]),
            D=np.array(d["D"]),
            nu_t=d["nu_t"],
            certificates=d.get("certificates", {}),
            discovery_metadata=d.get("discovery_metadata", {}),
        )

    def execute(self, a0: np.ndarray) -> np.ndarray:
        """
        Execute the ROM Poem: integrate ȧ = f(a) forward in time.

        Parameters
        ----------
        a0 : Initial modal amplitudes (n_modes,)

        Returns
        -------
        trajectory : (n_steps+1, n_modes) array
        """
        trajectory = np.zeros((self.n_steps + 1, self.n_modes))
        trajectory[0] = a0.copy()
        a = a0.copy()

        L_eff = self.L + self.nu_t * self.D

        for step in range(self.n_steps):
            # RK4 integration
            k1 = self._rhs(a, L_eff)
            k2 = self._rhs(a + 0.5 * self.dt * k1, L_eff)
            k3 = self._rhs(a + 0.5 * self.dt * k2, L_eff)
            k4 = self._rhs(a + self.dt * k3, L_eff)
            a = a + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            if np.any(np.isnan(a)) or np.max(np.abs(a)) > 1e10:
                trajectory[step + 1:] = np.nan
                return trajectory

            trajectory[step + 1] = a

        return trajectory

    def _rhs(self, a: np.ndarray, L_eff: np.ndarray) -> np.ndarray:
        """RHS: ȧ = L_eff·a + Q(a,a) + c."""
        da = L_eff @ a + self.c
        for i in range(self.n_modes):
            da[i] += a @ self.Q[i] @ a
        return da

    def energy(self, a: np.ndarray) -> float:
        """Kinetic energy: E = ½ Σ aᵢ²."""
        return 0.5 * float(np.sum(a ** 2))

    def energy_spectrum(self, a: np.ndarray) -> np.ndarray:
        """Modal energy spectrum: E(k) = ½ aₖ²."""
        return 0.5 * a ** 2

    def to_symbolic(self) -> str:
        """
        Generate symbolic representation of the ROM equations.

        Returns a human-readable string of the governing equations.
        """
        lines = [f"# Poem ROM: {self.name}", f"# Modes: {self.n_modes}", ""]

        for i in range(self.n_modes):
            terms = []

            # Linear terms
            for j in range(self.n_modes):
                L_eff = self.L[i, j] + self.nu_t * self.D[i, j]
                if abs(L_eff) > 1e-12:
                    terms.append(f"{L_eff:+.6f}*a{j}")

            # Quadratic terms
            for j in range(self.n_modes):
                for k in range(j, self.n_modes):
                    q_val = self.Q[i, j, k]
                    if j != k:
                        q_val += self.Q[i, k, j]
                    if abs(q_val) > 1e-12:
                        if j == k:
                            terms.append(f"{q_val:+.6f}*a{j}^2")
                        else:
                            terms.append(f"{q_val:+.6f}*a{j}*a{k}")

            # Constant
            if abs(self.c[i]) > 1e-12:
                terms.append(f"{self.c[i]:+.6f}")

            eq = " ".join(terms) if terms else "0"
            lines.append(f"da{i}/dt = {eq}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ROM Generator
# ---------------------------------------------------------------------------

class ROMGenerator:
    """
    Generate executable Poem ROMs from discovered models.

    Pipeline:
    1. Accept ROMModel from rom_synthesizer
    2. Build Poem AST
    3. Attach certificates
    4. Return PoemROM ready for Gideon execution
    """

    def from_rom_model(
        self,
        rom_model: "ROMModel",
        name: str = "discovered_rom",
        dt: float = 0.01,
        n_steps: int = 1000,
    ) -> PoemROM:
        """
        Convert a ROMModel into an executable PoemROM.

        Parameters
        ----------
        rom_model : ROMModel from rom_synthesizer
        name : Name for this Poem
        dt : Integration time step
        n_steps : Number of integration steps
        """
        nodes = self._build_ast(rom_model, dt, n_steps)

        certificates = {}
        if rom_model.sindy_result is not None:
            certificates["sindy_sparsity"] = rom_model.sindy_result.sparsity
            certificates["sindy_residual"] = rom_model.sindy_result.residual_norm
            certificates["n_active_terms"] = float(rom_model.sindy_result.n_active_terms)
        certificates["is_dissipative"] = float(rom_model.is_dissipative())
        certificates["nu_t"] = rom_model.nu_t

        return PoemROM(
            name=name,
            n_modes=rom_model.n_modes,
            dt=dt,
            n_steps=n_steps,
            nodes=nodes,
            L=rom_model.L.copy(),
            Q=rom_model.Q.copy(),
            c=rom_model.c.copy(),
            D=rom_model.D.copy(),
            nu_t=rom_model.nu_t,
            certificates=certificates,
            discovery_metadata=rom_model.metadata.copy(),
        )

    def from_koopman_rom(
        self,
        L: np.ndarray,
        Q: np.ndarray,
        nu_t: float,
        wavenumbers: Optional[np.ndarray] = None,
        name: str = "koopman_rom",
        dt: float = 0.01,
        n_steps: int = 1000,
    ) -> PoemROM:
        """
        Build PoemROM directly from Koopman operators L, Q, ν_t.

        This is the primary interface for the Autopoietic Scientist:
          Poem.from_koopman_rom(L, Q, nu_t)
        """
        r = L.shape[0]
        c = np.zeros(r)

        if wavenumbers is not None:
            D = -np.diag(wavenumbers[:r] ** 2)
        else:
            D = -np.diag(np.arange(1, r + 1, dtype=float) ** 2)

        nodes = []
        nodes.append(PoemROMNode(
            kind="linear",
            operands=["a"],
            result="La",
            parameters={"L": L},
            metadata={"shape": list(L.shape)},
        ))
        nodes.append(PoemROMNode(
            kind="quadratic",
            operands=["a"],
            result="Qa",
            parameters={"Q": Q},
            metadata={"shape": list(Q.shape)},
        ))
        nodes.append(PoemROMNode(
            kind="closure",
            operands=["a"],
            result="Da",
            parameters={"D": D, "nu_t": nu_t},
            metadata={"method": "ergon_thermodynamic"},
        ))
        nodes.append(PoemROMNode(
            kind="rk4",
            operands=["La", "Qa", "Da", "c"],
            result="a_next",
            parameters={"dt": dt},
            metadata={"method": "rk4"},
        ))

        return PoemROM(
            name=name,
            n_modes=r,
            dt=dt,
            n_steps=n_steps,
            nodes=nodes,
            L=L,
            Q=Q,
            c=c,
            D=D,
            nu_t=nu_t,
            certificates={"is_dissipative": float(np.trace(L + nu_t * D) < 0)},
        )

    def _build_ast(self, rom_model: "ROMModel", dt: float, n_steps: int) -> List[PoemROMNode]:
        """Build the Poem AST from a ROMModel."""
        nodes = []

        # Node 1: Linear operation La
        nodes.append(PoemROMNode(
            kind="linear",
            operands=["a"],
            result="La",
            parameters={"L": rom_model.L},
            metadata={"shape": list(rom_model.L.shape)},
        ))

        # Node 2: Quadratic operation Q(a,a)
        nodes.append(PoemROMNode(
            kind="quadratic",
            operands=["a"],
            result="Qa",
            parameters={"Q": rom_model.Q},
            metadata={"shape": list(rom_model.Q.shape)},
        ))

        # Node 3: Closure ν_t·D·a
        nodes.append(PoemROMNode(
            kind="closure",
            operands=["a"],
            result="Da",
            parameters={"D": rom_model.D, "nu_t": rom_model.nu_t},
            metadata={"method": "thermodynamic"},
        ))

        # Node 4: Constant c
        nodes.append(PoemROMNode(
            kind="constant",
            operands=[],
            result="c",
            parameters={"c": rom_model.c},
        ))

        # Node 5: RK4 integration
        nodes.append(PoemROMNode(
            kind="rk4",
            operands=["La", "Qa", "Da", "c"],
            result="a_next",
            parameters={"dt": dt, "n_steps": n_steps},
            metadata={"method": "rk4"},
        ))

        # Node 6: Output
        nodes.append(PoemROMNode(
            kind="output",
            operands=["a_next"],
            result="trajectory",
            parameters={},
        ))

        return nodes
