"""
acf_neuron_network.py — Red de Neuronas ACF
===========================================

Una red de neuronas ACF no es un grafo de pesos. Es un SISTEMA
DINÁMICO sobre una variedad funcional. Cada neurona es un punto en
el espacio de funciones, y las conexiones son morfismos.

Propiedades:
  - Forward con trazabilidad total (cada operación es atribuible)
  - Aprendizaje online distribuido (sin backprop global)
  - Catálogo de conocimiento compartido entre todas las neuronas
  - Topología de composición funcional (DAG)

Autor: AXIOM-1 sobre fundamentos de Eddy Manuel Piantini
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .acf_neuron import (
    ACFNeuron,
    FunctionCatalog,
    FunctionEntry,
    FMAInstruction,
)
from .acf_neuron_learning import ACFLearningEngine
from .acf_synapse import ACFSynapse


@dataclass
class ForwardStep:
    """Paso individual en forward pass con trazabilidad."""
    neuron_id: str
    neuron_name: str
    input_val: float
    output_val: float
    energy_consumed: int
    epsilon: float
    fma_chain_len: int


@dataclass
class ForwardTrace:
    """Trazabilidad completa de un forward pass."""
    steps: List[ForwardStep] = field(default_factory=list)
    total_energy: int = 0
    total_epsilon: float = 0.0

    def add(self, step: ForwardStep) -> None:
        self.steps.append(step)
        self.total_energy += step.energy_consumed
        self.total_epsilon = max(self.total_epsilon, step.epsilon)

    def summary(self) -> str:
        lines = ["Forward Trace:"]
        for s in self.steps:
            lines.append(f"  {s.neuron_name}: in={s.input_val:.4f} → out={s.output_val:.4f} "
                         f"(E={s.energy_consumed}, ε={s.epsilon:.2e})")
        lines.append(f"  Total: E={self.total_energy}, ε={self.total_epsilon:.2e}")
        return "\n".join(lines)


class ACFNeuralNetwork:
    """
    Red de Neuronas ACF.

    Una red donde cada neurona es una función certificada y las
    conexiones son composiciones funcionales. El aprendizaje es
    distribuido: cada neurona aprende de su propio error local.
    """

    def __init__(self, name: str = "acf_network"):
        self.name = name
        self.neurons: Dict[str, ACFNeuron] = {}
        self.synapses: List[ACFSynapse] = []
        self.catalog = FunctionCatalog()
        self.topology_order: List[str] = []  # orden topológico para forward
        self.input_neuron_id: Optional[str] = None
        self.output_neuron_id: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════
    # CONSTRUCCIÓN
    # ═══════════════════════════════════════════════════════════════

    def add_neuron(self, neuron: ACFNeuron) -> ACFNeuron:
        """Añadir neurona a la red. Hereda acceso al catálogo compartido."""
        neuron.catalog = self.catalog
        self.neurons[neuron.neuron_id] = neuron
        return neuron

    def create_neuron(self, name: str, **kwargs) -> ACFNeuron:
        """Crear y añadir una neurona a la red."""
        neuron = ACFNeuron(name=name, catalog=self.catalog, **kwargs)
        self.neurons[neuron.neuron_id] = neuron
        return neuron

    def connect(self, source_id: str, target_id: str) -> ACFSynapse:
        """Conectar dos neuronas vía sinapsis ACF."""
        if source_id not in self.neurons:
            raise KeyError(f"Neurona fuente '{source_id}' no encontrada")
        if target_id not in self.neurons:
            raise KeyError(f"Neurona destino '{target_id}' no encontrada")

        synapse = ACFSynapse(
            source=self.neurons[source_id],
            target=self.neurons[target_id],
        )
        self.synapses.append(synapse)
        return synapse

    def set_input(self, neuron_id: str) -> None:
        self.input_neuron_id = neuron_id

    def set_output(self, neuron_id: str) -> None:
        self.output_neuron_id = neuron_id

    def build_topology(self) -> None:
        """
        Construir orden topológico de neuronas para forward pass.
        Ordenamiento por Kahn (BFS) del DAG de sinapsis.
        """
        # Calcular grados de entrada
        in_degree: Dict[str, int] = {nid: 0 for nid in self.neurons}
        adj: Dict[str, List[str]] = {nid: [] for nid in self.neurons}

        for syn in self.synapses:
            src = syn.source.neuron_id
            tgt = syn.target.neuron_id
            if tgt not in in_degree:
                in_degree[tgt] = 0
            in_degree[tgt] += 1
            if src not in adj:
                adj[src] = []
            adj[src].append(tgt)

        # BFS de Kahn
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        self.topology_order = []

        while queue:
            node = queue.pop(0)
            self.topology_order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    # ═══════════════════════════════════════════════════════════════
    # FORWARD
    # ═══════════════════════════════════════════════════════════════

    def forward(self, x: float) -> Tuple[float, ForwardTrace]:
        """
        Propagación hacia adelante con trazabilidad total.

        Cada operación es atribuible a una neurona específica.
        """
        if not self.topology_order:
            self.build_topology()

        trace = ForwardTrace()
        activations: Dict[str, float] = {}

        for neuron_id in self.topology_order:
            neuron = self.neurons[neuron_id]

            # Determinar entrada: de conexiones entrantes o input directo
            incoming = [
                sid for sid, syn in neuron.incoming.items()
                if sid in activations
            ]

            if not incoming:
                # Neurona de entrada
                if neuron_id == self.input_neuron_id or self.input_neuron_id is None:
                    current_input = x
                else:
                    continue  # sin entradas, saltar
            elif len(incoming) == 1:
                current_input = activations[incoming[0]]
            else:
                # Múltiples entradas: promedio ponderado por energía
                total_e = sum(self.neurons[sid].energy for sid in incoming)
                current_input = sum(
                    activations[sid] * self.neurons[sid].energy / max(total_e, 1)
                    for sid in incoming
                )

            # Evaluar neurona
            output = float(neuron.evaluate(current_input))
            activations[neuron_id] = output

            trace.add(ForwardStep(
                neuron_id=neuron_id,
                neuron_name=neuron.name,
                input_val=current_input,
                output_val=output,
                energy_consumed=neuron.energy,
                epsilon=neuron.epsilon,
                fma_chain_len=len(neuron.fma_chain),
            ))

        # Salida final
        if self.output_neuron_id and self.output_neuron_id in activations:
            final_output = activations[self.output_neuron_id]
        else:
            final_output = activations.get(self.topology_order[-1], x) if self.topology_order else x

        return final_output, trace

    def __call__(self, x: float) -> float:
        out, _ = self.forward(x)
        return out

    # ═══════════════════════════════════════════════════════════════
    # APRENDIZAJE DISTRIBUIDO
    # ═══════════════════════════════════════════════════════════════

    def learn_online(self, x: float, y_target: float) -> Dict[str, float]:
        """
        Aprendizaje online distribuido.

        NO hay backpropagation global. Cada neurona es responsable
        de aprender de su propio error local.

        El error se distribuye proporcionalmente a la energía de cada neurona.
        """
        # Forward con trazabilidad
        y_pred, trace = self.forward(x)
        global_error = y_target - y_pred

        # Distribuir error a cada neurona
        errors = {}
        for step in trace.steps:
            neuron = self.neurons[step.neuron_id]
            if neuron._learning_enabled and neuron.learning_engine is not None:
                # Error proporcional a la contribución energética
                if trace.total_energy > 0:
                    local_weight = step.energy_consumed / trace.total_energy
                else:
                    local_weight = 1.0 / max(len(trace.steps), 1)

                local_error = global_error * local_weight
                local_target = step.output_val + local_error

                evo_step = neuron.learning_engine.observe_and_learn(
                    x=step.input_val,
                    y_target=local_target,
                )
                errors[neuron.name] = evo_step.error
            else:
                errors[neuron.name] = 0.0

        return errors

    # ═══════════════════════════════════════════════════════════════
    # CONOCIMIENTO COMPARTIDO
    # ═══════════════════════════════════════════════════════════════

    def share_all_knowledge(self) -> int:
        """Publicar todas las neuronas en el catálogo compartido."""
        count = 0
        for neuron in self.neurons.values():
            if neuron.fma_chain:
                neuron.publish_to_catalog()
                count += 1
        return count

    def share_knowledge_between(self, source_name: str, target_name: str) -> bool:
        """
        Compartir conocimiento entre dos neuronas por composición.

        Encuentra las neuronas por nombre y transmite el conocimiento
        de fuente a destino vía sinapsis temporal.
        """
        source = None
        target = None
        for n in self.neurons.values():
            if n.name == source_name:
                source = n
            if n.name == target_name:
                target = n

        if source is None or target is None:
            return False

        synapse = ACFSynapse(source=source, target=target)
        synapse.transmit_knowledge()
        self.synapses.append(synapse)
        return True

    # ═══════════════════════════════════════════════════════════════
    # DIAGNÓSTICO
    # ═══════════════════════════════════════════════════════════════

    def summary(self) -> str:
        lines = [f"ACFNeuralNetwork('{self.name}'):"]
        lines.append(f"  Neuronas: {len(self.neurons)}")
        lines.append(f"  Sinapsis: {len(self.synapses)}")
        lines.append(f"  Catálogo: {self.catalog.size()} funciones")
        lines.append(f"  Topología: {'→'.join(self.topology_order) if self.topology_order else 'no construida'}")

        total_energy = sum(n.energy for n in self.neurons.values())
        total_obs = sum(n.total_observations for n in self.neurons.values())
        lines.append(f"  Energía total: {total_energy} FMAs")
        lines.append(f"  Observaciones totales: {total_obs}")

        lines.append("\n  Neuronas:")
        for n in self.neurons.values():
            lines.append(f"    {n.name}: {n.functional_form.value}, E={n.energy}, "
                         f"ε={n.epsilon:.2e}, α_A={n.alpha_A:.3f}, obs={n.total_observations}")
        return "\n".join(lines)

    def get_network_fingerprint(self) -> Dict[str, Any]:
        """Huella digital completa de la red."""
        return {
            "name": self.name,
            "n_neurons": len(self.neurons),
            "n_synapses": len(self.synapses),
            "total_energy": sum(n.energy for n in self.neurons.values()),
            "total_observations": sum(n.total_observations for n in self.neurons.values()),
            "max_epsilon": max((n.epsilon for n in self.neurons.values()), default=float("inf")),
            "neurons": {
                n.name: n.snapshot() for n in self.neurons.values()
            },
            "catalog_size": self.catalog.size(),
        }
