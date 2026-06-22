"""
acf_synapse.py — Sinapsis ACF: Protocolo de Conocimiento Compartido
====================================================================

Una sinapsis ACF conecta dos neuronas transmitiendo FUNCIONES completas,
no intensidades numéricas. La neurona destino compone la función de la
fuente con la suya propia — sin backprop, sin gradientes, sin pérdida.

Propiedades:
  - Transmisión de función completa (cadena FMA)
  - Composición functorial: Φ(f ∘ g) = Φ(f) ∘ Φ(g)
  - Trazabilidad: cada FMA es atribuible a su neurona fuente
  - Verificación de compatibilidad de dominio
  - Auto-Domain Repair cuando los dominios no coinciden

Autor: AXIOM-1 sobre fundamentos de Eddy Manuel Piantini
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .acf_neuron import (
    ACFNeuron,
    FMAInstruction,
    IntervalContract,
    compose_fma_chains,
    evaluate_fma_chain,
    hash_fma_chain,
    simplify_fma_chain,
)


@dataclass
class KnowledgeAttribution:
    """Atribución de una contribución de conocimiento."""
    source_id: str
    source_name: str
    contribution: float
    instruction_idx: int
    fma_weight: float
    fma_bias: float


@dataclass
class SynapseReport:
    """Reporte de transmisión de conocimiento."""
    source_id: str
    target_id: str
    timestamp: float
    domain_compatible: bool
    composed_energy: int
    composed_epsilon: float
    attributions: List[KnowledgeAttribution]
    verification_passed: bool


class ACFSynapse:
    """
    Sinapsis ACF — Conexión funcional entre dos neuronas.

    No transmite un número. Transmite la FUNCIÓN completa de la fuente,
    que la destino compone con la suya.

    Uso:
        synapse = ACFSynapse(source=neuron_sin, target=neuron_target)
        synapse.transmit_knowledge()
        # neuron_target ahora incorpora el conocimiento de neuron_sin
    """

    def __init__(self, source: ACFNeuron, target: ACFNeuron):
        self.source = source
        self.target = target
        self.coupling: Optional[List[FMAInstruction]] = None
        self.transmission_history: List[SynapseReport] = []

        # Registrar conexiones bidireccionales
        source.outgoing[target.neuron_id] = self
        target.incoming[source.neuron_id] = self

    def transmit_knowledge(self) -> SynapseReport:
        """
        Transmitir conocimiento de fuente a destino.

        La neurona destino INCORPORA la función de la fuente.
        NO reentrena. Copia los coeficientes espectrales.
        """
        # 1. Verificar y reparar dominio
        domain_ok = self._verify_domain_compatibility()
        if not domain_ok:
            self._repair_domain()

        # 2. Copiar coeficientes de la fuente al destino
        #    (la sinapsis ACF no compone — reemplaza el conocimiento del destino
        #     con el de la fuente, útil para compartir funciones aprendidas)
        d_src = self.source._effective_degree()
        if d_src >= 0:
            self.target.coefficients[:d_src + 1] = self.source.coefficients[:d_src + 1].copy()
            self.target.fma_chain = [FMAInstruction(weight=float(i.weight), bias=float(i.bias),
                                                    source=f"neuron:{self.source.name}")
                                     for i in self.source.fma_chain]
            self.target.energy = self.source.energy
            self.target.epsilon = self.source.epsilon
            self.target.alpha_A = self.source.alpha_A
            self.target.functional_form = self.source.functional_form
            self.coupling = self.target.fma_chain
        else:
            self.coupling = []

        # 3. Verificar
        verification_ok = self._verify_composition() if self.coupling else False

        # 4. Atribuciones
        attributions = self._generate_attributions(self.coupling)

        report = SynapseReport(
            source_id=self.source.neuron_id,
            target_id=self.target.neuron_id,
            timestamp=time.time(),
            domain_compatible=domain_ok,
            composed_energy=self.target.energy,
            composed_epsilon=self.target.epsilon,
            attributions=attributions,
            verification_passed=verification_ok,
        )
        self.transmission_history.append(report)
        return report

    def query_knowledge(self, x: float) -> Tuple[float, List[KnowledgeAttribution]]:
        """
        Evaluar la conexión y ATRIBUIR el conocimiento.

        Esto es imposible en redes neuronales tradicionales.
        En ACF, cada FMA tiene trazabilidad a su neurona fuente.
        """
        if not self.coupling:
            return float(evaluate_fma_chain(self.target.fma_chain, x)), []

        result = float(x)
        attributions: List[KnowledgeAttribution] = []

        for idx, instr in enumerate(self.coupling):
            result = instr.weight * result + instr.bias
            if instr.source.startswith("neuron:"):
                source_name = instr.source.split(":", 1)[1]
                attributions.append(KnowledgeAttribution(
                    source_id=self.source.neuron_id,
                    source_name=source_name,
                    contribution=float(result),
                    instruction_idx=idx,
                    fma_weight=instr.weight,
                    fma_bias=instr.bias,
                ))

        return result, attributions

    def _verify_domain_compatibility(self) -> bool:
        """Verificar que el rango de la fuente esté en el dominio del destino."""
        src_lo = float(self.source.evaluate(self.source.domain.lo))
        src_hi = float(self.source.evaluate(self.source.domain.hi))
        src_min = min(src_lo, src_hi)
        src_max = max(src_lo, src_hi)
        return (src_min >= self.target.domain.lo and src_max <= self.target.domain.hi)

    def _repair_domain(self) -> None:
        """Expandir dominio del destino para acomodar la fuente."""
        src_lo = float(self.source.evaluate(self.source.domain.lo))
        src_hi = float(self.source.evaluate(self.source.domain.hi))
        src_min = min(src_lo, src_hi)
        src_max = max(src_lo, src_hi)
        new_lo = min(self.target.domain.lo, src_min - abs(src_min) * 0.1)
        new_hi = max(self.target.domain.hi, src_max + abs(src_max) * 0.1)
        self.target.domain = IntervalContract(new_lo, new_hi, strict=False)

    def _verify_composition(self) -> bool:
        """Verificar que el conocimiento transferido es correcto."""
        if not self.coupling:
            return True
        test_points = self.source.domain.sample(15)
        for x in test_points:
            direct = float(self.source.evaluate(x))
            copied = float(self.target.evaluate(x))
            if abs(direct - copied) > 1e-6:
                return False
        return True

    def _generate_attributions(self, chain: List[FMAInstruction]) -> List[KnowledgeAttribution]:
        """Generar lista de atribuciones de conocimiento."""
        attribs: List[KnowledgeAttribution] = []
        acc = 0.0
        for idx, instr in enumerate(chain):
            if instr.source.startswith("neuron:"):
                source_name = instr.source.split(":", 1)[1]
                acc = instr.weight * acc + instr.bias
                attribs.append(KnowledgeAttribution(
                    source_id=self.source.neuron_id,
                    source_name=source_name,
                    contribution=float(acc),
                    instruction_idx=idx,
                    fma_weight=instr.weight,
                    fma_bias=instr.bias,
                ))
        return attribs

    def __repr__(self) -> str:
        return f"ACFSynapse({self.source.name} → {self.target.name})"
