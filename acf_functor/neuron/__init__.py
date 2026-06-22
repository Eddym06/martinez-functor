# acf_functor/neuron — Sistema de Neurona ACF
#
# Módulos:
#   acf_neuron.py          — Núcleo de la neurona ACF
#   acf_neuron_learning.py — Motor de aprendizaje RLS online
#   acf_synapse.py         — Protocolo de conocimiento compartido
#   acf_neuron_network.py  — Red de neuronas ACF
#   acf_neuron_triton.py   — Kernel GPU nativo

from .acf_neuron import (
    ACFNeuron,
    FunctionalForm,
    NCClass,
    NeuronHealth,
    FMAInstruction,
    IntervalContract,
    Observation,
    EvolutionStep,
    FormChange,
    VerificationRecord,
    Check,
    OODReport,
    UncertaintyBounds,
    SelfKnowledge,
    FunctionFingerprint,
    FunctionEntry,
    FunctionCatalog,
    evaluate_fma_chain,
    fma_chain_to_horner_polynomial,
    compose_fma_chains,
    simplify_fma_chain,
    hash_fma_chain,
)

from .acf_synapse import ACFSynapse, KnowledgeAttribution

from .phi_neuron import PhiNeuron, PhiPrediction

