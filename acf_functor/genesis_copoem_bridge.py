"""
Bridge between Genesis discoveries and CoPoem synthesis.

Converts mathematical discoveries from the Genesis engine into
CoPoem synthesis specifications, enabling the system to use
discovered mathematical relationships to guide matrix synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .core import ACFInvariant


def from_genesis_discovery(
    discovery: Any,
    dimension: int = 8,
    dtype: torch.dtype = torch.float64,
) -> Optional[Dict[str, Any]]:
    """
    Convierte un descubrimiento de Genesis en una especificación de síntesis para CoPoem.
    
    Por ejemplo, si Genesis descubre que f satisface f' = f (exponential),
    CoPoem puede sintetizar la matriz de transición correspondiente.
    
    Args:
        discovery: MathematicalDiscovery from Genesis engine
        dimension: Dimension of the matrix to synthesize
        dtype: Data type for the matrix
        
    Returns:
        Dictionary with synthesis specification, or None if discovery
        cannot be mapped to a synthesis spec.
    """
    from acf_functor.genesis import DiscoveryType, MathematicalDiscovery
    
    if not isinstance(discovery, MathematicalDiscovery):
        return None
    
    spec: Dict[str, Any] = {
        "dimension": dimension,
        "dtype": dtype,
        "discovery_id": discovery.discovery_id,
        "discovery_type": discovery.discovery_type.name,
        "persistence_score": discovery.persistence_score,
    }
    
    # Map discovery type to synthesis strategy
    if discovery.discovery_type == DiscoveryType.DIFFERENTIAL_RELATION:
        # f' = f → exponential dynamics → spectral radius near 1
        spec["spectral_radius"] = 1.0
        spec["symmetry"] = "orthogonal"
        spec["stability"] = "marginal"
        
    elif discovery.discovery_type == DiscoveryType.ALGEBRAIC_IDENTITY:
        # sin² + cos² = 1 → bounded system → spectral radius < 1
        spec["spectral_radius"] = 0.95
        spec["symmetry"] = "symmetric"
        spec["stability"] = "stable"
        
    elif discovery.discovery_type == DiscoveryType.FIXED_POINT:
        # f(x*) = x* → eigenvalue 1
        spec["spectral_radius"] = 1.0
        spec["symmetry"] = None
        spec["stability"] = "marginal"
        
    elif discovery.discovery_type == DiscoveryType.SYMMETRY:
        # Symmetry detected → use symmetric matrix
        spec["spectral_radius"] = 0.9
        spec["symmetry"] = "symmetric"
        spec["stability"] = "stable"
        
    elif discovery.discovery_type == DiscoveryType.FUNCTIONAL_EQUATION:
        # Functional equation → use persistence score to guide spectral radius
        spec["spectral_radius"] = max(0.5, 1.0 - discovery.persistence_score * 0.3)
        spec["symmetry"] = "orthogonal"
        spec["stability"] = "stable"
        
    else:
        # Unknown type → conservative synthesis
        spec["spectral_radius"] = 0.8
        spec["symmetry"] = None
        spec["stability"] = "stable"
    
    # Use numerical evidence to refine spectral radius
    if discovery.numerical_evidence:
        max_error = discovery.numerical_evidence.get("max_error", 1.0)
        # Lower error → higher confidence → spectral radius closer to target
        confidence = max(0.0, 1.0 - max_error)
        spec["confidence"] = confidence
    
    return spec


def apply_genesis_spec(copoem, discovery: Any, dimension: int = 8) -> Tuple[torch.Tensor, Any]:
    """
    Synthesize a matrix using a Genesis discovery as guidance.
    
    Args:
        copoem: CoPoem instance
        discovery: MathematicalDiscovery from Genesis engine
        dimension: Dimension of the matrix to synthesize
        
    Returns:
        Tuple of (synthesized matrix, synthesis report)
    """
    spec_dict = from_genesis_discovery(discovery, dimension=dimension)
    if spec_dict is None:
        # Fallback to basic spectral synthesis
        spec = copoem.spectrum(spectral_radius=0.9, dimension=dimension)
        W = copoem.synthesize(spec)
        return W, None
    
    # Build multi-objective spec
    multi_spec = (copoem.multi_objective()
                  .spectrum(
                      spectral_radius=spec_dict.get("spectral_radius", 0.9),
                      dimension=dimension,
                      symmetry=spec_dict.get("symmetry"),
                  ))
    
    if spec_dict.get("stability") == "stable":
        multi_spec = multi_spec.stability(lyapunov_exponent=-0.1, dimension=dimension)
    
    W, report = copoem.synthesize_multi(multi_spec)
    return W, report
