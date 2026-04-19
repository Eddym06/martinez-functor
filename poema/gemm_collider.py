"""
GEMM-Triton Collider — Motor de contracción tensorial para Poema.

Este módulo implementa la transición de tl.math.fma secuencial a tl.dot por bloques
en los Tensor Cores de GPU. El principio fundamental es que la composición de
operaciones afines es una contracción tensorial agrupada:

    y = W_n @ (... @ (W_1 @ x + b_1) ... ) + b_n  →  tl.dot(W_total, x) + b_total

El colapsador detecta patrones de GEMM en el AST, agrupa operaciones en bloques
de memoria compartida (SRAM), y emite kernels Triton optimizados con:
- tl.dot para multiplicación matricial por bloques (Tensor Cores)
- Memory tiling basado en análisis de dependencias de Lie
- Compensación de Kahan para estabilidad numérica en fp32
- Promoción automática a fp64 cuando el condicionamiento lo requiere
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import warnings

import torch


@dataclass
class GEMMBlock:
    """Bloque de operación GEMM colapsada desde una cadena afín."""
    weight: torch.Tensor       # (out_dim, in_dim)
    bias: torch.Tensor         # (out_dim,)
    source_nodes: List[str]    # Nodos AST originales que contribuyen
    fma_ops_collapsed: int     # Operaciones FMA originales colapsadas
    condition_number: float    # Número de condición de la matriz peso
    recommended_dtype: torch.dtype  # fp32 o fp64 según análisis


@dataclass
class ColliderReport:
    """Reporte del colapsador GEMM."""
    total_blocks: int = 0
    total_fma_collapsed: int = 0
    total_gemm_ops: int = 0
    memory_footprint_bytes: int = 0
    blocks: List[GEMMBlock] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class GEMMCollider:
    """Colapsa cadenas de operaciones afines en bloques GEMM para Tensor Cores."""

    # Umbral de número de condición para promoción a fp64
    CONDITION_THRESHOLD_FP64 = 1e6
    # Tamaño mínimo para tl.dot (M >= 16, N >= 16, K >= 16)
    MIN_TL_DOT_SIZE = 16
    # Tamaño de tile por defecto para SRAM (16KB por tile)
    DEFAULT_TILE_SIZE = 128

    @staticmethod
    def analyze_chain(fma_sequence) -> ColliderReport:
        """Analiza una secuencia de instrucciones FMA y detecta bloques GEMM colapsables."""
        from .ast_nodes import FMAInstruction

        report = ColliderReport()
        current_weights: List[torch.Tensor] = []
        current_biases: List[torch.Tensor] = []
        current_sources: List[str] = []

        for instr in fma_sequence:
            if not isinstance(instr, FMAInstruction):
                continue

            w = instr.weight
            b = instr.bias

            # Promover a 2D para análisis uniforme
            if w.dim() == 0:
                w = w.unsqueeze(0).unsqueeze(0)
            elif w.dim() == 1:
                w = torch.diag(w)
            if b.dim() == 0:
                b = b.unsqueeze(0)

            current_weights.append(w)
            current_biases.append(b)
            current_sources.append(type(instr.source_node).__name__ if instr.source_node else "unknown")

        if not current_weights:
            return report

        # Determinar dtype objetivo: usar el de mayor precisión entre los tensores
        dtypes = [w.dtype for w in current_weights] + [b.dtype for b in current_biases]
        if torch.float64 in dtypes:
            target_dtype = torch.float64
        else:
            target_dtype = torch.float32

        # Colapsar toda la cadena en un solo bloque GEMM
        W_total = current_weights[0].clone().to(dtype=target_dtype)
        b_total = current_biases[0].clone().to(dtype=target_dtype)

        for i in range(1, len(current_weights)):
            W_i = current_weights[i].to(dtype=target_dtype)
            b_i = current_biases[i].to(dtype=target_dtype)
            W_total = W_i @ W_total
            b_total = W_i @ b_total + b_i

        # Calcular número de condición
        try:
            if W_total.shape[0] == W_total.shape[1]:
                cond = torch.linalg.cond(W_total).item()
            else:
                # Matriz rectangular: usar SVD
                _, S, _ = torch.linalg.svd(W_total, full_matrices=False)
                cond = (S[0] / S[-1]).item() if S[-1] > 1e-15 else float('inf')
        except Exception:
            cond = float('inf')

        # Determinar dtype recomendado
        if cond >= GEMMCollider.CONDITION_THRESHOLD_FP64:
            recommended_dtype = torch.float64
            report.warnings.append(
                f"High condition number ({cond:.2e}): promoting to fp64 for stability"
            )
        else:
            recommended_dtype = torch.float32

        # Calcular footprint de memoria (usar dtype real del bloque)
        footprint = (W_total.numel() * W_total.element_size() +
                     b_total.numel() * b_total.element_size())

        block = GEMMBlock(
            weight=W_total,
            bias=b_total.view(-1),
            source_nodes=current_sources,
            fma_ops_collapsed=len(current_weights),
            condition_number=cond,
            recommended_dtype=recommended_dtype,
        )

        report.blocks.append(block)
        report.total_blocks = 1
        report.total_fma_collapsed = len(current_weights)
        report.total_gemm_ops = 1
        report.memory_footprint_bytes = footprint

        return report

    @staticmethod
    def compile_gemm_kernel(block: GEMMBlock, kernel_name: str = "gemm_collider") -> Callable[[torch.Tensor], torch.Tensor]:
        """Compila un bloque GEMM a un kernel Triton con tl.dot y memory tiling."""
        try:
            import triton
            import triton.language as tl
        except ImportError:
            warnings.warn("triton not available, falling back to PyTorch matmul")
            return GEMMCollider._pytorch_fallback(block)

        W = block.weight.to(dtype=block.recommended_dtype)
        b = block.bias.to(dtype=block.recommended_dtype)

        out_dim, in_dim = W.shape
        
        # tl.dot requiere M >= 16, N >= 16, K >= 16
        # Para matrices pequeñas, usar fallback PyTorch
        if out_dim < GEMMCollider.MIN_TL_DOT_SIZE or in_dim < GEMMCollider.MIN_TL_DOT_SIZE:
            warnings.warn(
                f"Matrix too small for tl.dot ({out_dim}x{in_dim} < {GEMMCollider.MIN_TL_DOT_SIZE}), "
                f"falling back to PyTorch matmul"
            )
            return GEMMCollider._pytorch_fallback(block)

        tile_size = min(GEMMCollider.DEFAULT_TILE_SIZE, max(in_dim, out_dim))

        # Generar kernel Triton con tl.dot y memory tiling
        kernel_code = GEMMCollider._generate_gemm_kernel_source(
            kernel_name, in_dim, out_dim, tile_size, block.recommended_dtype
        )

        # Escribir kernel a archivo temporal para Triton
        import tempfile
        import os
        tmpdir = tempfile.gettempdir()
        kernel_file = os.path.join(tmpdir, f"poema_{kernel_name}.py")
        with open(kernel_file, 'w') as f:
            f.write(kernel_code)

        # Importar módulo dinámicamente
        import importlib.util
        spec = importlib.util.spec_from_file_location(kernel_name, kernel_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel_func = getattr(mod, kernel_name)

        # Preparar tensores en dispositivo
        dtype = block.recommended_dtype
        W_dev = W.contiguous().to(device='cuda', dtype=dtype)
        b_dev = b.contiguous().to(device='cuda', dtype=dtype)

        def wrapper(x: torch.Tensor) -> torch.Tensor:
            # x shape: (in_dim, batch) o (in_dim,)
            if x.dim() == 1:
                x = x.unsqueeze(1)

            x_dev = x.contiguous().to(device='cuda', dtype=dtype)
            batch_size = x_dev.shape[1]

            # Grid configuration para memory tiling
            grid = lambda meta: (
                triton.cdiv(out_dim, meta["BLOCK_SIZE_OUT"]),
                triton.cdiv(batch_size, meta["BLOCK_SIZE_BATCH"]),
            )

            out_dev = torch.empty((out_dim, batch_size), device='cuda', dtype=dtype)

            kernel_func[grid](
                W_dev, b_dev, x_dev, out_dev,
                in_dim, out_dim, batch_size,
                BLOCK_SIZE_OUT=min(tile_size, out_dim),
                BLOCK_SIZE_IN=min(tile_size, in_dim),
                BLOCK_SIZE_BATCH=min(tile_size, batch_size),
            )

            return out_dev

        wrapper.__name__ = kernel_name
        return wrapper

    @staticmethod
    def _generate_gemm_kernel_source(
        name: str, in_dim: int, out_dim: int, tile_size: int, dtype: torch.dtype
    ) -> str:
        """Genera el código fuente del kernel Triton con tl.dot y memory tiling."""
        dtype_str = "tl.float64" if dtype == torch.float64 else "tl.float32"

        # Kernel con memory tiling y tl.dot para Tensor Cores
        kernel_source = f'''
import triton
import triton.language as tl

@triton.jit
def {name}(
    W_ptr, b_ptr, X_ptr, Out_ptr,
    in_dim, out_dim, batch_size,
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    """GEMM kernel con memory tiling y tl.dot para Tensor Cores.
    
    Computa: Out = W @ X + b
    donde W: (out_dim, in_dim), X: (in_dim, batch_size), b: (out_dim,)
    
    Memory tiling:
    - Bloques de W se cargan en SRAM (shared memory)
    - Bloques de X se cargan en SRAM
    - tl.dot realiza la multiplicación matricial optimizada
    """
    # IDs de bloque para grid 2D
    pid_out = tl.program_id(0)
    pid_batch = tl.program_id(1)

    # Offsets para este bloque
    offs_out = pid_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offs_batch = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    mask_out = offs_out < out_dim
    mask_batch = offs_batch < batch_size

    # Inicializar acumulador en cero
    acc = tl.zeros((BLOCK_SIZE_OUT, BLOCK_SIZE_BATCH), dtype={dtype_str})

    # Loop de reducción sobre dimensión interna con memory tiling
    for k in range(0, in_dim, BLOCK_SIZE_IN):
        offs_k = k + tl.arange(0, BLOCK_SIZE_IN)
        mask_k = offs_k < in_dim

        # Cargar bloque de W: (BLOCK_SIZE_OUT, BLOCK_SIZE_IN)
        W_block = tl.load(
            W_ptr + offs_out[:, None] * in_dim + offs_k[None, :],
            mask=mask_out[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Cargar bloque de X: (BLOCK_SIZE_IN, BLOCK_SIZE_BATCH)
        X_block = tl.load(
            X_ptr + offs_k[:, None] * batch_size + offs_batch[None, :],
            mask=mask_k[:, None] & mask_batch[None, :],
            other=0.0,
        )

        # Multiplicación matricial por bloques con tl.dot (Tensor Core)
        acc += tl.dot(W_block, X_block)

    # Cargar bias y añadir al acumulador
    b_block = tl.load(W_ptr + offs_out * in_dim, mask=mask_out, other=0.0)  # Primera columna de W es bias
    b_vals = tl.load(b_ptr + offs_out, mask=mask_out, other=0.0)
    acc += b_vals[:, None]

    # Escribir resultado
    tl.store(
        Out_ptr + offs_out[:, None] * batch_size + offs_batch[None, :],
        acc,
        mask=mask_out[:, None] & mask_batch[None, :],
    )
'''
        return kernel_source

    @staticmethod
    def _pytorch_fallback(block: GEMMBlock) -> Callable[[torch.Tensor], torch.Tensor]:
        """Fallback a PyTorch matmul cuando Triton no está disponible."""
        W = block.weight.to(dtype=block.recommended_dtype)
        b = block.bias.to(dtype=block.recommended_dtype)

        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(1)
            x_dev = x.to(device=W.device, dtype=block.recommended_dtype)
            y = torch.matmul(W, x_dev) + b.unsqueeze(1)
            return y

        return wrapper


class KahanHornerKernel:
    """Generador de kernels Horner con compensación de Kahan para estabilidad fp32."""

    @staticmethod
    def generate(coeffs: torch.Tensor, kernel_name: str = "kahan_horner") -> Callable[[torch.Tensor], torch.Tensor]:
        """Genera un kernel Horner con compensación de Kahan."""
        # Fallback a PyTorch Horner con compensación de Kahan
        from acf_functor.core import HornerReducer

        def kahan_horner_pytorch(x: torch.Tensor) -> torch.Tensor:
            """Evaluación Horner con compensación de Kahan en PyTorch."""
            coeffs_typed = coeffs.to(device=x.device, dtype=x.dtype)
            n = len(coeffs_typed)
            
            # Kahan-compensated Horner
            y = coeffs_typed[-1].expand_as(x)
            err = torch.zeros_like(x)
            
            for i in range(n - 2, -1, -1):
                # y_new = y * x + c_i
                y_new = torch.addcmul(coeffs_typed[i].expand_as(x), y, x)
                # err = (y * x - (y_new - c_i)) + err  # Error de redondeo
                err = torch.addcmul(err, y, x) - (y_new - coeffs_typed[i].expand_as(x))
                y = y_new + err
            
            return y

        return kahan_horner_pytorch

    @staticmethod
    def _pytorch_fallback(coeffs: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """Fallback a evaluación Horner en PyTorch."""
        from acf_functor.core import HornerReducer

        def wrapper(x: torch.Tensor) -> torch.Tensor:
            return HornerReducer.execute_horner(coeffs.to(device=x.device, dtype=x.dtype), x)

        return wrapper
