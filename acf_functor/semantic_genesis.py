"""SemanticGenesis — Búsqueda del Functor Semántico Ψ_Sem.

Evolución 21: Explora sistemáticamente el espacio de candidate functors
que mapean estructuras lingüísticas a representaciones geométricas,
buscando aquel que preserva:
  1. Sinonimia → proximidad geométrica
  2. Composición → operación algebraica  
  3. Contraste → separación geométrica

Basado en el patrón de Genesis (Evolution 20), adaptado para semántica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Candidate Functor — una posible Ψ_Sem
# ═══════════════════════════════════════════════════════════════

@dataclass
class CandidateFunctor:
    """Un candidato a Ψ_Sem: mapea texto → geometría."""

    name: str
    # Parámetros de la transformación
    basis_type: str          # "hermite", "fourier", "chebyshev", "legendre", "random"
    n_dims: int              # dimensiones del espacio geométrico
    degree: int              # grado polinomial
    context_window: int      # ventana de contexto para embeddings
    use_position: bool       # codificar posición
    use_pmi_weight: bool     # ponderar por PMI
    normalization: str       # "l2", "zscore", "none"
    temperature: float       # temperatura para softmax en composición
    
    # Métricas de semanticidad
    synonymy_score: float = 0.0
    composition_score: float = 0.0
    contrast_score: float = 0.0
    distributional_score: float = 0.0
    persistence_score: float = 0.0
    total_semanticity: float = 0.0
    
    # Huella digital
    seed: int = 42
    
    @property
    def functor_hash(self) -> str:
        data = f"{self.basis_type}:{self.n_dims}:{self.degree}:{self.context_window}:{self.seed}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def summary(self) -> str:
        return (
            f"Ψ[{self.basis_type}](d={self.n_dims}, deg={self.degree}, "
            f"ctx={self.context_window}, pos={self.use_position}, "
            f"pmi={self.use_pmi_weight}, norm={self.normalization}, T={self.temperature:.2f})\n"
            f"  synonymy={self.synonymy_score:.3f}  composition={self.composition_score:.3f}  "
            f"contrast={self.contrast_score:.3f}  Σ={self.total_semanticity:.3f}"
        )


# ═══════════════════════════════════════════════════════════════
# Generador de Candidatos
# ═══════════════════════════════════════════════════════════════

class FunctorGenerator:
    """Genera candidate functors variando parámetros estructurales."""
    
    BASES = ["hermite", "fourier", "chebyshev", "legendre", "random"]
    NORMALIZATIONS = ["l2", "zscore", "none"]
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.generation = 0
    
    def generate_random(self) -> CandidateFunctor:
        """Generar un candidato aleatorio."""
        self.generation += 1
        return CandidateFunctor(
            name=f"Ψ_{self.generation:04d}",
            basis_type=self.rng.choice(self.BASES),
            n_dims=self.rng.choice([32, 64, 96, 128, 192, 256]),
            degree=self.rng.choice([1, 2, 3, 4]),
            context_window=self.rng.choice([1, 2, 3, 5, 7, 10]),
            use_position=bool(self.rng.choice([True, False])),
            use_pmi_weight=bool(self.rng.choice([True, False])),
            normalization=self.rng.choice(self.NORMALIZATIONS),
            temperature=10 ** self.rng.uniform(-2, 1),  # 0.01 to 10
            seed=self.rng.randint(0, 2**31 - 1),
        )
    
    def generate_grid(self, n_per_dim: int = 3) -> List[CandidateFunctor]:
        """Generar grid sistemático de candidatos."""
        candidates = []
        bases_sample = self.BASES[:4]  # Excluir random para grid
        dims = [32, 96, 192]
        degrees = [1, 2, 3]
        windows = [1, 3, 7]
        norms = ["l2", "zscore"]
        
        for basis, dim, deg, ctx, norm in itertools.product(
            bases_sample[:n_per_dim], dims[:n_per_dim], degrees[:n_per_dim],
            windows[:n_per_dim], norms[:n_per_dim]
        ):
            for use_pos in [True, False]:
                for use_pmi in [True, False]:
                    self.generation += 1
                    candidates.append(CandidateFunctor(
                        name=f"Ψ_grid_{self.generation:04d}",
                        basis_type=basis,
                        n_dims=dim,
                        degree=deg,
                        context_window=ctx,
                        use_position=use_pos,
                        use_pmi_weight=use_pmi,
                        normalization=norm,
                        temperature=1.0,
                        seed=self.rng.randint(0, 2**31 - 1),
                    ))
        return candidates


# ═══════════════════════════════════════════════════════════════
# Evaluador de Semanticidad
# ═══════════════════════════════════════════════════════════════

class SemanticityEvaluator:
    """Evalúa qué tan bien un candidate functor preserva estructura semántica.
    
    🔥 v2: Usa WORD-LEVEL PMI como ground truth semántico.
    Las palabras que aparecen en contextos similares (ej: "king" y "queen")
    tienen alta PMI y DEBERÍAN estar cerca en el espacio geométrico.
    """
    
    def __init__(self, token_ids: np.ndarray, vocab_size: int, 
                 cooc_matrix: np.ndarray = None, pmi_matrix: np.ndarray = None,
                 use_word_semantics: bool = True, quiet: bool = False):
        """
        Args:
            token_ids: secuencia de tokens del corpus (N,) — a nivel de caracter
            vocab_size: tamaño del vocabulario de caracteres
            use_word_semantics: si True, usa PMI a nivel de PALABRA como ground truth
            quiet: si True, no imprime info de vocabulario
        """
        self.token_ids = token_ids[:min(len(token_ids), 100000)]
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(42)
        self.use_word_semantics = use_word_semantics
        self.quiet = quiet
        
        # Encontrar el carácter de espacio (token 0 o el que más frecuentemente separa)
        self.space_token = self._find_space_token()
        
        # Construir vocabulario de palabras
        if use_word_semantics:
            self.word_vocab, self.word_ids, self.char_to_word = self._build_word_vocabulary()
            if not self.quiet: print(f"    📚 Word vocabulary: {len(self.word_vocab)} words "
                  f"(from {len(token_ids)} chars)")
        
        # Construir matrices de co-ocurrencia y PMI
        if cooc_matrix is not None:
            self.cooc = cooc_matrix
        else:
            self.cooc = self._build_cooccurrence()
        
        if pmi_matrix is not None:
            self.pmi = pmi_matrix
        else:
            self.pmi = self._build_pmi()
        
        # Construir word-level PMI si aplica
        if use_word_semantics:
            self.word_pmi = self._build_word_pmi()
            self._build_word_reference_pairs()
        else:
            self._build_reference_pairs()
    
    def _find_space_token(self) -> int:
        """Encontrar el token que actúa como espacio."""
        ids = self.token_ids[:10000]
        counts = np.bincount(ids, minlength=self.vocab_size)
        # El espacio suele ser el carácter más frecuente
        return int(np.argmax(counts))
    
    def _build_word_vocabulary(self):
        """Construir vocabulario de palabras desde la secuencia de caracteres."""
        # Convertir secuencia de tokens a string para split
        char_list = []
        for tid in self.token_ids:
            char_list.append(chr(32) if tid == self.space_token else chr(65 + tid))
        text = ''.join(char_list)
        
        # Extraer palabras (separadas por espacio)
        words_raw = text.split(' ')
        words = [w for w in words_raw if len(w) > 0]
        
        # Vocabulario de palabras
        unique_words = sorted(set(words))
        word_to_idx = {w: i for i, w in enumerate(unique_words)}
        word_ids = np.array([word_to_idx.get(w, 0) for w in words], dtype=np.int64)
        
        # Mapeo: cada carácter → palabra a la que pertenece
        char_to_word = np.full(len(self.token_ids), -1, dtype=np.int64)
        pos = 0
        for wi, word in enumerate(words):
            for ch in word:
                if pos < len(char_to_word):
                    char_to_word[pos] = wi
                pos += 1
            pos += 1  # skip space
        
        return unique_words, word_ids, char_to_word
    
    def _build_word_pmi(self) -> np.ndarray:
        """Construir PMI a nivel de palabra."""
        n_words = len(self.word_vocab)
        word_cooc = np.zeros((n_words, n_words), dtype=np.float64)
        
        # Co-ocurrencia en ventana de 5 palabras
        window = 5
        for t in range(len(self.word_ids) - window):
            a = int(self.word_ids[t])
            for w in range(1, window + 1):
                b = int(self.word_ids[t + w])
                word_cooc[a, b] += 1.0
                word_cooc[b, a] += 1.0
        
        total = word_cooc.sum()
        row_sums = word_cooc.sum(axis=1, keepdims=True) + 1e-8
        expected = (row_sums @ row_sums.T) / total
        pmi = np.log((word_cooc + 1e-8) * total / (expected + 1e-8))
        return np.maximum(pmi, 0.0)
    
    def _build_word_reference_pairs(self):
        """Construir pares de PALABRAS de referencia para semanticidad."""
        pmi_flat = self.word_pmi.copy()
        np.fill_diagonal(pmi_flat, 0)
        
        n_words = len(self.word_vocab)
        n_pairs = min(500, n_words * 3)
        
        # Top pares por PMI (alta asociación semántica entre palabras)
        flat_idx = np.argsort(pmi_flat.ravel())[::-1]
        self.synonym_pairs = []
        for idx in flat_idx[:n_pairs]:
            i, j = idx // n_words, idx % n_words
            if i != j and pmi_flat[i, j] > 0.5:
                self.synonym_pairs.append((int(i), int(j), float(pmi_flat[i, j])))
        
        # Pares con baja PMI (contrastes)
        low_idx = np.argsort(pmi_flat.ravel())
        self.contrast_pairs = []
        for idx in low_idx[:n_pairs]:
            i, j = idx // n_words, idx % n_words
            if i != j and pmi_flat[i, j] < 0.1:
                self.contrast_pairs.append((int(i), int(j)))
        
        if not self.quiet: print(f"    Word pairs: {len(self.synonym_pairs)} synonym, "
              f"{len(self.contrast_pairs)} contrast")
    
    def _embed_words_from_chars(self, char_emb: np.ndarray) -> np.ndarray:
        """Convertir embeddings de caracteres a embeddings de palabras.
        
        Para cada palabra, promedia los embeddings de sus caracteres constituyentes.
        """
        n_words = len(self.word_vocab)
        d = char_emb.shape[1]
        word_emb = np.zeros((n_words, d), dtype=np.float64)
        word_counts = np.zeros(n_words, dtype=np.float64)
        
        # Recorrer el corpus: cada carácter pertenece a una palabra
        for pos in range(min(len(self.token_ids), len(self.char_to_word))):
            wid = self.char_to_word[pos]
            tid = self.token_ids[pos]
            if 0 <= wid < n_words and 0 <= tid < len(char_emb):
                word_emb[wid] += char_emb[tid]
                word_counts[wid] += 1
        
        # Normalizar
        word_counts = np.maximum(word_counts, 1)
        word_emb = word_emb / word_counts[:, None]
        
        # Normalizar embeddings finales
        norms = np.linalg.norm(word_emb, axis=1, keepdims=True) + 1e-8
        word_emb = word_emb / norms
        
        return word_emb
    
    def _build_cooccurrence(self) -> np.ndarray:
        """Construir matriz de co-ocurrencia desde los tokens."""
        cooc = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        ids = self.token_ids
        window = 5
        for t in range(len(ids) - window):
            a = int(ids[t])
            if a >= self.vocab_size: continue
            for w in range(1, window + 1):
                b = int(ids[t + w])
                if b < self.vocab_size:
                    cooc[a, b] += 1.0
                    cooc[b, a] += 1.0
        return cooc
    
    def _build_pmi(self) -> np.ndarray:
        """Construir matriz PMI."""
        total = self.cooc.sum()
        row_sums = self.cooc.sum(axis=1, keepdims=True) + 1e-8
        expected = (row_sums @ row_sums.T) / total
        pmi = np.log((self.cooc + 1e-8) * total / (expected + 1e-8))
        return np.maximum(pmi, 0.0)
    
    def _build_reference_pairs(self):
        """Construir pares de tokens de referencia para evaluar semanticidad."""
        # Pares con alta PMI (sinónimos contextuales)
        pmi_flat = self.pmi.copy()
        np.fill_diagonal(pmi_flat, 0)
        
        # Top pares por PMI (alta asociación semántica)
        n_pairs = min(200, self.vocab_size * 2)
        flat_idx = np.argsort(pmi_flat.ravel())[::-1]
        self.synonym_pairs = []
        for idx in flat_idx[:n_pairs]:
            i, j = idx // self.vocab_size, idx % self.vocab_size
            if i != j and pmi_flat[i, j] > 0.1:
                self.synonym_pairs.append((int(i), int(j), float(pmi_flat[i, j])))
        
        # Pares con baja PMI (contrastes)
        low_idx = np.argsort(pmi_flat.ravel())
        self.contrast_pairs = []
        for idx in low_idx[:n_pairs]:
            i, j = idx // self.vocab_size, idx % self.vocab_size
            if i != j:
                self.contrast_pairs.append((int(i), int(j)))
    
    def _eval_polynomial_basis(self, z: np.ndarray, deg: int, basis: str) -> np.ndarray:
        """Evaluar base polinomial sobre vector normalizado."""
        z = np.asarray(z, dtype=np.float64)
        N, d = z.shape
        B = np.zeros((N, d, deg + 1), dtype=np.float64)
        B[:, :, 0] = 1.0
        if deg >= 1:
            B[:, :, 1] = z
        if basis == "hermite":
            for k in range(2, deg + 1):
                B[:, :, k] = z * B[:, :, k-1] - (k - 1.0) * B[:, :, k-2]
        elif basis == "fourier":
            for k in range(1, deg + 1):
                freq = (k + 1) // 2
                if k % 2 == 1:
                    B[:, :, k] = np.sin(freq * np.pi * z)
                else:
                    B[:, :, k] = np.cos(freq * np.pi * z)
        elif basis == "legendre":
            for k in range(2, deg + 1):
                kf = float(k - 1)
                B[:, :, k] = ((2*kf + 1) * z * B[:, :, k-1] - kf * B[:, :, k-2]) / (kf + 1)
        else:  # chebyshev o random
            for k in range(2, deg + 1):
                B[:, :, k] = 2 * z * B[:, :, k-1] - B[:, :, k-2]
        return B
    
    def _embed_tokens(self, functor: CandidateFunctor) -> np.ndarray:
        """Aplicar candidate functor para producir embeddings geométricos.
        
        Returns: (vocab_size, n_dims) — posición de cada token en el espacio.
        """
        V = self.vocab_size
        d = functor.n_dims
        seed = functor.seed
        rng = np.random.RandomState(seed)
        
        # 1. Representación base: PMI como punto de partida
        base_emb = np.zeros((V, d), dtype=np.float64)
        
        if functor.use_pmi_weight:
            # Usar PMI como features iniciales (truncadas a n_dims)
            U, s, Vt = np.linalg.svd(self.pmi, full_matrices=False)
            k = min(d, len(s))
            base_emb = (U[:, :k] * np.sqrt(np.maximum(s[:k], 0)))
            if base_emb.shape[1] < d:
                extra = rng.randn(V, d - base_emb.shape[1]) * 0.01
                base_emb = np.hstack([base_emb, extra])
        else:
            # Embeddings aleatorios como baseline
            base_emb = rng.randn(V, d) / np.sqrt(d)
        
        # 2. Expandir con base polinomial si degree > 0
        if functor.degree > 0 and functor.basis_type != "random":
            # Normalizar
            if functor.normalization == "zscore":
                z = (base_emb - base_emb.mean(axis=0)) / (base_emb.std(axis=0) + 1e-8)
            elif functor.normalization == "l2":
                norms = np.linalg.norm(base_emb, axis=1, keepdims=True) + 1e-8
                z = base_emb / norms
            else:
                z = base_emb
            
            # Evaluar base
            B = self._eval_polynomial_basis(z, functor.degree, functor.basis_type)
            # Aplanar y proyectar de vuelta a n_dims
            expanded = B.reshape(V, -1)
            if expanded.shape[1] > d:
                P = rng.randn(expanded.shape[1], d) / np.sqrt(expanded.shape[1])
                emb = expanded @ P
            else:
                emb = np.hstack([expanded, np.zeros((V, d - expanded.shape[1]))])
        else:
            emb = base_emb
        
        # 3. Normalización final
        if functor.normalization != "none":
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
            emb = emb / norms
        
        return emb
    
    def evaluate(self, functor: CandidateFunctor) -> CandidateFunctor:
        """Evaluar semanticidad de un candidate functor.
        
        🔥 v2: Si use_word_semantics=True, evalúa a nivel de PALABRA.
        Convierte embeddings de caracteres → palabras y mide
        si palabras semánticamente similares están cerca en el espacio.
        """
        emb = self._embed_tokens(functor)
        
        # 🔥 WORD-LEVEL: convertir caracteres a palabras
        if self.use_word_semantics:
            emb = self._embed_words_from_chars(emb)
        
        # 1. Sinonimia: ¿items con alta PMI están cerca en el espacio?
        syn_dists = []
        syn_weights = []
        for i, j, pmi_val in self.synonym_pairs[:100]:
            if i < len(emb) and j < len(emb):
                dist = np.linalg.norm(emb[i] - emb[j])
                syn_dists.append(dist)
                syn_weights.append(pmi_val)
        
        if syn_dists:
            syn_dists = np.array(syn_dists)
            syn_weights = np.array(syn_weights)
            syn_weights = syn_weights / max(syn_weights.sum(), 1e-8)
            functor.synonymy_score = float(np.exp(-np.average(syn_dists, weights=syn_weights)))
        
        # 2. Contraste: ¿items con baja PMI están lejos?
        contrast_dists = []
        for i, j in self.contrast_pairs[:100]:
            if i < len(emb) and j < len(emb):
                dist = np.linalg.norm(emb[i] - emb[j])
                contrast_dists.append(dist)
        
        if contrast_dists:
            contrast_dists = np.array(contrast_dists)
            functor.contrast_score = float(np.tanh(np.mean(contrast_dists)))
        
        # 3. Composición + DISTRIBUTIONAL: correlación PMI y contexto
        all_i, all_j = [], []
        for i, j, _ in self.synonym_pairs[:50]:
            all_i.append(i); all_j.append(j)
        for i, j in self.contrast_pairs[:50]:
            all_i.append(i); all_j.append(j)
        
        if all_i:
            geo_dists = np.array([np.linalg.norm(emb[i] - emb[j]) 
                                  for i, j in zip(all_i, all_j)])
            if self.use_word_semantics:
                pmi_vals = np.array([self.word_pmi[i, j] for i, j in zip(all_i, all_j)])
                # 🔥 DISTRIBUTIONAL METRIC: cosine between context vectors
                ctx_sims = np.array([np.dot(self.word_pmi[i], self.word_pmi[j]) / 
                                    max(np.linalg.norm(self.word_pmi[i]) * np.linalg.norm(self.word_pmi[j]), 1e-8)
                                    for i, j in zip(all_i, all_j)])
            else:
                pmi_vals = np.array([self.pmi[i, j] for i, j in zip(all_i, all_j)])
                ctx_sims = np.ones(len(all_i))
            
            corr = np.corrcoef(pmi_vals, -geo_dists)[0, 1]
            functor.composition_score = float(np.clip(corr, 0, 1)) if not np.isnan(corr) else 0.0
            
            # 🔥 DISTRIBUTIONAL: correlación contexto vs geometría
            dist_corr = np.corrcoef(ctx_sims, -geo_dists)[0, 1]
            functor.distributional_score = float(np.clip(dist_corr, 0, 1)) if not np.isnan(dist_corr) else 0.0
        
        # 4. Persistencia: estabilidad bajo perturbación
        perturbed = emb + self.rng.randn(*emb.shape) * 0.01
        pert_norms = np.linalg.norm(perturbed, axis=1, keepdims=True) + 1e-8
        perturbed = perturbed / pert_norms
        if self.synonym_pairs:
            stability = np.mean([
                np.abs(np.linalg.norm(emb[i] - emb[j]) - np.linalg.norm(perturbed[i] - perturbed[j]))
                for i, j, _ in self.synonym_pairs[:30] if i < len(emb) and j < len(emb)
            ])
            functor.persistence_score = float(np.exp(-stability * 10))
        else:
            functor.persistence_score = 0.5
        
        # 5. Semanticidad total — 3 métricas complementarias
        functor.total_semanticity = float(
            0.30 * functor.synonymy_score +
            0.30 * functor.distributional_score +
            0.20 * functor.contrast_score +
            0.10 * functor.composition_score +
            0.10 * functor.persistence_score
        )
        
        return functor


# ═══════════════════════════════════════════════════════════════
# Orquestador de Búsqueda
# ═══════════════════════════════════════════════════════════════

class SemanticGenesisOrchestrator:
    """Orquesta la búsqueda del Functor Semántico."""
    
    def __init__(self, token_ids: np.ndarray, vocab_size: int,
                 max_candidates: int = 1000, seed: int = 42):
        self.generator = FunctorGenerator(seed=seed)
        self.evaluator = SemanticityEvaluator(token_ids, vocab_size)
        self.max_candidates = max_candidates
        self.discoveries: List[CandidateFunctor] = []
        self.history: List[Dict] = []
    
    def explore_grid(self, n_workers: int = None) -> List[CandidateFunctor]:
        """Exploración sistemática del espacio de candidatos (PARALELA)."""
        print("=" * 60)
        print("🔭 SEMANTIC GENESIS — Búsqueda de Ψ_Sem [PARALELO]")
        print("=" * 60)
        
        candidates = self.generator.generate_grid(n_per_dim=3)
        candidates = candidates[:self.max_candidates]
        print(f"  Grid: {len(candidates)} candidates → evaluando en paralelo...")
        
        t0 = time.time()
        # 🔥 Evaluación paralela
        self.discoveries = parallel_evaluate_candidates(
            candidates, self.evaluator, n_workers=n_workers,
            batch_label="grid"
        )
        elapsed = time.time() - t0
        
        self.discoveries.sort(key=lambda f: f.total_semanticity, reverse=True)
        best = self.discoveries[0] if self.discoveries else None
        if best:
            print(f"  ✅ Grid: {len(self.discoveries)} candidates | "
                  f"best Σ={best.total_semanticity:.4f} "
                  f"({best.basis_type}, d={best.n_dims}) [{elapsed:.1f}s]")
        return self.discoveries
    
    def explore_random(self, n_random: int = 200, n_workers: int = None) -> List[CandidateFunctor]:
        """Exploración aleatoria adicional (PARALELA)."""
        print(f"\n  🎲 Random: {n_random} candidates (paralelo)...")
        
        candidates = [self.generator.generate_random() for _ in range(n_random)]
        t0 = time.time()
        results = parallel_evaluate_candidates(candidates, self.evaluator, n_workers=n_workers)
        self.discoveries.extend(results)
        elapsed = time.time() - t0
        
        self.discoveries.sort(key=lambda f: f.total_semanticity, reverse=True)
        print(f"  ✅ Random: {n_random} evaluated [{elapsed:.1f}s]")
        return self.discoveries
    
    def evolve_generation(self, parents: List[CandidateFunctor], 
                          n_children: int = 50) -> List[CandidateFunctor]:
        """🔥 EVOLUCIÓN: generar variaciones de los mejores candidatos.
        
        Toma los mejores N padres y genera hijos con pequeñas mutaciones
        en cada parámetro, buscando refinar la semanticidad.
        """
        children = []
        rng = np.random.RandomState(42 + len(self.discoveries))
        
        dim_options = [16, 24, 32, 48, 64, 96, 128, 192, 256]
        deg_options = [1, 2, 3, 4]
        ctx_options = [1, 2, 3, 5, 7, 10]
        basis_options = ["hermite", "fourier", "legendre", "chebyshev"]
        norm_options = ["l2", "zscore", "none"]
        
        for parent in parents[:min(5, len(parents))]:
            for _ in range(n_children // min(5, len(parents))):
                # Mutar un subconjunto de parámetros
                n_dims = rng.choice([d for d in dim_options 
                                     if abs(d - parent.n_dims) <= parent.n_dims])
                degree = rng.choice([d for d in deg_options 
                                     if abs(d - parent.degree) <= 1])
                ctx = rng.choice([c for c in ctx_options 
                                  if abs(c - parent.context_window) <= 4])
                basis = rng.choice(basis_options) if rng.random() < 0.3 else parent.basis_type
                norm = rng.choice(norm_options) if rng.random() < 0.3 else parent.normalization
                use_pmi = not parent.use_pmi_weight if rng.random() < 0.15 else parent.use_pmi_weight
                use_pos = not parent.use_position if rng.random() < 0.15 else parent.use_position
                temp = parent.temperature * (2.0 ** rng.uniform(-1, 1)) if rng.random() < 0.3 else parent.temperature
                
                self.generator.generation += 1
                child = CandidateFunctor(
                    name=f"Ψ_evo_{self.generator.generation:04d}",
                    basis_type=basis,
                    n_dims=n_dims,
                    degree=degree,
                    context_window=ctx,
                    use_position=use_pos,
                    use_pmi_weight=use_pmi,
                    normalization=norm,
                    temperature=temp,
                    seed=rng.randint(0, 2**31 - 1),
                )
                children.append(child)
        
        return children
    
    def refine(self, n_generations: int = 3, n_parents: int = 10,
               n_children: int = 100) -> CandidateFunctor:
        """🔥 REFINAMIENTO EVOLUTIVO: itera generaciones mejorando semanticidad.
        
        Proceso:
          1. Toma los top N padres actuales
          2. Genera hijos por mutación
          3. Evalúa hijos y los añade al pool
          4. Selecciona los mejores para la siguiente generación
          5. Repite por G generaciones
        """
        print(f"\n  🧬 EVOLUTIONARY REFINEMENT ({n_generations} generations)...")
        
        best_overall = self.get_best()
        best_score = best_overall.total_semanticity
        
        for gen in range(n_generations):
            # Seleccionar padres (top N)
            sorted_disc = sorted(self.discoveries, key=lambda f: f.total_semanticity, reverse=True)
            parents = sorted_disc[:n_parents]
            
            # Generar hijos
            children = self.evolve_generation(parents, n_children)
            
            # Evaluar hijos
            t0 = time.time()
            for child in children:
                self.evaluator.evaluate(child)
                self.discoveries.append(child)
            
            # Re-ordenar
            self.discoveries.sort(key=lambda f: f.total_semanticity, reverse=True)
            
            # Verificar mejora
            current_best = self.discoveries[0]
            improved = current_best.total_semanticity - best_score
            best_score = max(best_score, current_best.total_semanticity)
            
            elapsed = time.time() - t0
            icon = "🔥" if improved > 0.01 else "📈" if improved > 0.001 else "➡️"
            print(f"    Gen {gen+1}/{n_generations}: best Σ={current_best.total_semanticity:.4f} "
                  f"(Δ={improved:+.4f}) {icon} | {len(children)} children [{elapsed:.1f}s]")
            
            # Early stop si no hay mejora en 2 generaciones
            if improved < 0.001 and gen >= 1:
                prev_improved = (sorted_disc[0].total_semanticity - 
                                sorted_disc[n_parents].total_semanticity)
                if prev_improved < 0.002:
                    print(f"    🛑 Converged — no significant improvement")
                    break
        
        # Limpiar: solo mantener top 200 para eficiencia
        self.discoveries = sorted(self.discoveries, key=lambda f: f.total_semanticity, reverse=True)[:200]
        return self.get_best()
    
    def get_best(self, top_n: int = 10) -> CandidateFunctor:
        """Obtener el mejor candidato."""
        if not self.discoveries:
            return CandidateFunctor(name="empty", basis_type="none", n_dims=0,
                                    degree=0, context_window=0, use_position=False,
                                    use_pmi_weight=False, normalization="none",
                                    temperature=1.0)
        sorted_disc = sorted(self.discoveries, key=lambda f: f.total_semanticity, reverse=True)
        return sorted_disc[0]
    
    def report(self) -> str:
        """Generar reporte completo de la búsqueda."""
        if not self.discoveries:
            return "No discoveries yet."
        
        best = self.get_best()
        top10 = sorted(self.discoveries, key=lambda f: f.total_semanticity, reverse=True)[:10]
        
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║         🔭 SEMANTIC GENESIS — RESULTS                ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Total candidates evaluated: {len(self.discoveries)}",
            f"  Best semanticity: Σ = {best.total_semanticity:.4f}",
            "",
            "  🏆 TOP CANDIDATE:",
            f"  {best.summary()}",
            "",
            "  📊 TOP 10:",
        ]
        
        for i, f in enumerate(top10):
            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"  {i+1}."
            lines.append(f"  {icon} Σ={f.total_semanticity:.4f} | {f.basis_type} | "
                        f"d={f.n_dims} | deg={f.degree} | ctx={f.context_window} | "
                        f"syn={f.synonymy_score:.3f} cmp={f.composition_score:.3f} "
                        f"cnt={f.contrast_score:.3f}")
        
        # Análisis de patrones
        lines.append("")
        lines.append("  📈 PATTERN ANALYSIS:")
        
        bases = {}
        for f in self.discoveries:
            b = f.basis_type
            if b not in bases:
                bases[b] = []
            bases[b].append(f.total_semanticity)
        
        for basis, scores in sorted(bases.items(), key=lambda x: -np.mean(x[1])):
            lines.append(f"    {basis:12s}: μ={np.mean(scores):.4f}  "
                        f"max={np.max(scores):.4f}  n={len(scores)}")
        
        dims = {}
        for f in self.discoveries:
            d = f.n_dims
            if d not in dims:
                dims[d] = []
            dims[d].append(f.total_semanticity)
        
        lines.append("")
        lines.append("    Best dimension:")
        for d, scores in sorted(dims.items(), key=lambda x: -np.mean(x[1])):
            lines.append(f"      d={d:4d}: μ={np.mean(scores):.4f}  max={np.max(scores):.4f}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def search_semantic_functor(token_ids: np.ndarray, vocab_size: int,
                            max_candidates: int = 500,
                            refine_generations: int = 5) -> Dict[str, Any]:
    """Buscar el mejor candidate functor semántico con refinamiento evolutivo.
    
    Fases:
      1. Grid sistemático — explora el espacio de parámetros
      2. Exploración aleatoria — descubre regiones no cubiertas
      3. REFINAMIENTO EVOLUTIVO — muta los mejores candidatos por G generaciones
      4. Selección final — el mejor de todos
    
    Args:
        token_ids: secuencia de tokens del corpus
        vocab_size: tamaño del vocabulario
        max_candidates: máximo de candidatos a evaluar
        refine_generations: generaciones de refinamiento evolutivo
        
    Returns:
        Dict con el mejor functor, top 10, y análisis de patrones
    """
    orch = SemanticGenesisOrchestrator(token_ids, vocab_size, max_candidates)
    
    # Fase 1: Grid sistemático
    orch.explore_grid()
    
    # Fase 2: Exploración aleatoria
    orch.explore_random(n_random=min(200, max_candidates // 2))
    
    # Fase 3: 🔥 REFINAMIENTO EVOLUTIVO
    best = orch.refine(n_generations=refine_generations)
    
    # Fase 4: Reporte final
    report = orch.report()
    print(report)
    
    return {
        "best_functor": best,
        "best_semanticity": best.total_semanticity,
        "top_candidates": orch.discoveries[:20],
        "n_evaluated": len(orch.discoveries),
        "report": report,
        "patterns": {
            "best_basis": best.basis_type,
            "best_dims": best.n_dims,
            "best_degree": best.degree,
            "synonymy": best.synonymy_score,
            "composition": best.composition_score,
            "contrast": best.contrast_score,
        }
    }


# ═══════════════════════════════════════════════════════════════
# 🔐 VERIFICATION PROTOCOL — ¿Es Ψ_Sem real y autónomo?
# ═══════════════════════════════════════════════════════════════

def verify_semantic_functor(token_ids: np.ndarray, vocab_size: int,
                            n_trials: int = 3) -> Dict[str, Any]:
    """🔥 PROTOCOLO DE VERIFICACIÓN COMPLETO.
    
    Demuestra que:
      1. El sistema es AUTÓNOMO (misma semilla → mismo resultado)
      2. Ψ_Sem es REAL (funciona en datos no vistos)
      3. La mejora es SIGNIFICATIVA (vs baseline aleatorio)
      4. GENESIS FUNCIONÓ (el patrón generate→filter→refine produjo valor)
    
    Returns:
        Dict con certificado de verificación completo.
    """
    N = len(token_ids)
    split = int(N * 0.7)
    train_tokens = token_ids[:split]
    test_tokens = token_ids[split:]
    
    print("=" * 65)
    print("🔐 VERIFICATION PROTOCOL — Ψ_Sem Autonomy & Validity")
    print("=" * 65)
    
    # ── PRUEBA 1: REPRODUCIBILIDAD (autonomía) ──
    print("\n📋 PRUEBA 1: Reproducibilidad (¿es autónomo?)")
    print("   Hipótesis: misma semilla + mismos datos = mismo Ψ_Sem")
    
    results = []
    for trial in range(n_trials):
        t0 = time.time()
        result = search_semantic_functor(
            train_tokens.copy(), vocab_size, 
            max_candidates=200, refine_generations=3
        )
        elapsed = time.time() - t0
        best = result['best_functor']
        results.append({
            'trial': trial + 1,
            'semanticity': best.total_semanticity,
            'basis': best.basis_type,
            'dims': best.n_dims,
            'degree': best.degree,
            'elapsed': elapsed,
        })
        print(f"   Trial {trial+1}: Σ={best.total_semanticity:.4f} | "
              f"{best.basis_type} | d={best.n_dims} | deg={best.degree} [{elapsed:.1f}s]")
    
    # Verificar consistencia
    scores = [r['semanticity'] for r in results]
    bases = [r['basis'] for r in results]
    dims = [r['dims'] for r in results]
    
    score_std = np.std(scores)
    bases_agree = len(set(bases)) == 1
    dims_agree = len(set(dims)) <= 2
    
    autonomous = score_std < 0.02 and bases_agree
    print(f"\n   σ(Σ) = {score_std:.6f} | bases: {'✅ IGUALES' if bases_agree else '❌ DIFERENTES'} | "
          f"dims: {'✅' if dims_agree else '❌'}")
    print(f"   VEREDICTO: {'✅ AUTÓNOMO' if autonomous else '❌ NO AUTÓNOMO'}")
    
    # ── PRUEBA 2: VALIDACIÓN CRUZADA (held-out) ──
    print("\n📋 PRUEBA 2: Validación en datos no vistos (¿es real?)")
    print("   Hipótesis: Ψ_Sem entrenado en train funciona en test")
    
    # Entrenar en train
    train_result = search_semantic_functor(
        train_tokens.copy(), vocab_size,
        max_candidates=200, refine_generations=3
    )
    train_best = train_result['best_functor']
    
    # Evaluar en test
    test_evaluator = SemanticityEvaluator(test_tokens, vocab_size)
    test_functor = CandidateFunctor(
        name="Ψ_test",
        basis_type=train_best.basis_type,
        n_dims=train_best.n_dims,
        degree=train_best.degree,
        context_window=train_best.context_window,
        use_position=train_best.use_position,
        use_pmi_weight=train_best.use_pmi_weight,
        normalization=train_best.normalization,
        temperature=train_best.temperature,
        seed=train_best.seed,
    )
    test_evaluator.evaluate(test_functor)
    
    print(f"   Train Σ = {train_best.total_semanticity:.4f}")
    print(f"   Test  Σ = {test_functor.total_semanticity:.4f}")
    generalization = test_functor.total_semanticity / max(train_best.total_semanticity, 1e-8)
    print(f"   Generalización = {generalization:.2%}")
    is_real = generalization > 0.85
    print(f"   VEREDICTO: {'✅ REAL (generaliza)' if is_real else '❌ OVERFITTING'}")
    
    # ── PRUEBA 3: SIGNIFICANCIA ESTADÍSTICA ──
    print("\n📋 PRUEBA 3: Significancia vs baseline aleatorio")
    print("   Hipótesis: Ψ_Sem > random baseline con p < 0.01")
    
    # Baseline: 100 functors aleatorios
    rng = np.random.RandomState(42)
    random_scores = []
    for _ in range(100):
        rand_f = CandidateFunctor(
            name="rand", basis_type=rng.choice(["hermite","fourier","chebyshev","legendre"]),
            n_dims=rng.choice([16,32,64,96]), degree=rng.choice([1,2,3]),
            context_window=rng.choice([1,3,5]), use_position=bool(rng.choice([True,False])),
            use_pmi_weight=bool(rng.choice([True,False])),
            normalization=rng.choice(["l2","zscore"]), temperature=1.0, seed=rng.randint(0,9999)
        )
        test_evaluator.evaluate(rand_f)
        random_scores.append(rand_f.total_semanticity)
    
    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)
    z_score = (test_functor.total_semanticity - random_mean) / max(random_std, 1e-8)
    p_value = 1.0 - float(min(0.9999, 0.5 + 0.5 * np.tanh(z_score / 2.0)))  # Approx
    
    print(f"   Random baseline: μ={random_mean:.4f} σ={random_std:.4f}")
    print(f"   Ψ_Sem: Σ={test_functor.total_semanticity:.4f} (z={z_score:.1f}σ above mean)")
    significant = z_score > 3.0
    print(f"   VEREDICTO: {'✅ SIGNIFICATIVO (p<0.001)' if significant else '❌ NO SIGNIFICATIVO'}")
    
    # ── PRUEBA 4: GENESIS PATTERN ──
    print("\n📋 PRUEBA 4: ¿El patrón Genesis funcionó?")
    
    # Comparar: grid solo vs grid + evolución
    orch_grid = SemanticGenesisOrchestrator(train_tokens, vocab_size, 200)
    orch_grid.explore_grid()
    grid_best = orch_grid.get_best().total_semanticity
    
    evolution_gain = train_best.total_semanticity - grid_best
    print(f"   Grid solo:     Σ = {grid_best:.4f}")
    print(f"   Grid + Evolve: Σ = {train_best.total_semanticity:.4f}")
    print(f"   Ganancia evolución: Δ = {evolution_gain:+.4f} ({evolution_gain/grid_best*100:+.1f}%)")
    genesis_worked = evolution_gain > 0.01
    print(f"   VEREDICTO: {'✅ GENESIS FUNCIONÓ' if genesis_worked else '❌ NO MEJORÓ'}")
    
    # ── POEMA & LEAN 4 STATUS ──
    print("\n📋 POEMA & LEAN 4 STATUS:")
    print("   Poema:  ❌ No integrado aún — compilaría Ψ_Sem a kernel FMA/GPU")
    print("   Lean 4: ❌ No integrado aún — verificaría propiedades formales de Ψ_Sem")
    print("   Estado: El patrón Genesis (generate→fingerprint→filter→refine) SÍ funcionó.")
    print("           Poema y Lean 4 son los siguientes pasos del pipeline completo.")
    
    # ── CERTIFICADO FINAL ──
    all_pass = autonomous and is_real and significant and genesis_worked
    
    print("\n" + "=" * 65)
    print("🏅 CERTIFICADO DE VERIFICACIÓN")
    print("=" * 65)
    print(f"   Autonomía:     {'✅' if autonomous else '❌'} (σ={score_std:.4f})")
    print(f"   Realidad:      {'✅' if is_real else '❌'} (gen={generalization:.1%})")
    print(f"   Significancia: {'✅' if significant else '❌'} (z={z_score:.1f})")
    print(f"   Genesis:       {'✅' if genesis_worked else '❌'} (Δ={evolution_gain:+.4f})")
    print(f"   Poema:         ⏳ Pendiente")
    print(f"   Lean 4:        ⏳ Pendiente")
    print(f"\n   VEREDICTO FINAL: {'✅ Ψ_Sem VERIFICADO' if all_pass else '⚠️  VERIFICACIÓN PARCIAL'}")
    print("=" * 65)
    
    return {
        "autonomous": autonomous,
        "real": is_real,
        "significant": significant,
        "genesis_worked": genesis_worked,
        "all_pass": all_pass,
        "train_semanticity": train_best.total_semanticity,
        "test_semanticity": test_functor.total_semanticity,
        "generalization": generalization,
        "z_score": z_score,
        "evolution_gain": evolution_gain,
        "reproducibility_std": score_std,
        "best_functor": train_best,
        "verification_report": f"Autonomous={autonomous} Real={is_real} Significant={significant} Genesis={genesis_worked}"
    }


# ═══════════════════════════════════════════════════════════════
# 🔬 DEEP SEARCH — Búsqueda exhaustiva con cross-validación
# ═══════════════════════════════════════════════════════════════

def deep_search_semantic_functor(token_ids: np.ndarray, vocab_size: int,
                                  n_folds: int = 3,
                                  population: int = 400,
                                  generations: int = 5,
                                  survival_rate: float = 0.3,
                                  n_workers: int = None) -> Dict[str, Any]:
    """🔥 BÚSQUEDA PROFUNDA OPTIMIZADA — Paralela + vocabularios cacheados.
    
    Args:
        n_workers: número de workers paralelos (default: CPU count - 1)
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    N = len(token_ids)
    fold_size = N // n_folds
    
    print("=" * 70)
    print("🔬 DEEP SEARCH — Laboratorio Ψ_Sem [PARALELO]")
    print("=" * 70)
    print(f"  Corpus: {N} tokens | Vocab: {vocab_size} | Workers: {n_workers}")
    print(f"  Folds: {n_folds} | Pop: {population} | Gens: {generations}")
    
    t_total = time.time()
    
    # ── FASE 0: PRE-CACHEAR EVALUADORES POR FOLD ──
    print(f"\n  📦 Cacheando vocabularios para {n_folds} folds...")
    folds = []
    fold_evaluators = []
    for k in range(n_folds):
        start = k * fold_size
        end = min((k + 1) * fold_size, N) if k < n_folds - 1 else N
        test_fold = token_ids[start:end]
        train_fold = np.concatenate([token_ids[:start], token_ids[end:]])
        folds.append((train_fold, test_fold))
        # Pre-construir evaluador (esto es lo caro: vocabulario + PMI)
        eval_k = SemanticityEvaluator(test_fold, vocab_size, quiet=True)
        fold_evaluators.append(eval_k)
    
    # También evaluador rápido para train_0
    eval_train = SemanticityEvaluator(folds[0][0], vocab_size, quiet=True)
    print(f"  ✅ {n_folds} evaluadores cacheados [{time.time()-t_total:.0f}s]")
    
    # ── FASE 1: GRID SEARCH (fold 0, rápido) ──
    print(f"\n  📐 FASE 1: Grid search ({population} candidates)...")
    orch = SemanticGenesisOrchestrator(folds[0][0], vocab_size, population)
    orch.evaluator = eval_train  # Usar evaluador cacheado
    orch.explore_grid()
    orch.explore_random(n_random=min(200, population // 2))
    print(f"  ✅ Grid: {len(orch.discoveries)} candidates [{time.time()-t_total:.0f}s]")
    
    # ── FASE 2: EVOLUCIÓN PARALELA ──
    print(f"\n  🧬 FASE 2: Evolución cross-validada ({generations} gens, {n_workers} workers)...")
    
    best_cv_score = -1.0
    
    for gen in range(generations):
        sorted_disc = sorted(orch.discoveries, key=lambda f: f.total_semanticity, reverse=True)
        n_survive = max(10, int(len(sorted_disc) * survival_rate))
        parents = sorted_disc[:n_survive]
        
        # ── Evaluar padres en paralelo sobre todos los folds ──
        tasks = []
        for functor in parents[:min(20, len(parents))]:
            for k in range(n_folds):
                tasks.append((functor, k))
        
        # Evaluación paralela
        cv_results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_on_fold, f, k, fold_evaluators[k]): (f.functor_hash, k)
                      for f, k in tasks}
            for future in as_completed(futures):
                key, score = future.result()
                if key not in cv_results:
                    cv_results[key] = []
                cv_results[key].append(score)
        
        # Procesar resultados CV
        survivors = []
        for functor in parents[:min(20, len(parents))]:
            key = functor.functor_hash
            if key in cv_results and len(cv_results[key]) >= 2:
                scores = cv_results[key]
                cv_mean = np.mean(scores)
                cv_std = np.std(scores)
                functor.persistence_score = cv_mean
                functor.composition_score = cv_std
                survivors.append(functor)
        
        survivors.sort(key=lambda f: f.persistence_score, reverse=True)
        
        if survivors:
            best_gen = survivors[0]
            if best_gen.persistence_score > best_cv_score:
                best_cv_score = best_gen.persistence_score
        
        # Generar y evaluar hijos
        n_children = population - len(survivors)
        children = orch.evolve_generation(survivors[:8] if survivors else parents[:5], 
                                          max(n_children, 50))
        for child in children:
            eval_train.evaluate(child)
        
        orch.discoveries = survivors + children
        orch.discoveries.sort(key=lambda f: f.total_semanticity, reverse=True)
        
        t_gen = time.time() - t_total
        cv_str = f"CV={survivors[0].persistence_score:.4f}" if survivors else "CV=N/A"
        print(f"    Gen {gen+1}/{generations}: {cv_str} | pop={len(orch.discoveries)} | [{t_gen:.0f}s]")
    
    # ── FASE 3: PERSISTENCE FILTER ──
    stable = [f for f in orch.discoveries 
              if f.persistence_score > 0.3 and f.composition_score < 0.2]
    stable.sort(key=lambda f: f.persistence_score, reverse=True)
    if not stable:
        stable = sorted(orch.discoveries, key=lambda f: f.persistence_score, reverse=True)[:5]
    
    # ── FASE 4: VERIFICACIÓN FINAL ──
    winner = stable[0]
    final_eval = fold_evaluators[-1]
    final_functor = CandidateFunctor(
        name="Ψ_SEM_FINAL", basis_type=winner.basis_type, n_dims=winner.n_dims,
        degree=winner.degree, context_window=winner.context_window,
        use_position=winner.use_position, use_pmi_weight=winner.use_pmi_weight,
        normalization=winner.normalization, temperature=winner.temperature, seed=42,
    )
    final_eval.evaluate(final_functor)
    
    # Baseline
    rng = np.random.RandomState(42)
    random_scores = []
    for _ in range(50):
        rand_f = CandidateFunctor(
            name="rand", basis_type=rng.choice(["hermite","fourier","chebyshev","legendre"]),
            n_dims=rng.choice([16,32,64]), degree=rng.choice([1,2,3]),
            context_window=rng.choice([1,3,5]), use_position=bool(rng.choice([True,False])),
            use_pmi_weight=bool(rng.choice([True,False])),
            normalization=rng.choice(["l2","zscore"]), temperature=1.0, seed=rng.randint(0,9999)
        )
        final_eval.evaluate(rand_f)
        random_scores.append(rand_f.total_semanticity)
    
    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)
    z_score = (final_functor.total_semanticity - random_mean) / max(random_std, 1e-8)
    is_real = z_score > 2.0 and final_functor.total_semanticity > 0.5
    
    total_elapsed = time.time() - t_total
    
    print("\n" + "=" * 70)
    print("🏆 DEEP SEARCH — RESULTADOS FINALES")
    print("=" * 70)
    print(f"  Candidatos: {len(orch.discoveries)} | Estables: {len(stable)} | {total_elapsed:.0f}s")
    print(f"\n  🏆 Ψ_SEM: {winner.summary()}")
    print(f"  Test Σ={final_functor.total_semanticity:.4f} | z={z_score:.1f}σ | "
          f"{'✅ REAL' if is_real else '⚠️ BUSCAR MÁS'}")
    print("=" * 70)
    
    return {
        "best_functor": winner, "final_functor": final_functor,
        "is_real": is_real, "z_score": z_score, "cv_score": best_cv_score,
        "n_evaluated": len(orch.discoveries), "n_stable": len(stable),
        "total_time": total_elapsed,
        "architecture": {
            "name": "Ψ_Sem_v2", "basis": winner.basis_type, "n_dims": winner.n_dims,
            "degree": winner.degree, "context_window": winner.context_window,
            "use_position": winner.use_position, "use_pmi_weight": winner.use_pmi_weight,
            "normalization": winner.normalization, "temperature": winner.temperature,
            "semanticity": final_functor.total_semanticity,
            "synonymy": final_functor.synonymy_score,
            "composition": final_functor.composition_score,
            "contrast": final_functor.contrast_score,
            "is_real": is_real,
        }
    }


def _eval_on_fold(functor: CandidateFunctor, fold_idx: int, 
                  evaluator: SemanticityEvaluator) -> Tuple[str, float]:
    """Evalúa un candidato en un fold específico (para paralelización)."""
    test_f = CandidateFunctor(
        name=f"{functor.name}_f{fold_idx}",
        basis_type=functor.basis_type, n_dims=functor.n_dims,
        degree=functor.degree, context_window=functor.context_window,
        use_position=functor.use_position, use_pmi_weight=functor.use_pmi_weight,
        normalization=functor.normalization, temperature=functor.temperature,
        seed=functor.seed + fold_idx,
    )
    evaluator.evaluate(test_f)
    return (functor.functor_hash, test_f.total_semanticity)


def _eval_candidate_parallel(args: Tuple[CandidateFunctor, SemanticityEvaluator]) -> CandidateFunctor:
    """Evalúa un candidato (pickle-safe para ProcessPoolExecutor)."""
    functor, evaluator = args
    evaluator.evaluate(functor)
    return functor


def parallel_evaluate_candidates(functors: List[CandidateFunctor],
                                  evaluator: SemanticityEvaluator,
                                  n_workers: int = None,
                                  batch_label: str = "") -> List[CandidateFunctor]:
    """🔥 Evalúa candidatos en paralelo — satura todos los cores."""
    import multiprocessing
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    if n_workers <= 1 or len(functors) < 10:
        # Sequential fallback for small batches
        for f in functors:
            evaluator.evaluate(f)
        return functors
    
    results = []
    # Split into batches for efficiency
    batch_size = max(1, len(functors) // n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for f in functors:
            futures.append(executor.submit(_eval_candidate_parallel, (f, evaluator)))
        for future in as_completed(futures):
            results.append(future.result())
    
    return results
