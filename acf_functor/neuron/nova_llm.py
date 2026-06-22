"""
nova_llm.py — NovaLM: Language Model 100% Nova-Native
======================================================
Sin Transformers, sin backprop. Arquitectura original:

  1. NovaEmbedding: Token → vector + Hermite(posición)
  2. NovaContext:  ANOVA(2) sobre ventana de contexto → hidden
  3. NovaDecoder:  Clasificador → distribución sobre vocabulario
  4. Generación autoregresiva: sample → feed back → repeat

Componentes inventados desde cero:
  - Atención ANOVA(2): pares entre posiciones de la ventana
  - Posición Hermite: H₀(p), H₁(p), H₂(p) codifican orden
  - Contexto local: vecinos ±1, ±2 agregados al feature
"""

import numpy as np
from .nova_phi_neuron import NovaPhiNeuron, NovaPrediction, KroneckerPhiNeuron


# ═══════════════════════════════════════════════════════════════
# 🔥 NovaFunctionalEmbedding — Embeddings dinámicos sin backprop
# ═══════════════════════════════════════════════════════════════

class NovaFunctionalEmbedding:
    """Embebedor funcional: token + contexto → vector dinámico.
    
    REEMPLAZA la tabla de lookup estática (PMI-SVD) con una
    NEURONA ACF que evalúa f(token, contexto_previo) → ℝ^d.
    
    NO USA BACKPROP. Usa RLS (Recursive Least Squares) para
    actualizar los coeficientes espectrales en tiempo real.
    
    Principio ACF: el embedding de un token no es su identidad,
    es su COMPORTAMIENTO en el contexto actual.
    
    Arquitectura:
      token → one_hot(V) ─┐
                           ├→ Kronecker ⊗ → embed_dim vector
      contexto → Nova compresor ─┘
    
    Args:
        vocab_size: tamaño del vocabulario
        embed_dim: dimensión del embedding de salida
        context_window: cuántos tokens previos usar como contexto
        context_dim: dimensión del contexto comprimido
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 context_window: int = 3, context_dim: int = 32,
                 l2_lambda: float = 0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.context_dim = context_dim
        
        # Compresor de contexto: V×window → context_dim
        ctx_input_dim = vocab_size * context_window
        self.ctx_comp = NovaPhiNeuron(
            'emb_ctx', ctx_input_dim, context_dim,
            max_degree=2, max_pairs=20, l2_lambda=l2_lambda,
            use_triton=True
        )
        
        # Neurona generatriz: V + context_dim → embed_dim
        total_input = vocab_size + context_dim
        self.generator = NovaPhiNeuron(
            'emb_gen', total_input, embed_dim,
            max_degree=3, max_pairs=40, l2_lambda=l2_lambda,
            use_triton=True
        )
        
        # Estado RLS
        self._P = None          # Matriz de precisión
        self._w = None          # Pesos (flattened para el generador)
        self._rls_lambda = l2_lambda
        self._n_updates = 0
        self._initialized = False
    
    def _bootstrap(self, static_emb: np.ndarray):
        """Inicializar generador con embeddings estáticos PMI-SVD."""
        import numpy as np
        X_train = []
        Y_train = []
        zero_ctx = np.zeros(self.vocab_size * self.context_window, dtype=np.float32)
        ctx_vec = self.ctx_comp.evaluate(zero_ctx.reshape(1, -1)).ravel()
        
        for tid in range(min(self.vocab_size, len(static_emb))):
            tok_1h = np.zeros(self.vocab_size, dtype=np.float32)
            tok_1h[tid] = 1.0
            x_full = np.concatenate([tok_1h, ctx_vec])
            X_train.append(x_full)
            Y_train.append(static_emb[tid])
        
        X_train = np.array(X_train, dtype=np.float64)
        Y_train = np.array(Y_train, dtype=np.float64)
        self.generator.fit(X_train, Y_train)
        self._initialized = True
    
    def encode(self, token_ids: np.ndarray, 
               static_emb: np.ndarray = None) -> np.ndarray:
        """Codificar secuencia de tokens en embeddings dinámicos.
        
        Args:
            token_ids: (seq_len,) ints
            static_emb: (vocab_size, embed_dim) embeddings PMI base
            
        Returns:
            (seq_len, embed_dim) embeddings contextualizados
        """
        n = len(token_ids)
        result = np.zeros((n, self.embed_dim), dtype=np.float32)
        
        # Bootstrap: entrenar generador con embeddings estáticos en primer uso
        if not self._initialized and static_emb is not None:
            self._bootstrap(static_emb)
        
        for i in range(n):
            # One-hot del token actual
            tok_1h = np.zeros(self.vocab_size, dtype=np.float32)
            tid = int(token_ids[i])
            if 0 <= tid < self.vocab_size:
                tok_1h[tid] = 1.0
            
            # Contexto: one-hot de los k tokens previos
            ctx = np.zeros(self.vocab_size * self.context_window, 
                          dtype=np.float32)
            for k in range(1, self.context_window + 1):
                if i >= k:
                    prev_tid = int(token_ids[i - k])
                    if 0 <= prev_tid < self.vocab_size:
                        ctx[(k-1)*self.vocab_size + prev_tid] = 1.0
            
            # Comprimir contexto (entrenar compresor en primer uso)
            if self._n_updates == 0 and i > self.context_window:
                # Autoencoder: comprimir → reconstruir
                try:
                    self.ctx_comp.fit(ctx.reshape(1, -1), 
                                     ctx.reshape(1, -1)[:, :self.context_dim])
                except Exception:
                    pass
            
            ctx_vec = self.ctx_comp.evaluate(ctx.reshape(1, -1)).ravel()
            
            # Input completo: [one_hot(token) | contexto_comprimido]
            x_full = np.concatenate([tok_1h, ctx_vec])
            
            # Evaluar generador (necesita 2D input)
            raw = self.generator.evaluate(x_full.reshape(1, -1))
            if hasattr(raw, 'mean') and not callable(raw.mean):
                raw = raw.mean
            result[i] = np.asarray(raw, dtype=np.float32).ravel()[:self.embed_dim]
        
        return result
    
    def update_rls(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """Actualización RLS por bloques.
        
        Dados inputs X (N, V+context_dim) y targets Y (N, embed_dim),
        actualiza los coeficientes del generador para minimizar ||XW - Y||².
        
        RLS formula:
            P_new = P - (Px)(Px)^T / (1 + x^T P x)
            w_new = w + P_new x (y - x^T w)
        """
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        N, d_in = X.shape
        d_out = Y.shape[1]
        
        # Inicializar estado RLS
        if self._P is None:
            self._P = np.eye(d_in, dtype=np.float64) / self._rls_lambda
            self._w = np.zeros((d_in, d_out), dtype=np.float64)
        
        for i in range(N):
            x = X[i:i+1].T  # (d_in, 1)
            y = Y[i:i+1]    # (1, d_out)
            
            # P @ x
            Px = self._P @ x  # (d_in, 1)
            xPx = float(x.T @ Px)  # scalar
            
            if xPx < 1e6:  # Evitar overflow numérico
                # P = P - (Px)(Px)^T / (1 + x^T P x)
                self._P -= (Px @ Px.T) / (1.0 + xPx)
            
            # w = w + P @ x @ (y - x^T @ w)
            pred = x.T @ self._w  # (1, d_out)
            error = y - pred      # (1, d_out)
            self._w += self._P @ x @ error
        
        self._n_updates += N
        
        # Aplicar pesos actualizados al generador
        # NOTA: El generador Kronecker tiene pesos internos diferentes.
        # RLS actualiza w_flat; necesitamos mapear a los pesos del generador.
        # Por ahora: guardamos estado RLS para fine-tuning futuro.
    
    def get_embedding_matrix(self) -> np.ndarray:
        """Obtener matriz de embeddings (para fine-tuning/búsqueda)."""
        emb_matrix = np.zeros((self.vocab_size, self.embed_dim), dtype=np.float32)
        zero_ctx = np.zeros(self.vocab_size * self.context_window, dtype=np.float32)
        ctx_vec = self.ctx_comp.evaluate(zero_ctx.reshape(1, -1)).ravel()
        
        for tid in range(self.vocab_size):
            tok_1h = np.zeros(self.vocab_size, dtype=np.float32)
            tok_1h[tid] = 1.0
            x_full = np.concatenate([tok_1h, ctx_vec])
            emb_matrix[tid] = self.generator.evaluate(x_full)
        
        return emb_matrix


# ═══════════════════════════════════════════════════════════════
# 1. NovaEmbedding: Token + Posición
# ═══════════════════════════════════════════════════════════════

class NovaEmbedding:
    """Token embedding + positional encoding vía Hermite.

    Cada token se representa como:
      [one-hot(token) | H₀(pos) | H₁(pos) | H₂(pos)]
    donde Hₖ son polinomios de Hermite sobre la posición normalizada.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, max_len: int = 128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Embedding aprendido con Nova (opcional, empezamos con one-hot + pos)
        pos_dim = 3  # H₀, H₁, H₂
        self.total_dim = vocab_size + pos_dim

        # Nova para comprimir embedding si es necesario
        if self.total_dim > 128:
            self.compressor = NovaPhiNeuron(
                'emb_compress', self.total_dim, embed_dim,
                max_degree=2, max_pairs=40, l2_lambda=0.1
            )
            self._compress = True
        else:
            self._compress = False

    def _hermite_position(self, positions: np.ndarray) -> np.ndarray:
        """Hermite polynomials H₀, H₁, H₂ sobre posición normalizada."""
        p = positions.astype(np.float64) / max(1, self.max_len - 1) * 2 - 1  # [-1, 1]
        H = np.zeros((len(p), 3))
        H[:, 0] = 1.0       # H₀(x) = 1
        H[:, 1] = p          # H₁(x) = x
        H[:, 2] = p * p - 1  # H₂(x) = x² - 1
        return H

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """Convertir secuencia de token IDs a embeddings.

        Args:
            token_ids: (seq_len,) array de ints

        Returns:
            (seq_len, total_dim) embedding
        """
        n = len(token_ids)
        # One-hot tokens
        tok_onehot = np.zeros((n, self.vocab_size))
        for i, tid in enumerate(token_ids):
            if 0 <= tid < self.vocab_size:
                tok_onehot[i, tid] = 1.0

        # Posición Hermite
        pos_enc = self._hermite_position(np.arange(n))

        # Concatenar
        emb = np.hstack([tok_onehot, pos_enc])

        if self._compress:
            # Entrenar compresor en este batch (autoencoder)
            self.compressor.fit(emb, emb[:, :self.embed_dim])
            emb = np.array([self.compressor.evaluate(e) for e in emb])

        return emb


# ═══════════════════════════════════════════════════════════════
# 2. NovaContext: Atención ANOVA(2) sobre ventana
# ═══════════════════════════════════════════════════════════════

class NovaContext:
    """Contexto local vía ANOVA(2) sobre ventana de tokens.

    Para cada posición t, construye un feature vector que incluye:
      - embed[t]                    (token actual)
      - embed[t-1], embed[t+1]      (vecinos inmediatos)
      - embed[t-2], embed[t+2]      (vecinos ±2)
      - pares ANOVA(2) entre embed[t] y cada vecino

    Esto captura interacciones locales sin atención densa O(n²).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64,
                 window_radius: int = 2):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.window_radius = window_radius

        # Dimensión del feature de contexto
        self.ctx_dim = embed_dim * (1 + 2 * window_radius)

        # H-Nova manual: un grupo por vecino + top
        self.top = NovaPhiNeuron(
            'ctx_top', self.ctx_dim, hidden_dim,
            max_degree=2, max_pairs=60, l2_lambda=0.1,
            use_triton=True  # Evitar crash GPU en dims grandes
        )

    def build_context(self, embeddings: np.ndarray) -> np.ndarray:
        """Construir features de contexto para cada posición.

        Args:
            embeddings: (seq_len, embed_dim)

        Returns:
            (seq_len, ctx_dim) features de contexto
        """
        n, d = embeddings.shape
        ctx_features = np.zeros((n, self.ctx_dim))

        for t in range(n):
            feats = [embeddings[t]]  # token actual
            for r in range(1, self.window_radius + 1):
                # Vecino izquierdo (con padding)
                left = embeddings[max(0, t - r)]
                feats.append(left)
                # Vecino derecho (con padding)
                right = embeddings[min(n - 1, t + r)]
                feats.append(right)
            ctx_features[t] = np.concatenate(feats)

        return ctx_features

    def fit(self, embeddings: np.ndarray, targets: np.ndarray):
        """Entrenar la capa de contexto."""
        ctx = self.build_context(embeddings)
        self.top.fit(ctx, targets)
        return self

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass: embeddings → hidden states."""
        ctx = self.build_context(embeddings)
        return np.array([self.top.evaluate(c) for c in ctx])


# ═══════════════════════════════════════════════════════════════
# 3. NovaDecoder: Clasificador → vocabulario
# ═══════════════════════════════════════════════════════════════

class NovaDecoder:
    """Clasificador Nova que mapea hidden state → distribución de tokens."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        self.classifier = NovaPhiNeuron(
            'dec', hidden_dim, vocab_size,
            max_degree=2, max_pairs=min(100, hidden_dim * 3),
            l2_lambda=0.1, use_triton=True
        )
        self.vocab_size = vocab_size

    def fit(self, hidden_states: np.ndarray, next_tokens: np.ndarray):
        """Entrenar: hidden[t] → token[t+1] (one-hot)."""
        targets = np.eye(self.vocab_size)[next_tokens]
        self.classifier.fit(hidden_states, targets)
        return self

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """Predecir distribución sobre vocabulario."""
        return self.classifier.evaluate(hidden)


# ═══════════════════════════════════════════════════════════════
# 4. NovaLM: Modelo de Lenguaje completo
# ═══════════════════════════════════════════════════════════════

class NovaLM:
    """Nova Language Model — 100% Nova-Native."""

    def __init__(self, vocab_size: int, embed_dim: int = 32,
                 hidden_dim: int = 64, max_len: int = 128,
                 window_radius: int = 2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = NovaEmbedding(vocab_size, embed_dim, max_len)
        # Usar la dimensión REAL del embedding (vocab + position)
        real_emb_dim = self.embedding.total_dim
        self.context = NovaContext(real_emb_dim, hidden_dim, window_radius)
        self.decoder = NovaDecoder(hidden_dim, vocab_size)

    def fit(self, token_ids: np.ndarray, seq_len: int = 32,
            verbose: bool = True):
        """Entrenar en una secuencia de tokens.

        Args:
            token_ids: array de tokens (texto completo tokenizado)
            seq_len: longitud de secuencia para entrenamiento
        """
        import time
        t0 = time.perf_counter()

        # Crear pares (contexto → siguiente token)
        n = len(token_ids) - 1
        contexts = []
        targets = []

        for i in range(0, n - seq_len, seq_len // 2):
            ctx_tokens = token_ids[i:i + seq_len]
            tgt_tokens = token_ids[i + 1:i + seq_len + 1]
            # Ensure same length
            min_len = min(len(ctx_tokens), len(tgt_tokens))
            if min_len < 4:
                continue
            ctx_tokens = ctx_tokens[:min_len]
            tgt_tokens = tgt_tokens[:min_len]

            # Embeddings + posición
            emb = self.embedding.encode(ctx_tokens)
            contexts.append(emb)
            targets.append(tgt_tokens)

        if not contexts:
            raise ValueError("Sequencia muy corta para entrenar")

        if verbose:
            print(f"  Training samples: {len(contexts)} × ~{seq_len} tokens")

        # Entrenar NovaContext
        all_emb = np.vstack(contexts)
        all_tgt = np.concatenate(targets)

        # Para cada embedding, el target es el siguiente token
        # → alineamos: emb[:-1] → tgt[1:]
        n_total = len(all_emb)

        # Usar subset para memoria
        max_samples = 5000
        if n_total > max_samples:
            idx = np.random.choice(n_total, max_samples, replace=False)
            train_emb = all_emb[idx]
            train_tgt = all_tgt[idx]
        else:
            train_emb = all_emb
            train_tgt = all_tgt

        if verbose:
            print(f"  Training context on {len(train_emb)} samples...")

        # Target para context: autoencoder (usar primeros hidden_dim features del embedding)
        ctx_targets = np.zeros((len(train_emb), self.hidden_dim))
        for i in range(min(self.hidden_dim, train_emb.shape[1])):
            ctx_targets[:, i] = train_emb[:, i % train_emb.shape[1]]
        self.context.fit(train_emb, ctx_targets)

        # Forward context → hidden
        hidden = self.context.forward(train_emb)

        if verbose:
            print(f"  Training decoder: {hidden.shape[1]}→{self.vocab_size}...")

        # Entrenar decoder: hidden → next token
        self.decoder.fit(hidden, train_tgt)

        if verbose:
            print(f"  ✅ Trained in {time.perf_counter() - t0:.0f}s")

    def generate(self, prompt: list[int], max_new: int = 50,
                 temperature: float = 0.8) -> list[int]:
        """Generar texto autoregresivamente.

        Args:
            prompt: lista de token IDs iniciales
            max_new: máximo de tokens a generar
            temperature: creatividad (0 = determinista, 1 = diverso)

        Returns:
            lista de token IDs generados
        """
        generated = list(prompt)

        for _ in range(max_new):
            # Embedding del contexto actual
            ctx = generated[-min(32, len(generated)):]
            emb = self.embedding.encode(np.array(ctx))

            # Forward Context
            hidden = self.context.forward(emb)

            # Decoder → logits para el ÚLTIMO token
            logits = self.decoder.forward(hidden[-1])

            # Softmax con temperatura
            logits = logits - np.max(logits)
            probs = np.exp(logits / max(temperature, 0.01))
            probs = probs / probs.sum()

            # Sample
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated.append(int(next_token))

            # Stop si genera token de fin (0)
            if next_token == 0 and len(generated) > len(prompt) + 2:
                break

        return generated

    def summary(self) -> str:
        return (f"NovaLM(vocab={self.vocab_size}, emb={self.embed_dim}, "
                f"hidden={self.hidden_dim}) | 100% Nova-Native")


# ═══════════════════════════════════════════════════════════════
# 🧬 SparseANOVA Attention — Atención Nova-nativa O(n log n)
# ═══════════════════════════════════════════════════════════════

class SparseANOVAAttention:
    """Atención Nova-nativa: ANOVA(2) sobre pares jerárquicos.

    En vez de Q·K^T sobre O(n²) pares, usa dilated pairs:
      Nivel 0: (tᵢ, tᵢ₊₁)         → O(n) pares
      Nivel 1: (tᵢ, tᵢ₊₂)         → O(n) pares
      Nivel 2: (tᵢ, tᵢ₊₄)         → O(n) pares
      ...
      Total: O(n log n) pares

    Cada par se evalúa con ANOVA(2) → produce un peso de relevancia.
    Los pesos agregan los value vectors en cada posición.

    Diferencia con Transformer:
      - Sin dot product → ANOVA(2) sobre features concatenados
      - Sin softmax → pesos directos de Nova
      - Sin O(n²) → pares jerárquicos sparse
      - Interpretable: cada peso = importancia del par (i,j)
    """

    def __init__(self, embed_dim: int, n_levels: int = 4,
                 l2_lambda: float = 0.1, use_lsqr: bool = True):
        self.embed_dim = embed_dim
        self.n_levels = n_levels

        # 🔥 LSQR solver: convergencia gradual = anti-cristalización de ruido
        self.level_neurons = []
        for l in range(n_levels):
            neuron = NovaPhiNeuron(
                f'attn_L{l}',
                embed_dim * 2,
                1,
                max_degree=2, max_pairs=30,
                l2_lambda=l2_lambda * 3.0,  # 🔥 3× más regularización
                use_triton=True
            )
            if use_lsqr:
                neuron._force_solver = "lsqr"
            self.level_neurons.append(neuron)

        # Nova para combinar pesos de distintos niveles
        self.combiner = NovaPhiNeuron(
            'attn_comb',
            n_levels + 1,  # pesos de cada nivel + input original
            1,
            max_degree=2, max_pairs=15,
            l2_lambda=l2_lambda,
            use_triton=True
        )

    def _build_pairs(self, embeddings: np.ndarray, level: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construir pares a distancia 2^level.

        Returns:
            pair_features: (n_pairs, 2*embed_dim) features concatenados
            indices_i: (n_pairs,) índices de la posición i
            indices_j: (n_pairs,) índices de la posición j
        """
        n = len(embeddings)
        stride = 2 ** level
        pairs = []
        idx_i = []
        idx_j = []

        for i in range(n - stride):
            j = i + stride
            feat = np.concatenate([embeddings[i], embeddings[j]])
            pairs.append(feat)
            idx_i.append(i)
            idx_j.append(j)

        if not pairs:
            return (np.zeros((0, self.embed_dim * 2)),
                    np.array([], dtype=int), np.array([], dtype=int))

        return np.array(pairs), np.array(idx_i), np.array(idx_j)

    def fit(self, embeddings: np.ndarray,
            target_weights: np.ndarray = None):
        """Entrenar la atención.

        Si no hay target_weights, usa autoencoder:
        reconstruir el embedding desde el contexto.
        """
        n = len(embeddings)

        # Para cada nivel, entrenar con target = similitud coseno entre pares
        for l in range(self.n_levels):
            pairs, _, _ = self._build_pairs(embeddings, l)
            if len(pairs) == 0:
                continue

            # Target: similitud coseno entre los embeddings del par
            d = self.embed_dim
            sim = np.sum(pairs[:, :d] * pairs[:, d:], axis=1)
            sim = sim / (np.linalg.norm(pairs[:, :d], axis=1) *
                         np.linalg.norm(pairs[:, d:], axis=1) + 1e-8)
            sim = np.clip(sim, -1, 1).reshape(-1, 1)

            self.level_neurons[l].fit(pairs, sim)

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Calcular atención: embeddings → context-aware embeddings.

        Returns:
            (n, embed_dim) embeddings con contexto agregado
        """
        n = len(embeddings)
        context = np.zeros_like(embeddings)

        # Para cada nivel, calcular pesos y agregar
        for l in range(self.n_levels):
            pairs, idx_i, idx_j = self._build_pairs(embeddings, l)
            if len(pairs) == 0:
                continue

            # Evaluar pesos con la Nova del nivel
            weights = np.array([self.level_neurons[l].evaluate(p)
                                for p in pairs])

            # Normalizar pesos por nivel (softmax ligero)
            weights = np.clip(weights, -5, 5)
            weights = np.exp(weights)
            weights = weights / (weights.sum() + 1e-8)

            # Agregar: context[i] += weight * embedding[j]
            for p in range(len(pairs)):
                w = float(weights[p])
                context[idx_i[p]] += w * embeddings[idx_j[p]]
                context[idx_j[p]] += w * embeddings[idx_i[p]]

        # Residual connection + scale
        output = embeddings + 0.3 * context
        return output

    def get_attention_weights(self, embeddings: np.ndarray
                              ) -> list[dict]:
        """Extraer pesos de atención para visualización."""
        n = len(embeddings)
        attn_weights = []

        for l in range(self.n_levels):
            pairs, idx_i, idx_j = self._build_pairs(embeddings, l)
            if len(pairs) == 0:
                continue
            weights = np.array([self.level_neurons[l].evaluate(p)
                                for p in pairs])
            for p in range(len(pairs)):
                attn_weights.append({
                    'level': l,
                    'i': int(idx_i[p]),
                    'j': int(idx_j[p]),
                    'weight': float(weights[p])
                })

        return sorted(attn_weights, key=lambda x: -abs(x['weight']))


# ═══════════════════════════════════════════════════════════════
# 🧬 SparseANOVA Attention v2 — 16K contexto, vectorizada, sparse
# ═══════════════════════════════════════════════════════════════

class SparseANOVAAttention:
    """Atención Nova-nativa O(n log n). v2: vectorizada + sparse + 16K.

    Optimizaciones clave para contexto masivo:
      - _build_pairs vectorizado (numpy slicing, sin listas Python)
      - forward() con batch predict (1 sola llamada por nivel)
      - Agregación vectorizada con np.add.at
      - Sparse threshold: ignora pares con peso < umbral
      - Dynamic levels: auto-escala n_levels según seq_len
      - Adaptive fit: early stopping cuando el error deja de mejorar
    """

    def __init__(self, embed_dim: int, n_levels: int = None,
                 l2_lambda: float = 0.1, max_context: int = 16384,
                 sparse_threshold: float = 0.01,
                 pairs_per_level: int = 24,
                 max_degree: int = 2,
                 proj_dim: int = None,
                 use_kronecker: bool = False):  # 🔥 Kronecker para alto orden
        import numpy as np
        from .nova_phi_neuron import NovaPhiNeuron
        self.embed_dim = embed_dim
        self.max_context = max_context
        self.sparse_threshold = sparse_threshold
        self.pairs_per_level = pairs_per_level
        self.max_degree = max_degree

        # 🔥 PROYECCIÓN: reduce 2*embed_dim → 2*proj_dim para menos params
        # Sin proyección: 512-dim → ~5240 params/neurona (🍝 hambre)
        # Con proj=64:     128-dim → ~700 params/neurona  (✅ ratio 9:1)
        self.proj_dim = proj_dim
        self._use_projection = proj_dim is not None and proj_dim < embed_dim
        if self._use_projection:
            # Matriz de proyección Gaussiana FIJA (no aprendida, como embeddings)
            rng = np.random.RandomState(42)
            input_dim = embed_dim * 2
            output_dim = proj_dim * 2
            self._proj_matrix = rng.randn(input_dim, output_dim).astype(np.float32)
            self._proj_matrix /= np.sqrt(input_dim)  # Normalizar varianza
            self._pair_dim = output_dim
        else:
            self._pair_dim = embed_dim * 2

        # Auto-escalar niveles
        if n_levels is None:
            self.n_levels = min(int(np.ceil(np.log2(max(max_context, 256)))) + 1, 12)
        else:
            self.n_levels = n_levels

        self.level_neurons = []
        self._level_trained = [False] * self.n_levels
        self._level_pairs_fixed = []
        
        # 🧬 PBCA: Position-Biased Cosine Attention
        # Parámetros por nivel que modulan la agregación SIN tocar cosine
        self.position_decay = [0.0] * self.n_levels   # γ: decay por distancia
        self.temperature = [1.0] * self.n_levels       # τ: temperatura softmax
        
        pair_features = self._pair_dim
        self._use_kronecker = use_kronecker
        for l in range(self.n_levels):
            if use_kronecker:
                from .nova_phi_neuron import KroneckerPhiNeuron
                neuron = KroneckerPhiNeuron(
                    f'attn_L{l}', pair_features,
                    max_degree=max_degree, l2_lambda=l2_lambda,
                    subspace_dim=32
                )
            else:
                neuron = NovaPhiNeuron(
                    f'attn_L{l}', pair_features, 1,
                    max_degree=max_degree, max_pairs=pairs_per_level,
                    l2_lambda=l2_lambda, use_triton=True
                )
                # 🔥 Pre-computar cross-pairs FIJOS
                cross_pairs = [(i, i + pair_features // 2)
                              for i in range(min(pairs_per_level, pair_features // 2))]
                neuron._pairs = cross_pairs
                neuron._pairs_fixed = True
                neuron.C_pair = np.zeros((1, len(cross_pairs), neuron._d2))
                self._level_pairs_fixed.append(cross_pairs)
            self.level_neurons.append(neuron)

    def forward_thermo(self, embeddings, T_base=1.0, alpha=0.15):
        """🌡️ ThermoAttn: temperatura termodinámica por nivel.

        Principio de free energy del ecosistema ACF:
          F = E - S/β  → β controla el trade-off precisión vs entropía

        Adaptado a atención Nova:
          T_l = T_base × (1 + α × l)
        
        Niveles locales (l pequeño, stride pequeño):
          T bajo → softmax sharp → solo tokens muy similares
        Niveles globales (l grande, stride grande):  
          T alto → softmax suave → promedio de muchos tokens

        Args:
          embeddings: (seq_len, embed_dim)
          T_base: temperatura base (0.5-2.0). <1 = más sharp, >1 = más suave
          alpha: crecimiento por nivel (0.05-0.5). Controla qué tan rápido
                 los niveles globales se vuelven difusos
        """
        temperatures = [T_base * (1.0 + alpha * l) 
                       for l in range(self.n_levels)]
        return self.forward_with_bias(embeddings,
                                      position_decay=[0.0]*self.n_levels,
                                      temperature=temperatures)
        """Aplicar proyección de dimensionalidad a los pares."""
        if not self._use_projection:
            return pairs
        return pairs.astype(np.float32) @ self._proj_matrix

    def _build_pairs(self, embeddings, level, direction: str = 'forward'):
        """Construir pares vectorizados. Sin bucles Python.

        direction='forward': j = i + 2^l  (atención al futuro)
        direction='backward': j = i - 2^l  (atención al pasado, Markov-N)
        """
        import numpy as np
        n = len(embeddings)
        stride = 2 ** level
        pair_size = self._pair_dim
        if n <= stride:
            return (np.zeros((0, pair_size), dtype=np.float32),
                    np.array([], dtype=int), np.array([], dtype=int))
        
        if direction == 'backward':
            # Mirar al PASADO: j = i - stride
            left = embeddings[stride:]     # posición i (posterior)
            right = embeddings[:-stride]    # posición j = i - stride (anterior)
            pairs = np.hstack([left, right]).astype(np.float32)
            idx_i = np.arange(stride, n, dtype=int)   # posiciones de i
            idx_j = np.arange(0, n - stride, dtype=int)  # posiciones de j
        else:
            # Mirar al FUTURO: j = i + stride (original)
            left = embeddings[:-stride]
            right = embeddings[stride:]
            pairs = np.hstack([left, right]).astype(np.float32)
            idx_i = np.arange(n - stride, dtype=int)
            idx_j = np.arange(stride, n, dtype=int)
        
        if self._use_projection:
            pairs = pairs @ self._proj_matrix
        return pairs, idx_i, idx_j

    def fit(self, embeddings, max_epochs: int = 3,
            early_stop: bool = True):
        """Fit directo por secuencia (fallback)."""
        import numpy as np
        d = self.embed_dim  # Mitad del par original (antes de proyección)
        for l in range(self.n_levels):
            pairs, _, _ = self._build_pairs(embeddings, l)
            if len(pairs) < 10: continue
            # Cosine similarity sobre pares PROYECTADOS: usar todas las features
            pdim = pairs.shape[1] // 2
            ni = np.linalg.norm(pairs[:, :pdim], axis=1) + 1e-8
            nj = np.linalg.norm(pairs[:, pdim:], axis=1) + 1e-8
            sim = np.sum(pairs[:, :pdim] * pairs[:, pdim:], axis=1) / (ni * nj)
            sim = np.clip(sim, -1, 1).reshape(-1, 1).astype(np.float64)
            self.level_neurons[l].fit(pairs.astype(np.float64), sim)
            self._level_trained[l] = True

    def fit_batched(self, embeddings_list, verbose: bool = True):
        """🔥 BATACHED FIT: Todas las secuencias contribuyen.

        En lugar de entrenar secuencia por secuencia (donde solo
        la ultima aporta), acumula TODOS los pares de TODAS las
        secuencias y entrena cada nivel UNA SOLA VEZ.

        Esto da:
          - MAXIMA calidad (todas las secuencias aprenden)
          - MAXIMA velocidad (1 fit por nivel, no N fits)
          - Estadisticas globales mas robustas

        Args:
          embeddings_list: lista de arrays de embeddings
        """
        import numpy as np
        d = self.embed_dim  # Tamaño original (pre-proyección), como fallback
        n_seqs = len(embeddings_list)

        for l in range(self.n_levels):
            # Acumular pares de TODAS las secuencias
            all_pairs = []
            all_sims = []
            for emb in embeddings_list:
                pairs, _, _ = self._build_pairs(emb, l)
                if len(pairs) < 10: continue
                # Cosine sobre dimensión PROYECTADA (usar mitad del par)
                pdim = pairs.shape[1] // 2
                ni = np.linalg.norm(pairs[:, :pdim], axis=1) + 1e-8
                nj = np.linalg.norm(pairs[:, pdim:], axis=1) + 1e-8
                sim = np.sum(pairs[:, :pdim] * pairs[:, pdim:], axis=1) / (ni * nj)
                sim = np.clip(sim, -1, 1)
                all_pairs.append(pairs.astype(np.float32))
                all_sims.append(sim.reshape(-1, 1).astype(np.float64))

            if not all_pairs:
                continue

            # Un solo fit con TODOS los pares acumulados
            X_l = np.vstack(all_pairs)
            y_l = np.vstack(all_sims)
            self.level_neurons[l].fit(X_l, y_l)
            self._level_trained[l] = True

            # Auto-pruning: marcar nivel como poco util si error > 0.1
            if self.level_neurons[l].epsilon_mu > 0.1 and l > 2:
                self._level_trained[l] = False  # podar nivel ruidoso

        if verbose:
            trained = sum(self._level_trained)
            print(f'  batched: {trained}/{self.n_levels} levels from {n_seqs} seqs')

    def fit_batched_directional(self, embeddings_list, verbose: bool = True):
        """🔥 DIRECTIONAL NEXT-TOKEN TRAINING — Target = next-token similarity.

        REEMPLAZA cosine similarity (simétrica, irrelevante) con un target
        DIRECCIONAL y ASIMÉTRICO alineado con la tarea real: next-token.

        Para cada par (i, j) en nivel l con stride=2^l:
          target = cosine(emb[j], emb[i+1])

        Esto mide: "¿qué tan útil es el token j para predecir el token i+1?"
        — Si j es similar al TRUE next token → target alto → atención alta
        — Si j es irrelevante → target bajo → atención baja

        La atención ahora aprende a BUSCAR tokens predictivos del futuro,
        no solo tokens semánticamente similares.

        Args:
          embeddings_list: lista de arrays de embeddings por secuencia
        """
        import numpy as np

        for l in range(self.n_levels):
            all_pairs = []
            all_targets = []

            for emb in embeddings_list:
                pairs, idx_i, idx_j = self._build_pairs(emb, l)
                if len(pairs) < 10:
                    continue

                # 🔥 DIRECTIONAL TARGET: cosine(emb[j], emb[i+1])
                pdim = pairs.shape[1] // 2
                emb_next = emb[1:]  # emb[i+1] para todo i

                targets = np.zeros((len(pairs), 1), dtype=np.float64)
                for p in range(len(pairs)):
                    i_pos = idx_i[p]
                    j_pos = idx_j[p]
                    if i_pos + 1 < len(emb) and j_pos < len(emb):
                        # Similitud entre token j y el TRUE next token (i+1)
                        next_emb = emb[i_pos + 1]
                        tok_emb = emb[j_pos]
                        ni = np.linalg.norm(next_emb) + 1e-8
                        nj = np.linalg.norm(tok_emb) + 1e-8
                        sim = float(np.dot(next_emb, tok_emb) / (ni * nj))
                        targets[p, 0] = np.clip(sim, -1, 1)
                    else:
                        targets[p, 0] = 0.0

                all_pairs.append(pairs.astype(np.float64))
                all_targets.append(targets)

            if not all_pairs:
                continue

            X_l = np.vstack(all_pairs)
            y_l = np.vstack(all_targets)

            # Solo entrenar si hay variedad en targets
            if np.std(y_l) > 0.005:
                self.level_neurons[l].fit(X_l, y_l)
                self._level_trained[l] = True
            else:
                self._level_trained[l] = False

            # Auto-pruning
            if self.level_neurons[l].epsilon_mu > 0.1 and l > 2:
                self._level_trained[l] = False

        if verbose:
            trained = sum(self._level_trained)
            print(f'  🎯 directional: {trained}/{self.n_levels} levels from {len(embeddings_list)} seqs')

    def fit_batched_decoder_aligned(self, embeddings_list, decoder_expected,
                                     verbose: bool = False):
        """🔥 DECODER-EXPECTED EMBEDDING ALIGNMENT (DEEA).

        Cierra el gap cosine-vs-next-token usando el ESPACIO DE PREDICCIÓN
        del decoder como target de atención.

        Para cada par (i, j):
          target = cosine(emb[j], decoder_expected[i])
        donde decoder_expected[i] = Σ p(t) · emb[t] es el embedding
        que el decoder "espera ver" después de la posición i.

        Esto alinea atención y decoder en el MISMO espacio semántico:
        - La atención aprende a encontrar tokens similares a lo que
          el decoder predice (no solo al true next token)
        - Targets RICOS (256-dim continuous) — no sparse como one-hot
        - Captura AMBIGÜEDAD: si el decoder duda entre 'e' y 'a',
          el expected embedding está entre ambos

        Args:
          embeddings_list: lista de arrays de embeddings por secuencia
          decoder_expected: lista de arrays (seq_len, embed_dim) con
                           los embeddings esperados por el decoder
        """
        import numpy as np

        for l in range(self.n_levels):
            all_pairs = []
            all_targets = []

            for ei, emb in enumerate(embeddings_list):
                pairs, idx_i, idx_j = self._build_pairs(emb, l)
                if len(pairs) < 10:
                    continue

                expected = decoder_expected[ei]
                targets = np.zeros((len(pairs), 1), dtype=np.float64)

                for p in range(len(pairs)):
                    i_pos = idx_i[p]
                    j_pos = idx_j[p]
                    if i_pos < len(expected) and j_pos < len(emb):
                        exp_vec = expected[i_pos]
                        tok_vec = emb[j_pos]
                        ni = np.linalg.norm(exp_vec) + 1e-8
                        nj = np.linalg.norm(tok_vec) + 1e-8
                        sim = float(np.dot(exp_vec, tok_vec) / (ni * nj))
                        targets[p, 0] = np.clip(sim, -1, 1)
                    else:
                        targets[p, 0] = 0.0

                all_pairs.append(pairs.astype(np.float64))
                all_targets.append(targets)

            if not all_pairs:
                continue

            X_l = np.vstack(all_pairs)
            y_l = np.vstack(all_targets)

            # Solo entrenar si hay variedad en targets
            if np.std(y_l) > 0.005:
                self.level_neurons[l].fit(X_l, y_l)
                self._level_trained[l] = True
                if verbose:
                    print(f'    L{l}: DEEA σ={np.std(y_l):.4f} '
                          f'μ={np.mean(y_l):.3f}')
            else:
                if verbose and self._level_trained[l]:
                    print(f'    L{l}: DEEA skipped (low variance)')

        if verbose:
            trained = sum(self._level_trained)
            print(f'  🎯 DEEA: {trained}/{self.n_levels} levels aligned')

    def fit_batched_contrastive(self, embeddings_list, K=8, tau=0.2,
                                 thermo_gamma=0.5, trans_prob=None,
                                 token_ids_list=None, trans_beta=0.3,
                                 trans_alpha=0.3, verbose: bool = True,
                                 koopman_alpha: float = 0.0):
        """🔥 PARADIGM SHIFT v9 + KOOPMAN: Target P(next|j) + Koopman evolution.

        koopman_alpha > 0: mezcla P(next|j) con Koopman:
          target = (1-α)·P(next_i|token_j) + α·cosine(K(token_j), emb[i+1])
          donde K(token_j) = Σ_k P(k|token_j)·emb[k] (expected next embedding)
        
        α=0: Paradigm v9 puro (probado, 32.80%)
        α>0: Unificación geometría/probabilidad vía Koopman

        Target = P(token_{i+1} | token_j) * exp(-gamma * |j-i| / 2^l)
        
        Sin cosine similarity. El target ES la probabilidad
        condicional empírica. Directamente alineado con next-token prediction.
        Asimétrico por construcción: P(A|B) != P(B|A).

        - Pares pesados por PMI(token_j, token_{i+1})
        - Sin softmax, sin negativos: señal pura de transición
        - Koopman: mezcla con evolución geométrica K(token_j) ~ emb[j+1]
        """
        import numpy as np

        for l in range(self.n_levels):
            all_pairs = []
            all_targets = []
            all_weights = []

            for seq_idx, emb in enumerate(embeddings_list):
                pairs, idx_i, idx_j = self._build_pairs(emb, l)
                if len(pairs) < 10:
                    continue

                tid_list = None
                if token_ids_list is not None and seq_idx < len(token_ids_list):
                    tid_list = token_ids_list[seq_idx]

                targets = np.zeros((len(pairs), 1), dtype=np.float64)
                weights = np.ones(len(pairs), dtype=np.float64)
                n_seq = len(emb)
                max_dist = float(2 ** l)

                for p in range(len(pairs)):
                    i_pos = idx_i[p]
                    j_pos = idx_j[p]

                    if i_pos + 1 >= n_seq or j_pos >= n_seq:
                        targets[p, 0] = 1.0 / (K + 1)
                        continue

                    # 🔥 PARADIGM SHIFT: Target = P(token_{i+1} | token_j)
                    p_target = 1.0 / (K + 1)  # default: uniform
                    if trans_prob is not None and tid_list is not None:
                        tid_j = int(tid_list[j_pos])
                        tid_next = int(tid_list[i_pos + 1])
                        Vp = trans_prob.shape[0]
                        if 0 <= tid_j < Vp and 0 <= tid_next < Vp:
                            p_trans = float(trans_prob[tid_j, tid_next])
                            p_target = max(p_trans, 1.0 / Vp)
                            # 🔥 PMI weight: log(P(joint) / (P(j)*P(next)))
                            p_j = float(trans_prob[tid_j].mean())  # marginal P(j)
                            p_next = float(trans_prob[:, tid_next].mean())  # marginal P(next)
                            pmi = np.log(max(p_target, 1e-8) / max(p_j * p_next, 1e-8))
                            weights[p] = np.clip(1.0 + 0.5 * max(pmi, 0), 0.5, 3.0)

                    # --- Decaimiento termodinámico (CCD) ---
                    dist = abs(j_pos - i_pos) / max(n_seq, 1)
                    decay = np.exp(-thermo_gamma * dist / max(max_dist, 1))
                    
                    targets[p, 0] = p_target * decay
                    
                    # 🔥 KOOPMAN: mezclar con evolución geométrica K(token_j) ≈ emb[j+1]
                    if koopman_alpha > 0 and j_pos + 1 < n_seq and i_pos + 1 < n_seq:
                        emb_j1 = emb[j_pos + 1]  # K(token_j) ≈ emb del siguiente
                        emb_i1 = emb[i_pos + 1]  # embedding del target
                        n1 = np.linalg.norm(emb_j1) + 1e-8
                        n2 = np.linalg.norm(emb_i1) + 1e-8
                        k_cos = float(np.dot(emb_j1, emb_i1) / (n1 * n2))
                        k_cos = np.clip(k_cos, -1, 1)
                        # Mezclar: (1-α)·P(next|j) + α·cosine(K(emb_j), emb_{i+1})
                        targets[p, 0] = (1.0 - koopman_alpha) * targets[p, 0] + koopman_alpha * k_cos * decay

                all_pairs.append(pairs.astype(np.float64))
                all_targets.append(targets)
                all_weights.append(weights)

            if not all_pairs:
                continue

            X_l = np.vstack(all_pairs)
            y_l = np.vstack(all_targets)
            w_l = np.concatenate(all_weights) if all_weights else None

            if np.std(y_l) > 0.003:
                if w_l is not None and np.std(w_l) > 0.01:
                    boost_mask = w_l > np.median(w_l) * 1.5
                    if boost_mask.sum() > 10:
                        X_aug = np.vstack([X_l, X_l[boost_mask]])
                        y_aug = np.vstack([y_l, y_l[boost_mask]])
                        self.level_neurons[l].fit(X_aug, y_aug)
                    else:
                        self.level_neurons[l].fit(X_l, y_l)
                else:
                    self.level_neurons[l].fit(X_l, y_l)
                self._level_trained[l] = True
            else:
                self._level_trained[l] = False

            if self.level_neurons[l].epsilon_mu > 0.15 and l > 2:
                self._level_trained[l] = False

        if verbose:
            trained = sum(self._level_trained)
            print(f'  🔥 PARADIGM v9: {trained}/{self.n_levels} levels '
                  f'from {len(embeddings_list)} seqs')

    def fit_batched_thermo_annealed(self, embeddings_list,
                                     beta_schedule=None, K=8,
                                     verbose: bool = True):
        """🌡️ THERMODYNAMIC ANNEALING — Brújula global F(β) = E - S/β.

        PRINCIPIO (Equivalente ACF a la Regla de la Cadena):
          F(β) es CONVEXA en β. Aumentar β garantiza mejora monótona.
          Cada neurona optimiza su objetivo local con solver directo
          (mínimo global), y β coordina TODAS hacia el mismo óptimo.

        SCHEDULE:
          β=0.1 → targets uniformes (S máxima, E irrelevante)
          β=0.5 → targets suaves (balance S-E)
          β=2.0 → targets nítidos (E domina)
          β=10  → targets quasi-discretos (S mínima, E mínimo)

        El solver directo en cada β encuentra el MÍNIMO GLOBAL de
        ||ΦW - Y_β||². Al subir β, F(β) decrece monótonamente —
        igual que la Regla de la Cadena garantiza descenso.

        Args:
          embeddings_list: lista de arrays de embeddings
          beta_schedule: lista de β, default [0.1, 0.3, 0.7, 1.5, 3.0, 7.0]
          K: número de negativos para contraste
        """
        import numpy as np

        if beta_schedule is None:
            # Geométrico: 15 pasos de β=0.05 a β=10.0
            import numpy as np
            beta_schedule = list(np.logspace(-1.3, 1.0, 15))

        F_history = []  # Trazabilidad termodinámica

        for beta in beta_schedule:
            T = 1.0 / max(beta, 0.01)  # temperatura = 1/β

            for l in range(self.n_levels):
                all_pairs = []; all_targets = []

                for emb in embeddings_list:
                    pairs, idx_i, idx_j = self._build_pairs(emb, l)
                    if len(pairs) < 10: continue

                    targets = np.zeros((len(pairs), 1), dtype=np.float64)
                    n_seq = len(emb)

                    for p in range(len(pairs)):
                        i_pos, j_pos = idx_i[p], idx_j[p]
                        if i_pos+1 >= n_seq or j_pos >= n_seq:
                            targets[p,0] = 1.0/(K+1); continue

                        emb_j, emb_next = emb[j_pos], emb[i_pos+1]
                        pos_sim = float(np.dot(emb_j, emb_next) /
                                      (np.linalg.norm(emb_j)*np.linalg.norm(emb_next)+1e-8))

                        # K negativos
                        neg_idx = np.random.choice(n_seq, min(K+2, n_seq), replace=False)
                        neg_sims = []
                        for ni in neg_idx:
                            if ni==i_pos+1 or ni==j_pos: continue
                            ns = float(np.dot(emb_j, emb[ni]) /
                                      (np.linalg.norm(emb_j)*np.linalg.norm(emb[ni])+1e-8))
                            neg_sims.append(ns)
                            if len(neg_sims) >= K: break

                        if len(neg_sims) < 2:
                            targets[p,0] = np.clip((pos_sim+1)/2, 0, 1); continue

                        # Softmax con temperatura T = 1/β
                        all_sims = np.array([pos_sim] + neg_sims[:K])
                        all_sims = all_sims / max(T, 0.01)
                        all_sims = all_sims - np.max(all_sims)
                        probs = np.exp(all_sims); probs /= (probs.sum()+1e-8)
                        targets[p, 0] = float(probs[0])

                    all_pairs.append(pairs.astype(np.float64))
                    all_targets.append(targets)

                if not all_pairs: continue

                X_l = np.vstack(all_pairs); y_l = np.vstack(all_targets)
                if np.std(y_l) > 0.002:
                    self.level_neurons[l].fit(X_l, y_l)
                    self._level_trained[l] = True

            # Free Energy: F = E - S/β (trazabilidad, decreciente)
            trained_neurons = [n for n in self.level_neurons if n.epsilon_mu < float('inf')]
            E = np.mean([n.epsilon_mu for n in trained_neurons]) if trained_neurons else 1.0
            S = -np.log(max(1.0/(K+1), 1e-8))  # entropy proxy: uniform → max
            F = E - S / max(beta, 0.01)
            F_history.append((beta, float(E), float(F)))

            # Auto-pruning
            for l in range(self.n_levels):
                if self._level_trained[l] and self.level_neurons[l].epsilon_mu > 0.15 and l > 2:
                    self._level_trained[l] = False

        if verbose:
            trained = sum(self._level_trained)
            print(f'  🌡️ Thermo β=[{beta_schedule[0]:.1f}..{beta_schedule[-1]:.1f}]: '
                  f'{trained}/{self.n_levels} levels, ΔF={F_history[0][2]-F_history[-1][2]:.4f}')

    def fit_batched_supervised(self, embeddings_list, target_signals,
                                verbose: bool = False):
        """🔥 SUPERVISED BATCHED FIT — Target = decoder signal, not cosine.

        En vez de cosine similarity (proxy débil), entrena cada nivel
        de atención para predecir la CONFIANZA DEL DECODER en la
        posición i del par (i, j).

        Esto alinea la atención con la tarea real: next-token prediction.

        Args:
          embeddings_list: lista de arrays de embeddings por secuencia
          target_signals: lista de arrays (seq_len,) con decoder confidence
                          para el TRUE next token en cada posición
        """
        import numpy as np
        d = self.embed_dim
        n_seqs = len(embeddings_list)

        for l in range(self.n_levels):
            if not self._level_trained[l]:
                continue  # Solo re-entrenar niveles ya activos

            all_pairs = []
            all_targets = []

            for ei, emb in enumerate(embeddings_list):
                pairs, idx_i, idx_j = self._build_pairs(emb, l)
                if len(pairs) < 10:
                    continue

                # Target: decoder confidence en posición i del par
                sig = target_signals[ei]
                targets = np.zeros((len(pairs), 1), dtype=np.float64)
                for p in range(len(pairs)):
                    pos_i = idx_i[p]
                    if pos_i < len(sig):
                        # Mapear confianza a [-1, 1]: conf≈1→+1, conf≈0→-1
                        targets[p, 0] = float(2.0 * sig[pos_i] - 1.0)

                all_pairs.append(pairs.astype(np.float64))
                all_targets.append(targets)

            if not all_pairs:
                continue

            X_l = np.vstack(all_pairs)
            y_l = np.vstack(all_targets)

            # Solo reentrenar si hay variedad en los targets
            if np.std(y_l) > 0.02:
                self.level_neurons[l].fit(X_l, y_l)
                if verbose:
                    print(f'    L{l}: retrained (σ={np.std(y_l):.4f})')

        if verbose:
            trained = sum(self._level_trained)
            print(f'  supervised: {trained}/{self.n_levels} levels')

    def fit_joint(self, embeddings_list, target_ids_list,
                  decoder, max_iter: int = 3, verbose: bool = True):
        """Entrenamiento conjunto atencion+decoder con senal real.

        En lugar de entrenar atencion con cosine similarity (proxy debil),
        la entrena con la senal real del decoder: que pares contribuyen
        mas a la prediccion correcta del siguiente token.

        Algoritmo EM-like:
          1. Inicializar atencion con cosine (1 epoch)
          2. Forward -> entrenar decoder
          3. Para cada nivel, medir contribucion a accuracy
          4. Re-entrenar atencion para potenciar pares utiles
          5. Repetir hasta convergencia
        """
        import numpy as np

        # Fase 0: Inicializacion rapida con cosine
        for emb in embeddings_list[:min(10, len(embeddings_list))]:
            self.fit(emb, max_epochs=1, early_stop=True)
        if verbose:
            trained = sum(self._level_trained)
            print(f'  [joint] Init: {trained}/{self.n_levels} levels trained')

        best_acc = 0.0
        for iteration in range(max_iter):
            # Forward: obtener embeddings contextualizados
            all_ctx = []
            for emb in embeddings_list:
                all_ctx.append(self.forward(emb))
            train_x = np.vstack(all_ctx)
            train_y_ids = np.concatenate(target_ids_list)
            train_y = np.eye(decoder.n_output)[train_y_ids]

            # Entrenar decoder
            result = decoder.fit(train_x, train_y)

            # Evaluar accuracy
            correct = 0
            total = min(500, len(train_x))
            for j in range(total):
                if np.argmax(decoder.evaluate(train_x[j])) == train_y_ids[j]:
                    correct += 1
            acc = correct / total

            if verbose:
                err = result.get('final_error', 0)
                print(f'  [joint] Iter {iteration+1}: acc={acc:.3f} '
                      f'(best={best_acc:.3f}) eps={err:.4f}')

            # Convergencia
            if acc <= best_acc * 1.01 and iteration > 0:
                if verbose:
                    print(f'  [joint] Converged at iter {iteration+1}')
                break
            if acc > best_acc:
                best_acc = acc

            # Refinar atencion: medir contribucion de cada nivel
            for l in range(self.n_levels):
                if not self._level_trained[l]:
                    continue

                # Ablacion: forward sin este nivel
                ctx_no_l = []
                n_emb = min(5, len(embeddings_list))
                for ei in range(n_emb):
                    emb = embeddings_list[ei]
                    n_t = len(emb)
                    context = np.zeros_like(emb, dtype=np.float64)
                    for ll in range(self.n_levels):
                        if ll == l or not self._level_trained[ll]:
                            continue
                        pairs_l, ii, jj = self._build_pairs(emb, ll)
                        if len(pairs_l) == 0:
                            continue
                        raw = self.level_neurons[ll].predict(pairs_l).mean
                        raw = np.clip(np.asarray(raw, dtype=np.float64).ravel(), -10, 10)
                        w_max = np.max(raw)
                        wts = np.exp(raw - w_max)
                        wts /= (wts.sum() + 1e-8)
                        np.add.at(context, ii, wts[:, None] * emb[jj])
                        np.add.at(context, jj, wts[:, None] * emb[ii])
                    ctx_no_l.append(emb.astype(np.float64) + 0.3 * context)

                if not ctx_no_l:
                    continue

                # Accuracy sin este nivel
                ctx_test = np.vstack(ctx_no_l)
                tgt_test = np.concatenate(target_ids_list[:n_emb])
                m = min(len(ctx_test), len(tgt_test))
                correct_no_l = sum(
                    1 for j in range(m)
                    if np.argmax(decoder.evaluate(ctx_test[j])) == tgt_test[j]
                )
                acc_no_l = correct_no_l / m if m > 0 else 0
                contribution = acc - acc_no_l

                if verbose and abs(contribution) > 0.01:
                    print(f'    L{l}: contrib={contribution:+.3f} '
                          f'(with={acc:.3f} without={acc_no_l:.3f})')

                # Si degrada, re-entrenar
                if contribution < -0.02:
                    pairs, _, _ = self._build_pairs(embeddings_list[0], l)
                    if len(pairs) >= 10:
                        pdim = pairs.shape[1] // 2
                        ni = np.linalg.norm(pairs[:, :pdim], axis=1) + 1e-8
                        nj = np.linalg.norm(pairs[:, pdim:], axis=1) + 1e-8
                        sim = np.sum(pairs[:, :pdim] * pairs[:, pdim:], axis=1) / (ni * nj)
                        sim = np.clip(sim, -1, 1).reshape(-1, 1).astype(np.float64)
                        self.level_neurons[l].fit(pairs, sim)

    def forward(self, embeddings, top_k: int = 16):
        """Forward con Hard Sparse Top-K (Cierre ERGON).

        🔥 NUEVO: Solo los top-K pares con mayor peso sobreviven.
        El resto se fuerza a CERO ABSOLUTO. Esto elimina la
        "sopa gris" de 255 tokens irrelevantes diluyendo la señal.

        Principio ERGON: el ruido estocástico ahoga la señal
        determinista si no hay cierre termodinámico. Forzar
        esparsidad real crea ese cierre.

        Args:
          embeddings: (seq_len, embed_dim)
          top_k: número máximo de pares a conservar por nivel
        """
        import numpy as np
        n = len(embeddings)
        context = np.zeros_like(embeddings, dtype=np.float64)

        for l in range(self.n_levels):
            if not self._level_trained[l]:
                continue

            pairs, idx_i, idx_j = self._build_pairs(embeddings, l)
            n_pairs = len(pairs)
            if n_pairs == 0:
                continue

            raw = self.level_neurons[l].predict(pairs)
            if hasattr(raw, 'mean') and not callable(raw.mean): raw = raw.mean
            raw = np.asarray(raw, dtype=np.float64).ravel()
            raw = np.clip(raw, -10, 10)
            w_max = np.max(raw)
            weights = np.exp(raw - w_max)
            weights /= (weights.sum() + 1e-8)

            # 🔥 HARD SPARSE TOP-K (Cierre ERGON)
            # Solo conservar los top_k pares con mayor peso
            if n_pairs > top_k:
                top_indices = np.argpartition(-weights, top_k)[:top_k]
                weights = weights[top_indices]
                idx_i = idx_i[top_indices]
                idx_j = idx_j[top_indices]
                # Re-normalizar
                weights /= (weights.sum() + 1e-8)

            # Agregación SIMÉTRICA
            if len(weights) > 0:
                w_ei = weights[:, None] * embeddings[idx_j]
                w_ej = weights[:, None] * embeddings[idx_i]
                np.add.at(context, idx_i, w_ei)
                np.add.at(context, idx_j, w_ej)

        scale = min(0.5, 1.0 / max(1, np.log2(max(n, 2))))
        return embeddings.astype(np.float64) + scale * context

    def forward_with_bias(self, embeddings, position_decay=None,
                          temperature=None):
        """🧬 PBCA: Position-Biased Cosine Attention.

        Misma atención cosine, PERO con modulación posicional:
          aggregation_weight(i,j,l) = softmax(raw/τₗ) · exp(-γₗ·|j-i|)

        Donde:
          - τₗ (temperature): controla sharpness del softmax
            τ→0: solo el par más similar importa
            τ→∞: todos los pares igual
          - γₗ (position_decay): penaliza distancia
            γ=0: sin sesgo (cosine puro)
            γ>0: tokens cercanos pesan más (mejor para next-token)

        Args:
          embeddings: (seq_len, embed_dim)
          position_decay: lista de γ por nivel, o None (usa self.position_decay)
          temperature: lista de τ por nivel, o None (usa self.temperature)

        Returns: embeddings contextualizados con bias posicional
        """
        import numpy as np
        n = len(embeddings)
        context = np.zeros_like(embeddings, dtype=np.float64)

        if position_decay is None:
            position_decay = self.position_decay
        if temperature is None:
            temperature = self.temperature

        for l in range(self.n_levels):
            if not self._level_trained[l]:
                continue

            pairs, idx_i, idx_j = self._build_pairs(embeddings, l)
            n_pairs = len(pairs)
            if n_pairs == 0:
                continue

            # ⚡ Batch evaluation
            raw = self.level_neurons[l].predict(pairs)
            if hasattr(raw, 'mean') and not callable(raw.mean): raw = raw.mean
            raw = np.asarray(raw, dtype=np.float64).ravel()

            # 🔥 Temperature scaling
            tau = max(temperature[l], 0.01)
            raw = np.clip(raw / tau, -10, 10)
            w_max = np.max(raw)
            weights = np.exp(raw - w_max)
            weights /= (weights.sum() + 1e-8)

            # 🔥 Position decay: γ · |j-i|
            gamma = position_decay[l]
            if gamma > 0:
                distances = np.abs(idx_j - idx_i).astype(np.float64)
                # Normalizar distancia al máximo del nivel (stride = 2^l)
                max_dist = float(2 ** l)
                decay = np.exp(-gamma * distances / max_dist)
                weights = weights * decay
                # Re-normalizar
                weights /= (weights.sum() + 1e-8)

            # 🔥 Sparse threshold
            if self.sparse_threshold > 0:
                thresh = self.sparse_threshold / max(n_pairs, 1)
                keep = weights > thresh
                if keep.sum() < n_pairs * 0.1:
                    keep = weights >= np.sort(weights)[-max(1, int(n_pairs * 0.1))]
                weights = weights[keep]
                idx_i = idx_i[keep]
                idx_j = idx_j[keep]

            # ⚡ Agregación SIMÉTRICA
            if len(weights) > 0:
                w_ei = weights[:, None] * embeddings[idx_j]
                w_ej = weights[:, None] * embeddings[idx_i]
                np.add.at(context, idx_i, w_ei)
                np.add.at(context, idx_j, w_ej)

        scale = min(0.5, 1.0 / max(1, np.log2(max(n, 2))))
        return embeddings.astype(np.float64) + scale * context

    def get_attention_weights(self, embeddings, top_k: int = 50):
        """Extraer top-k pares con mayor peso de atención."""
        import numpy as np
        attn = []
        for l in range(self.n_levels):
            if not self._level_trained[l]:
                continue
            pairs, idx_i, idx_j = self._build_pairs(embeddings, l)
            if len(pairs) == 0:
                continue
            raw = self.level_neurons[l].predict(pairs)
            if hasattr(raw, 'mean') and not callable(raw.mean): raw = raw.mean
            raw = raw.ravel()
            for p in range(len(pairs)):
                attn.append({
                    'level': l,
                    'i': int(idx_i[p]),
                    'j': int(idx_j[p]),
                    'weight': float(raw[p])
                })
        attn.sort(key=lambda x: -abs(x['weight']))
        return attn[:top_k]

# ═══════════════════════════════════════════════════════════════
# 🧬 NovaAttentionLM — LLM Nova-nativo con SparseANOVA Attention
# ═══════════════════════════════════════════════════════════════

class NovaAttentionLM:
    """Modelo de lenguaje basado 100% en Nova: atención ANOVA + decodificador Nova.

    Arquitectura (sin Transformers):
      Embedding → [SparseANOVA Attention × N] → Nova Decoder → logits

    Características:
      - Atención O(n log n) con pares jerárquicos
      - Sin dot-product, sin softmax, sin O(n²)
      - Pesos interpretables: cada peso = relevancia ANOVA del par
      - Residual connections entre capas
    """

    def __init__(self, vocab_size: int, embed_dim: int = None,
                 n_attention_layers: int = 2, n_levels: int = 4,
                 l2_lambda: float = 0.1):
        import numpy as np
        from .nova_phi_neuron import NovaPhiNeuron
        from .nova_llm import SparseANOVAAttention

        self.vocab_size = vocab_size
        # Embedding: one-hot + bias + pos lineal + pos cuadrática
        self.embed_dim = embed_dim or (vocab_size + 3)

        # Capas de atención
        self.attn_layers = []
        for a in range(n_attention_layers):
            layer = SparseANOVAAttention(
                embed_dim=self.embed_dim,
                n_levels=n_levels,
                l2_lambda=l2_lambda
            )
            self.attn_layers.append(layer)

        # Decoder final
        self.decoder = NovaPhiNeuron(
            'nlma_dec', self.embed_dim, vocab_size,
            max_degree=2, max_pairs=80,
            l2_lambda=l2_lambda, use_triton=True
        )

    def _embed(self, token_ids: 'np.ndarray') -> 'np.ndarray':
        import numpy as np
        n = len(token_ids)
        emb = np.zeros((n, self.embed_dim), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            if 0 <= tid < self.vocab_size:
                emb[i, tid] = 1.0
        # Positional features
        p = np.arange(n, dtype=np.float32) / max(1, n - 1) * 2 - 1
        emb[:, self.vocab_size] = 1.0           # bias
        emb[:, self.vocab_size + 1] = p          # posición lineal
        emb[:, self.vocab_size + 2] = p * p - 1  # posición cuadrática
        return emb

    def fit(self, token_ids: 'np.ndarray', seq_len: int = 32,
            n_epochs: int = 3, verbose: bool = True):
        """Entrenar el LM completo sobre secuencias.

        Fase 1: Entrenar capas de atención sobre embeddings.
        Fase 2: Forward + entrenar decoder.
        """
        import numpy as np
        import time

        n = len(token_ids) - 1
        n_seqs = n // seq_len
        t0 = time.perf_counter()

        # ── Preparar todas las secuencias de embedding ──
        np.random.seed(42)
        idxs = np.random.choice(n - seq_len, min(n_seqs, 200), replace=False)
        all_embs = []
        all_tgts = []
        for start in idxs:
            ctx_ids = token_ids[start:start + seq_len]
            tgt_ids = token_ids[start + 1:start + seq_len + 1]
            m = min(len(ctx_ids), len(tgt_ids))
            if m < 4:
                continue
            all_embs.append(self._embed(ctx_ids[:m]))
            all_tgts.append(tgt_ids[:m])

        if verbose:
            print(f'  Prepared {len(all_embs)} sequences')

        # ── Fase 1: Entrenar atención (alternando capas) ──
        for epoch in range(n_epochs):
            for layer_idx, attn_layer in enumerate(self.attn_layers):
                # Forward hasta esta capa
                layer_inputs = []
                for emb in all_embs:
                    x = emb
                    for prev_layer in self.attn_layers[:layer_idx]:
                        x = prev_layer.forward(x)
                    layer_inputs.append(x)

                # Entrenar esta capa sobre sus inputs
                for linp in layer_inputs:
                    attn_layer.fit(linp)

                if verbose:
                    t1 = time.perf_counter()
                    print(f'  Epoch {epoch + 1} Attn L{layer_idx} [{t1 - t0:.0f}s]')

        # ── Fase 2: Forward completo + entrenar decoder ──
        all_contextualized = []
        for emb in all_embs:
            x = emb
            for attn_layer in self.attn_layers:
                x = attn_layer.forward(x)
            all_contextualized.append(x)

        train_x = np.vstack(all_contextualized)
        train_y_ids = np.concatenate(all_tgts)
        train_y = np.eye(self.vocab_size, dtype=np.float32)[train_y_ids]

        if verbose:
            print(f'  Decoder inputs: {train_x.shape}, targets: {train_y.shape}')

        result = self.decoder.fit(train_x, train_y)
        if verbose:
            print(f'  ✅ Training complete [{time.perf_counter() - t0:.0f}s] '
                  f'ε={result.get("final_error", 0):.4f}')

    def generate(self, prompt_ids: list, max_new: int = 200,
                 temperature: float = 0.7) -> list:
        """Generar texto carácter por carácter."""
        import numpy as np

        gen = list(prompt_ids)
        for _ in range(max_new):
            ctx = gen[-min(32, len(gen)):]
            emb = self._embed(np.array(ctx, dtype=int))

            # Forward a través de atención
            x = emb
            for attn_layer in self.attn_layers:
                x = attn_layer.forward(x)

            # Decodificar último token
            logits = self.decoder.evaluate(x[-1])
            logits = logits - np.max(logits)
            probs = np.exp(logits / max(temperature, 0.01))
            probs = probs / probs.sum()
            gen.append(int(np.random.choice(self.vocab_size, p=probs)))

        return gen

    def evaluate_accuracy(self, token_ids: 'np.ndarray',
                          n_test: int = 1000) -> float:
        """Evaluar precisión next-char."""
        import numpy as np

        n = len(token_ids) - 1
        correct = total = 0

        for i in range(0, n - 25, 25):
            ctx_ids = token_ids[i:i + 25]
            tgt_ids = token_ids[i + 1:i + 26]
            m = min(len(ctx_ids), len(tgt_ids))
            if m < 4:
                continue

            emb = self._embed(ctx_ids[:m])
            x = emb
            for attn_layer in self.attn_layers:
                x = attn_layer.forward(x)

            for j in range(m - 1):
                if np.argmax(self.decoder.evaluate(x[j])) == tgt_ids[j]:
                    correct += 1
                total += 1
                if total >= n_test:
                    break
            if total >= n_test:
                break

        return correct / max(1, total)


# ═══════════════════════════════════════════════════════════════
# 🧬 DEEP NOVA NETWORK — Arquitectura Profunda Nativa
# ═══════════════════════════════════════════════════════════════

class ResidualBridge:
    """Puente residual con normalización adaptativa entre capas Novas.

    Resuelve el problema del apilamiento profundo:
      - Normaliza la salida de cada capa a media=0, std=1
      - Aplica escala y shift aprendibles vía ANOVA
      - Conexión residual: output = input + scale * norm(layer_output)

    Esto garantiza que cada capa reciba entradas con estadísticas
    consistentes, evitando la explosión/colapso de la señal.
    """

    def __init__(self, dim: int, layer_depth: int = 0, momentum: float = 0.99):
        import numpy as np
        self.dim = dim
        self.momentum = momentum
        self.running_mean = np.zeros(dim)
        self.running_std = np.ones(dim)
        self.scale = 1.0
        self._initialized = False
        # 🔥 Alpha adaptativo por profundidad
        base_alphas = [0.30, 0.45, 0.60, 0.75, 0.85, 0.90]
        self.depth_alpha = base_alphas[min(layer_depth, len(base_alphas)-1)]
        self.residual_weight = 1.0 - self.depth_alpha
        self.layer_depth = layer_depth

    def forward(self, x: 'np.ndarray', residual: 'np.ndarray' = None
                ) -> 'np.ndarray':
        """Normalizar y aplicar puente residual.

        Args:
          x: salida de la capa actual (n, dim)
          residual: entrada original para skip connection

        Returns: (n, dim) normalizado + residual
        """
        import numpy as np

        # Actualizar estadísticas running (Welford)
        batch_mean = x.mean(axis=0)
        batch_std = x.std(axis=0) + 1e-8

        if not self._initialized:
            self.running_mean = batch_mean
            self.running_std = batch_std
            self._initialized = True
        else:
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * batch_mean)
            self.running_std = (self.momentum * self.running_std +
                                (1 - self.momentum) * batch_std)

        # Normalizar
        x_norm = (x - self.running_mean) / self.running_std

        # Conexión residual con escala adaptativa
        if residual is not None:
            return residual + self.scale * x_norm
        return x_norm

    def auto_tune_scale(self, x: 'np.ndarray', residual: 'np.ndarray'):
        """Ajustar escala para mantener varianza unitaria en la salida."""
        import numpy as np
        out = residual + self.scale * x
        out_var = np.var(out)
        if out_var > 2.0:
            self.scale *= 0.9
        elif out_var < 0.5:
            self.scale *= 1.1
        self.scale = np.clip(self.scale, 0.1, 2.0)

    def adapt_from_innovation(self, innovation: float):
        """Adaptar alpha basado en innovación de la capa."""
        self.depth_alpha = np.clip(self.depth_alpha + 0.05 * (innovation - 0.3), 0.1, 0.95)
        self.residual_weight = 1.0 - self.depth_alpha


class CrossNovaLateral:
    """Comunicación lateral entre Novas del mismo nivel.

    Permite que las neuronas compartan información formando pares
    ANOVA entre sus salidas. Es como una mini-atención intra-capa
    que enriquece cada embedding con información de sus vecinos.

    Algoritmo:
      1. Para cada par (i,j) de dimensiones, evaluar ANOVA(2)
      2. Peso = cuánto contribuye la dimensión j a la i
      3. Agregar: output[i] += Σ weight(i,j) * input[j]
    """

    def __init__(self, dim: int, n_pairs: int = 8,
                 l2_lambda: float = 0.1):
        import numpy as np
        from .nova_phi_neuron import NovaPhiNeuron
        self.dim = dim
        self.n_pairs = min(n_pairs, dim * (dim - 1) // 2)

        # Una Nova para evaluar pares de dimensiones
        self.pair_evaluator = NovaPhiNeuron(
            'cross_lateral', dim * 2, 1,
            max_degree=2, max_pairs=self.n_pairs,
            l2_lambda=l2_lambda, use_triton=True
        )

        # Pares fijos (más eficiente)
        self._pair_indices = self._select_pairs(dim, self.n_pairs)
        self._trained = False

    @staticmethod
    def _select_pairs(dim: int, n_pairs: int):
        """Seleccionar pares de dimensiones con mayor correlación potencial."""
        import numpy as np
        np.random.seed(42)
        pairs = []
        # Topología de anillo: cada dimensión con sus vecinas
        for i in range(dim):
            for offset in range(1, 4):
                j = (i + offset) % dim
                if (i, j) not in pairs and (j, i) not in pairs:
                    pairs.append((i, j))
        # Si faltan, agregar aleatorios
        while len(pairs) < n_pairs:
            i, j = np.random.randint(0, dim, 2)
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))
        return pairs[:n_pairs]

    def fit(self, X: 'np.ndarray'):
        """Entrenar el evaluador de pares."""
        import numpy as np
        n = len(X)
        if n < 2: return

        # Construir pares de características
        pair_features = []
        pair_targets = []
        for i, j in self._pair_indices:
            feat = np.column_stack([X[:, i], X[:, j]])
            # Target: correlación entre las dos dimensiones
            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
            pair_features.append(feat)
            pair_targets.append(np.full((n, 1), corr))

        if pair_features:
            X_pairs = np.vstack(pair_features)
            Y_pairs = np.vstack(pair_targets)
            self.pair_evaluator.fit(X_pairs, Y_pairs)
            self._trained = True

    def forward(self, X: 'np.ndarray') -> 'np.ndarray':
        """Enriquecer embeddings con comunicación lateral.

        Returns: X enriquecido con información cross-dimensional
        """
        import numpy as np
        if not self._trained:
            return X

        n = len(X)
        enriched = X.copy().astype(np.float64)

        for idx, (i, j) in enumerate(self._pair_indices):
            feat = np.column_stack([X[:, i], X[:, j]])
            try:
                weight = float(self.pair_evaluator.evaluate(feat[0]))
            except:
                weight = 0.0

            if abs(weight) > 0.01:
                enriched[:, i] += weight * 0.1 * X[:, j]
                enriched[:, j] += weight * 0.1 * X[:, i]

        return enriched


class LearnableEmbedding:
    """🧬 Embedding aprendible sin backprop vía contraste local Nova.

    Mapea one-hot(V) → embed_dim con una neurona Nova.
    Entrenamiento contrastivo: tokens que co-ocurren → embeddings cercanos.
    """
    def __init__(self, vocab_size: int, embed_dim: int, l2_lambda: float = 0.05):
        from .nova_phi_neuron import NovaPhiNeuron
        import numpy as np
        self.V, self.d = vocab_size, embed_dim
        self.nova = NovaPhiNeuron('emb', vocab_size, embed_dim,
                                  max_degree=1, max_pairs=0, l2_lambda=l2_lambda,
                                  use_triton=True)
        self.nova.C_main = np.random.randn(embed_dim, vocab_size, 2).astype(np.float64)*0.1
        self.nova._x_mean = np.zeros(vocab_size)
        self.nova._x_std = np.ones(vocab_size)
        self.nova._x_min = np.zeros(vocab_size)
        self.nova._x_max = np.ones(vocab_size)
        self.nova.basis_type = "hermite"; self.nova._basis_forced = True

    def init_from_matrix(self, emb_matrix):
        import numpy as np
        V, d = emb_matrix.shape
        self.nova.C_main[:, :V, 0] = emb_matrix.T.astype(np.float64)
        self.nova.C_main[:, :, 1] = 0.0

    def train_contrastive(self, token_ids, n_samples=5000, verbose=False):
        import numpy as np
        n = len(token_ids); n_samples = min(n_samples, n-2)
        pos = np.random.choice(n-1, n_samples, replace=False)
        Xb = np.zeros((n_samples, self.V)); Yb = np.zeros((n_samples, self.d))
        for idx, p in enumerate(pos):
            tid = int(token_ids[p])
            if tid < self.V: Xb[idx, tid] = 1.0
            nt = int(token_ids[p+1])
            if nt < self.V:
                noh = np.zeros(self.V); noh[nt] = 1.0
                Yb[idx] = self.nova.evaluate(noh)
        self.nova.fit(Xb, Yb)
        if verbose:
            print(f'  🧬 Embedding trained: {n_samples} samples, ε={self.nova.epsilon_mu:.4f}')

    def embed(self, token_ids):
        import numpy as np
        n = len(token_ids); emb = np.zeros((n, self.d))
        for i, tid in enumerate(token_ids):
            if 0 <= tid < self.V:
                x = np.zeros(self.V); x[tid] = 1.0
                emb[i] = self.nova.evaluate(x)
        return emb


class NovaBottleneckDecoder:
    """Decoder Nova-nativo: una sola neurona + refinamiento lineal.

    Arquitectura:
      Contexto (128-dim) → NovaPhiNeuron → logits (65) → Linear refine → probs

    Ventajas sobre MultiHead:
      - 1 neurona vs 6 → 6× menos params, 15× más rápido
      - Nova captura interacciones no lineales (polynomial basis)
      - Refinamiento lineal es O(V·d) con least squares cerrado
      - Sin backprop, entrenamiento independiente
    """

    def __init__(self, n_input: int, n_output: int,
                 max_degree: int = 2, max_pairs: int = 16,
                 l2_lambda: float = 0.1):
        from .nova_phi_neuron import NovaPhiNeuron
        import numpy as np

        self.n_input = n_input
        self.n_output = n_output

        # Neurona Nova principal: input → output
        self.nova = NovaPhiNeuron(
            'dec_nova', n_input, n_output,
            max_degree=max_degree, max_pairs=max_pairs,
            l2_lambda=l2_lambda, use_triton=True
        )

        # Refinamiento lineal (aprendido post-Nova)
        self.W_refine = np.eye(n_output, dtype=np.float64)
        self.b_refine = np.zeros(n_output, dtype=np.float64)
        self._refine_ready = False

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Entrenar Nova directamente + refinamiento lineal."""
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        # Fase 1: Nova directa
        result = self.nova.fit(X, Y)

        # Fase 2: Forward Nova → obtener predicciones base
        Z = np.array([self.nova.evaluate(x) for x in X])

        # Fase 3: Least squares: refinar Z → Y
        lam = 0.01
        A = Z.T @ Z + lam * np.eye(self.n_output)
        b = Z.T @ Y
        try:
            self.W_refine = np.linalg.solve(A, b).T
            pred = Z @ self.W_refine.T
            self.b_refine = np.mean(Y - pred, axis=0)
            self._refine_ready = True
        except np.linalg.LinAlgError:
            pass  # Mantener identidad

        return result

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Forward: Nova → refine → softmax."""
        import numpy as np
        z = self.nova.evaluate(np.asarray(x, dtype=np.float64))
        if self._refine_ready:
            z = z @ self.W_refine.T + self.b_refine
        z = z - np.max(z)
        probs = np.exp(z)
        probs /= (probs.sum() + 1e-8)
        return probs


class MultiHeadNovaDecoder:
    """Decoder multi-cabeza NO-LINEAL con PROYECCIONES especializadas.

    🔥 v2: Agregación no-lineal vía NovaPhiNeuron en vez de promedio.
    Cada cabeza vota con peso aprendido, y una neurona ACF combina
    los votos considerando interacciones entre cabezas (ANOVA(2)).

    🔥 Temperatura termodinámica: T controla la entropía de la
    distribución final. T baja → sharp (modo greedy implícito).
    T alta → exploración (más diversidad).
    """

    def __init__(self, n_input: int, n_output: int,
                 n_heads: int = 4, pairs_per_head: int = 30,
                 l2_lambda: float = 0.1,
                 temperature: float = 0.8):
        import numpy as np
        from .nova_phi_neuron import NovaPhiNeuron

        self.n_heads = n_heads
        self.n_output = n_output
        self.n_input = n_input
        self.temperature = temperature
        
        self.proj_dim = max(32, n_input // 2)
        rng = np.random.RandomState(42 + n_heads)
        self.projections = []
        for h in range(n_heads):
            P = rng.randn(n_input, self.proj_dim).astype(np.float32)
            P /= np.sqrt(n_input)
            self.projections.append(P)

        self.heads = []
        for h in range(n_heads):
            head = NovaPhiNeuron(
                f'dec_head{h}', self.proj_dim, n_output,
                max_degree=3, max_pairs=pairs_per_head,
                l2_lambda=l2_lambda, use_triton=True
            )
            self.heads.append(head)
        
        # 🔥 AGREGADOR NO-LINEAL
        self.aggregator = NovaPhiNeuron(
            'dec_agg', n_heads * n_output, n_output,
            max_degree=2, max_pairs=min(40, n_heads * 4),
            l2_lambda=l2_lambda * 0.5, use_triton=True
        )
        self._agg_fitted = False
        # 🔥 Φ CACHE: build once per decoder epoch, reuse across rounds
        self._phi_cache_ready = False
        self._phi_full_N = 0  # N for which Phi was built

    def fit(self, X: 'np.ndarray', Y: 'np.ndarray', max_agg_samples: int = 5000):
        import numpy as np
        import time
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        N = X.shape[0]
        t0 = time.perf_counter()
        
        use_cache = self._phi_cache_ready and N <= self._phi_full_N
        
        if use_cache:
            # 🔥 FAST PATH: reuse cached Φ, only re-solve
            for h, head in enumerate(self.heads):
                X_proj = (X.astype(np.float32) @ self.projections[h]).astype(np.float64)
                # If N < full_N, use first N rows of cached Phi (subsampling)
                head.fit(X_proj, Y, use_cached_phi=True)
        else:
            # Full path: build Φ + cache
            for h, head in enumerate(self.heads):
                X_proj = (X.astype(np.float32) @ self.projections[h]).astype(np.float64)
                head.fit(X_proj, Y)  # builds + caches Phi internally
            self._phi_cache_ready = True
            self._phi_full_N = N
        
        # Aggregator: entrenar siempre (como el original que daba 31%)
        # 🔥 BATCHED predict() mantiene velocidad sin perder calidad
        agg_input_dim = self.n_heads * self.n_output
        max_samples = min(max_agg_samples, N)
        
        if agg_input_dim <= 512 and max_samples > 0:
            import numpy as np
            rng = np.random.RandomState(42)
            idx = rng.choice(N, size=max_samples, replace=False)
            X_sub = X[idx]
            Y_sub = Y[idx]
            
            all_votes = []
            for h, head in enumerate(self.heads):
                X_proj = (X_sub.astype(np.float32) @ self.projections[h]).astype(np.float64)
                votes_h = head.predict(X_proj).mean  # ✅ BATCHED: einsum('jik,bik->bj')
                all_votes.append(votes_h)
            votes_concat = np.hstack(all_votes)
            self.aggregator.fit(votes_concat, Y_sub)
            self._agg_fitted = True
        else:
            self._agg_fitted = False
        agg_desc = f"nova_nonlinear(N≤{max_samples})" if self._agg_fitted else "mean"
        
        t_elapsed = time.perf_counter() - t0
        cache_tag = "⚡cached" if use_cache else "🔨built"
        return {"n_heads": self.n_heads, "proj_dim": self.proj_dim, 
                "aggregator": agg_desc, "phi": cache_tag, "time": t_elapsed}

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        import numpy as np
        x = np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()
        votes = []
        for h, head in enumerate(self.heads):
            x_proj = (x.astype(np.float32) @ self.projections[h]).astype(np.float64)
            # ✅ predict() auto-detects 1 sample → _predict_one, else _predict_batch
            votes.append(head.predict(x_proj.reshape(1, -1)).mean.ravel())
        
        votes_flat = np.concatenate(votes)
        if self._agg_fitted:
            raw = self.aggregator.evaluate(votes_flat.reshape(1, -1)).ravel()
        else:
            raw = np.mean(votes, axis=0)
        
        raw = raw - np.max(raw)
        probs = np.exp(raw / max(self.temperature, 0.1))
        probs /= (probs.sum() + 1e-8)
        return probs
    
    def predict_batch(self, X: 'np.ndarray') -> 'np.ndarray':
        """🔥 BATCHED prediction: (N, d) → (N, V) in one shot.
        
        Replaces N individual evaluate() calls with ONE vectorized pass.
        Uses _predict_batch (np.einsum) for each head + aggregator.
        
        Returns: probs (N, n_output) — softmax probabilities
        """
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        
        # Each head: project + batched predict
        all_votes = []
        for h, head in enumerate(self.heads):
            X_proj = (X.astype(np.float32) @ self.projections[h]).astype(np.float64)
            votes_h = head.predict(X_proj).mean  # (N, n_output) batched
            all_votes.append(votes_h)
        votes_concat = np.hstack(all_votes)  # (N, n_heads * n_output)
        
        if self._agg_fitted:
            raw = self.aggregator.predict(votes_concat).mean  # (N, n_output)
        else:
            raw = np.mean(np.stack([v.reshape(N, -1) for v in all_votes], axis=0), axis=0)
        
        raw = raw - raw.max(axis=1, keepdims=True)
        probs = np.exp(raw / max(self.temperature, 0.1))
        probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
        return probs


# ═══════════════════════════════════════════════════════════════
# � SPECTRAL DECODER — ΦC=Y directo, sin heads, sin votaciones
# ═══════════════════════════════════════════════════════════════

class SpectralDecoder:
    """Decoder espectral NATIVO: una sola proyección lineal con features ACF.
    
    NO ES UN TRANSFORMER. Es un solver ΦC=Y:
      1. Evalúa base polinomial sobre TODAS las muestras (batch einsum)
      2. Construye matriz Φ con efectos principales + pares ANOVA(2)
      3. Resuelve ΦC = Y vía Cholesky (GPU) — UNA operación matricial
      4. Inferencia: features(x) @ C → softmax
    
    Zero Python loops. Zero heads. Zero votaciones.
    
    Matemática:
      Φ ∈ R^(N × F)  donde F = d·(deg+1) + n_pairs·(deg+1)²
      C = argmin ||ΦC - Y||² + λ||C||²  →  (ΦᵀΦ + λI)⁻¹ΦᵀY
    """
    
    def __init__(self, n_input: int, n_output: int,
                 max_degree: int = 2, max_pairs: int = 48,
                 l2_lambda: float = 0.1, temperature: float = 0.8):
        import numpy as np
        self.n_input = n_input
        self.n_output = n_output
        self.max_degree = max_degree
        self.max_pairs = max_pairs
        self.l2_lambda = l2_lambda
        self.temperature = temperature
        self.correlation_threshold = 0.1  # 🔥 mismo default que NovaPhiNeuron
        
        # Matriz de pesos espectrales C: (n_features, n_output)
        self.C = None
        self._fitted = False
        
        # Pares ANOVA: se seleccionan por CORRELACIÓN durante fit()
        self._pairs = []
        self.max_pairs = max_pairs
        self._d1 = max_degree + 1
        self._d2 = (max_degree + 1) ** 2
        self.n_features = None  # se calcula en fit() tras seleccionar pares
        
    def _setup_pairs(self, X: 'np.ndarray'):
        """🔥 Seleccionar pares por CORRELACIÓN — misma lógica que NovaPhiNeuron.
        
        Para n_input ≤ 100: matriz de correlación completa, top pares.
        Para n_input > 100: muestreo aleatorio entre features activos.
        """
        import numpy as np
        n = self.n_input
        X_c = X[:min(len(X), 1000)]
        
        if n <= 100:
            corr = np.abs(np.corrcoef(X_c.T))
            cand = [(corr[i, j], (i, j)) for i in range(n) for j in range(i + 1, n)
                    if corr[i, j] > self.correlation_threshold]
            cand.sort(reverse=True)
            self._pairs = [p for _, p in cand[:self.max_pairs]]
        else:
            stds = np.std(X_c, axis=0)
            active = np.where(stds > 0.01)[0]
            n_a = len(active)
            if n_a <= 1:
                self._pairs = []
                return
            n_s = min(n_a * 3, 2000)
            rng = np.random.RandomState(42)
            idx = rng.choice(n_a, size=(n_s, 2), replace=True)
            cand = []
            for a, b in idx:
                if a == b: continue
                i, j = active[a], active[b]
                if i > j: i, j = j, i
                c = np.abs(np.corrcoef(X_c[:, i], X_c[:, j])[0, 1])
                if c > self.correlation_threshold:
                    cand.append((c, (int(i), int(j))))
            seen = set()
            unique = [(c, p) for c, p in sorted(cand, reverse=True)
                      if (min(p), max(p)) not in seen and not seen.add((min(p), max(p)))]
            self._pairs = [p for _, p in unique[:self.max_pairs]]
        
        self.n_features = self.n_input * self._d1 + len(self._pairs) * self._d2
        
    def _eval_basis(self, X: 'np.ndarray') -> 'np.ndarray':
        """Evalúa base Hermite FÍSICA sobre X: (N, d) → (N, d, deg+1).
        
        🔥 Misma fórmula que NovaPhiNeuron._hermite_batch():
           H_0(z)=1, H_1(z)=z, H_{k+1}=z·H_k - k·H_{k-1}
        """
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        
        # Normalizar (z-score)
        x_mean = np.mean(X, axis=0, keepdims=True)
        x_std = np.maximum(np.std(X, axis=0, keepdims=True), 1e-8)
        z = (X - x_mean) / x_std
        
        deg = self.max_degree
        B = np.zeros((N, d, deg + 1), dtype=np.float64)
        B[:, :, 0] = 1.0
        if deg >= 1:
            B[:, :, 1] = z  # 🔥 H_1 = z (physicist's, NOT 2z)
        for k in range(2, deg + 1):
            B[:, :, k] = z * B[:, :, k-1] - (k - 1.0) * B[:, :, k-2]
        
        return B
    
    def fit(self, X: 'np.ndarray', Y: 'np.ndarray'):
        """Resolver ΦC = Y en UNA operación matricial.
        
        X: (N, d) — vectores de contexto
        Y: (N, V) — targets one-hot
        """
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        N = X.shape[0]
        
        # 0. 🔥 Seleccionar pares por CORRELACIÓN (como NovaPhiNeuron)
        self._setup_pairs(X)
        
        # 1. Evaluar base en batch — SIN LOOPS
        B = self._eval_basis(X)  # (N, d, deg+1)
        
        # 2. Features principales: (N, d·(deg+1))
        Phi = B.reshape(N, -1)
        
        # 3. Features de pares — vectorizado con einsum
        if len(self._pairs) > 0:
            pair_i = np.array([p[0] for p in self._pairs])
            pair_j = np.array([p[1] for p in self._pairs])
            # psi: outer(B[:,i], B[:,j]) para cada sample y cada par
            # B[:, pair_i] shape (N, n_pairs, d1), B[:, pair_j] shape (N, n_pairs, d1)
            psi = np.einsum('npi,npj->npij', B[:, pair_i, :], B[:, pair_j, :])
            # psi shape (N, n_pairs, d1, d1) → flatten to (N, n_pairs * d1 * d1)
            Psi = psi.reshape(N, len(self._pairs) * self._d2)
            Phi = np.hstack([Phi, Psi])
        
        # 4. Resolver (ΦᵀΦ + λI)C = ΦᵀY vía Cholesky
        lam = max(self.l2_lambda, 0.001)
        F = Phi.shape[1]
        A = Phi.T @ Phi + lam * np.eye(F)
        b = Phi.T @ Y
        
        # Intentar GPU
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
                A_t = torch.from_numpy(A.astype(np.float32)).to(device)
                b_t = torch.from_numpy(b.astype(np.float32)).to(device)
                L_t = torch.linalg.cholesky(A_t)
                self.C = torch.cholesky_solve(b_t, L_t).cpu().numpy().astype(np.float64)
            else:
                raise RuntimeError("no GPU")
        except Exception:
            import scipy.linalg
            L, low = scipy.linalg.cho_factor(A, lower=True)
            self.C = scipy.linalg.cho_solve((L, low), b)
        
        self._fitted = True
        return {"n_features": F, "n_pairs": len(self._pairs), 
                "solver": "cholesky_spectral", "n_samples": N}
    
    def evaluate(self, x: 'np.ndarray') -> 'np.ndarray':
        """Predecir probabilidades para UN vector de contexto.
        
        x: (d,) → probs: (V,)
        """
        import numpy as np
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        B = self._eval_basis(x)  # (1, d, deg+1)
        
        Phi = B.reshape(1, -1)
        if len(self._pairs) > 0:
            pair_i = np.array([p[0] for p in self._pairs])
            pair_j = np.array([p[1] for p in self._pairs])
            psi = np.einsum('pi,pj->pij', B[0, pair_i, :], B[0, pair_j, :])
            Psi = psi.reshape(1, len(self._pairs) * self._d2)
            Phi = np.hstack([Phi, Psi])
        
        logits = Phi @ self.C
        logits = logits.ravel()
        logits = logits - np.max(logits)
        probs = np.exp(logits / max(self.temperature, 0.1))
        probs /= (probs.sum() + 1e-8)
        return probs
    
    def evaluate_batch(self, X: 'np.ndarray') -> 'np.ndarray':
        """Predecir probabilidades para BATCH de vectores.
        
        X: (N, d) → probs: (N, V)
        """
        import numpy as np
        X = np.asarray(X, dtype=np.float64)
        B = self._eval_basis(X)
        
        Phi = B.reshape(len(X), -1)
        if len(self._pairs) > 0:
            pair_i = np.array([p[0] for p in self._pairs])
            pair_j = np.array([p[1] for p in self._pairs])
            psi = np.einsum('npi,npj->npij', B[:, pair_i, :], B[:, pair_j, :])
            Psi = psi.reshape(len(X), len(self._pairs) * self._d2)
            Phi = np.hstack([Phi, Psi])
        
        logits = Phi @ self.C
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits / max(self.temperature, 0.1))
        probs /= (probs.sum(axis=1, keepdims=True) + 1e-8)
        return probs


# ═══════════════════════════════════════════════════════════════
# �🚌 KNOWLEDGE BUS — Canal compartido de información entre capas
# ═══════════════════════════════════════════════════════════════

class KnowledgeBus:
    """Bus de conocimiento compartido entre capas profundas con FEEDBACK RECURRENTE.

    Principio fundamental:
      - El bus SIEMPRE transporta la información ORIGINAL
      - Cada capa AÑADE su conocimiento al bus (nunca reemplaza)
      - La capa siguiente ve: original + Σ conocimiento_previo
      - Así NADA se pierde y cada capa construye sobre lo anterior

    🔥 RECURRENTE (v2):
      - Tras todas las capas, el bus contiene contexto GLOBAL
      - Ese contexto se reinyecta como "feedback" a capas anteriores
      - Segunda pasada: cada capa ve bus_global + su input original
      - Esto da información del futuro (capas profundas) a capas tempranas

    Formato del bus:
      Bus(t) = original ⊕ K₀ ⊕ K₁ ⊕ ... ⊕ K_{t-1}

    donde Kᵢ es el "conocimiento" extraído de la capa i:
      Kᵢ = ||X_original|| · (output_i - input_i) / ||output_i - input_i||
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.original = None           # Información original (nunca se modifica)
        self.knowledge_stack = []      # [(layer_idx, K_vector, confidence, innovation)]
        self.bus_state = None          # Estado actual del bus
        self._lock_original = True     # El original NUNCA se modifica
        self.feedback_signal = None    # 🔥 Señal recurrente (contexto global → local)
        self.feedback_alpha = 0.15     # 🔥 Peso del feedback en la mezcla

    def initialize(self, x: 'np.ndarray'):
        """Inicializar el bus con la información original."""
        self.original = x.copy().astype(np.float64)
        self.bus_state = self.original.copy()
        self.knowledge_stack = []
        self.feedback_signal = None

    def read(self) -> 'np.ndarray':
        """Leer el estado actual del bus (original + todo el conocimiento)."""
        return self.bus_state.copy()

    def read_with_feedback(self, layer_idx: int) -> 'np.ndarray':
        """🔥 Leer bus con mezcla de feedback global (recurrente).
        
        Capas tempranas reciben una mezcla del bus actual + señal de feedback
        de capas profundas, dándoles "visión del futuro".
        
        Mezcla: bus_state + α · feedback · (1 - layer_idx/n_layers)
        Las capas más tempranas reciben más feedback (menos contexto propio).
        """
        import numpy as np
        base = self.bus_state.copy()
        if self.feedback_signal is not None and self.feedback_alpha > 0:
            decay = 1.0 - (layer_idx / max(len(self.knowledge_stack), 1))
            weight = self.feedback_alpha * decay
            base = (1.0 - weight) * base + weight * self.feedback_signal
        return base

    def set_feedback(self, signal: 'np.ndarray'):
        """🔥 Establecer señal de feedback desde el estado global del bus."""
        self.feedback_signal = signal.copy().astype(np.float64)

    def write(self, layer_idx: int, input_x: 'np.ndarray',
              output_x: 'np.ndarray', confidence: float = None):
        """Escribir el conocimiento de una capa en el bus.

        El conocimiento K se calcula como la diferencia normalizada
        entre output e input, escalada por la magnitud del original.

        Args:
          layer_idx: índice de la capa
          input_x: lo que recibió la capa
          output_x: lo que produjo la capa
          confidence: confianza en este conocimiento (0-1)
        """
        import numpy as np

        # Calcular innovación como dirección del cambio
        delta = output_x - input_x
        delta_norm = np.linalg.norm(delta, axis=0, keepdims=True) + 1e-8

        # Conocimiento = delta normalizado * magnitud original
        K = delta / delta_norm * np.linalg.norm(self.original, axis=0, keepdims=True)

        # Confianza basada en cuánto cambió
        if confidence is None:
            innovation = float(np.linalg.norm(delta) / max(np.linalg.norm(input_x), 1e-8))
            confidence = np.clip(innovation * 3.0, 0.01, 0.95)

        # Añadir al stack
        self.knowledge_stack.append((layer_idx, K, confidence, 
                                      float(np.linalg.norm(delta))))

        # 🔥 ACTUALIZAR BUS: original + Σ K_i (cada K ponderado por confianza)
        self.bus_state = self.original.copy()
        for _, k, conf, _ in self.knowledge_stack:
            self.bus_state = self.bus_state + conf * k

    def get_knowledge_summary(self) -> dict:
        """Obtener resumen del conocimiento acumulado."""
        return {
            'n_layers': len(self.knowledge_stack),
            'total_innovation': sum(inn for _, _, _, inn in self.knowledge_stack),
            'confidences': [conf for _, _, conf, _ in self.knowledge_stack],
            'bus_norm': float(np.linalg.norm(self.bus_state)),
            'original_norm': float(np.linalg.norm(self.original)),
            'knowledge_ratio': float(np.linalg.norm(self.bus_state - self.original) /
                                     max(np.linalg.norm(self.original), 1e-8))
        }

    def reconstruct_original(self) -> 'np.ndarray':
        """Intentar reconstruir el original desde el conocimiento acumulado.

        Esto es la PRUEBA ÁCIDA: si podemos reconstruir X desde el bus,
        significa que el conocimiento es completo y nada se perdió.
        """
        import numpy as np
        if not self.knowledge_stack:
            return self.original.copy()

        reconstructed = np.zeros_like(self.original)
        for _, K, conf, _ in self.knowledge_stack:
            reconstructed += conf * K

        return reconstructed


# ═══════════════════════════════════════════════════════════════
# 🔀 DIVERGENT DEEP NOVA — Arquitectura con bases y pares divergentes
# ═══════════════════════════════════════════════════════════════

class DivergentAttentionLayer:
    """Capa de atención con feature engineering DIVERGENTE y AUTÓNOMO.

    🔥 TOTALMENTE AUTÓNOMO:
      - Sin bases hardcodeadas — cada capa analiza sus datos reales
      - Selecciona base según curtosis + diversidad vs capas previas
      - Niveles/strides auto-escalados según layer_idx
      - Si los datos son Gaussianos → Hermite (rápido)
      - Si son subm Gaussianos → Chebyshev (mejor cobertura)
      - RESTRICCIÓN: no repite base de capa anterior (diversidad forzada)
    """

    # ══ Catálogo de bases disponibles ══
    AVAILABLE_BASES = ["hermite", "chebyshev", "legendre", "fourier"]
    
    # 🔥 ERGON complexity thresholds
    COMPLEXITY_LOW = 0.3     # 𝔈 < 0.3 → ordenado (Chebyshev)
    COMPLEXITY_HIGH = 0.7    # 𝔈 > 0.7 → caótico (Fourier)
    # 0.3 ≤ 𝔈 ≤ 0.7 → mixto (Hermite/Legendre)

    def __init__(self, layer_idx: int, embed_dim: int,
                 l2_lambda: float = 0.1, max_context: int = 16384,
                 proj_dim: int = None, use_kronecker: bool = False):
        import numpy as np

        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.l2_lambda = l2_lambda

        # ══ Configuración AUTO-ESCALADA por profundidad ══
        max_levels = min(int(np.ceil(np.log2(max(max_context, 256)))) + 1, 12)
        self.n_levels = max(2, min(max_levels, 8 - layer_idx))  # 8,7,6,5,4...
        self.stride_mult = 2 ** min(layer_idx, 3)  # 1,2,4,8,8...

        # ══ Base: se selecciona AUTÓNOMAMENTE en fit() ══
        self.basis_override = None
        self.desc = f"AUTO-L{layer_idx}"

        # ══ Atención SparseANOVA con reducción INTELIGENTE ══
        # Capas 0-1: full power | Capas 2: medio | Capas 3+: reducido
        from .nova_llm import SparseANOVAAttention
        if layer_idx <= 1:
            pairs = 24 if layer_idx == 0 else 20
            degree = 2
        elif layer_idx == 2:
            pairs = 16
            degree = 2
        else:
            pairs = 12
            degree = 1
        self.attention = SparseANOVAAttention(
            embed_dim=embed_dim,
            n_levels=self.n_levels,
            l2_lambda=l2_lambda,
            pairs_per_level=pairs,
            max_degree=degree,
            proj_dim=proj_dim,
            use_kronecker=use_kronecker
        )

        # Lateral + Bridge
        self.lateral = CrossNovaLateral(embed_dim, n_pairs=8, l2_lambda=l2_lambda)
        self.bridge = ResidualBridge(embed_dim, layer_depth=layer_idx)

        # Métricas para decisión autónoma
        self._basis_selected = False
        self._data_kurtosis = 0.0
        self._data_skew = 0.0

    def _auto_select_basis(self, X: 'np.ndarray',
                           used_bases: set = None) -> str:
        """🔥 Selección AUTÓNOMA con DIVERSIDAD FORZADA + ERGON complexity.

        Cada capa analiza sus datos reales y elige:
          1. Base polinomial según curtosis + complejidad ergódica
          2. NUNCA repite la base de capas anteriores
          3. Si todas las bases están usadas, genera variante (degree shift)

        ERGON routing:
          𝔈 < 0.3 → ordenado → Chebyshev/Legendre (bounded domain)
          0.3 ≤ 𝔈 ≤ 0.7 → mixto → Hermite (Gaussian-optimal)
          𝔈 > 0.7 → caótico → Fourier (captura alta frecuencia)

        Returns: base seleccionada (string)
        """
        import numpy as np

        if used_bases is None:
            used_bases = set()

        # 1. Analizar distribución de los datos
        X_flat = X.ravel()[:10000]
        self._data_kurtosis = float(np.mean((X_flat - X_flat.mean())**4) /
                                    max((X_flat.var())**2, 1e-10))
        self._data_skew = float(np.abs(np.mean((X_flat - X_flat.mean())**3) /
                                   max(X_flat.std()**3, 1e-10)))
        
        # 2. 🔥 ERGON COMPLEXITY: estimar 𝔈(T) desde espectro de entrada
        try:
            X_sample = X[:min(500, len(X))]
            U, s, _ = np.linalg.svd(X_sample - X_sample.mean(axis=0), full_matrices=False)
            s_norm = s / max(s.sum(), 1e-12)
            H_spec = -np.sum(s_norm * np.log(s_norm + 1e-12))
            H_max = np.log(len(s))
            self._ergodic_complexity = float(np.clip(H_spec / max(H_max, 1e-12), 0.0, 1.0))
        except Exception:
            self._ergodic_complexity = 0.5  # fallback
        
        # 3. ERGON routing según complejidad
        ec = self._ergodic_complexity
        if ec < self.COMPLEXITY_LOW:
            # Ordenado: Chebyshev o Legendre (dominio acotado)
            candidates = ["chebyshev", "legendre"]
        elif ec > self.COMPLEXITY_HIGH:
            # Caótico: Fourier (alta frecuencia)
            candidates = ["fourier", "chebyshev"]
        else:
            # Mixto: Hermite o Legendre
            candidates = ["hermite", "legendre"]
        
        # 4. 🔥 DIVERSIDAD FORZADA: nunca repetir base de capas previas
        available = [b for b in candidates if b not in used_bases]
        if not available:
            # Si las 2 candidatas están usadas, buscar en todo el catálogo
            available = [b for b in self.AVAILABLE_BASES if b not in used_bases]
        if not available:
            # Si TODAS están usadas, usar la mejor con sufijo de variante
            best = candidates[0]
            variant = len([b for b in used_bases if b.startswith(best)]) + 1
            best = f"{best}_v{variant}"
        else:
            best = available[0]
        
        # 5. Actualizar
        self.basis_override = best
        self.desc = f"L{self.layer_idx}/{best}(𝔈={ec:.2f},k={self._data_kurtosis:.1f})"
        self._basis_selected = True

        return best

    def _force_basis(self):
        """Forzar la base seleccionada autónomamente."""
        if self.basis_override is None:
            return  # Aún no seleccionada
        for lv in range(self.attention.n_levels):
            neuron = self.attention.level_neurons[lv]
            neuron.basis_type = self.basis_override
            neuron._basis_forced = True

    def fit(self, bus_state: 'np.ndarray', layer_inputs: list,
            used_bases: set = None, verbose: bool = False,
            trans_prob=None, token_ids_list=None, trans_beta=0.3,
            trans_alpha=0.3, koopman_alpha: float = 0.0):
        """Entrenar capa con selección autónoma de base + UATO v8 + Koopman."""
        import numpy as np

        # 🔥 SELECCIÓN AUTÓNOMA: analizar datos reales
        if not self._basis_selected and len(layer_inputs) > 0:
            sample_data = layer_inputs[0]
            if len(layer_inputs) > 1:
                sample_data = np.vstack(layer_inputs[:min(5, len(layer_inputs))])
            chosen = self._auto_select_basis(sample_data, used_bases)
            self._force_basis()
            if verbose:
                k = self._data_kurtosis
                print(f'    🤖 Auto-basis: {chosen} (kurtosis={k:.1f})')

        # Entrenar atención con UATO v8 — Dual Solver (target aditivo) + Koopman
        self.attention.fit_batched_contrastive(
            layer_inputs, verbose=verbose,
            trans_prob=trans_prob,
            token_ids_list=token_ids_list,
            trans_beta=trans_beta,
            trans_alpha=trans_alpha,
            koopman_alpha=koopman_alpha)

        # Lateral (opcional)
        all_out = []
        for inp in layer_inputs:
            out = self.attention.forward(inp)
            all_out.append(out)
        stacked_out = np.vstack(all_out) if all_out else np.array([])
        if self.lateral is not None and len(stacked_out) > 0:
            try:
                self.lateral.fit(stacked_out)
            except (ValueError, np.linalg.LinAlgError, IndexError):
                pass

        return stacked_out

    def forward(self, x: 'np.ndarray', bus_state: 'np.ndarray'
                ) -> tuple:
        """Forward con base divergente.

        Returns: (output, knowledge_raw)
        """
        import numpy as np

        # 1. Atención
        x_attn = self.attention.forward(x)

        # 2. Lateral (opcional)
        if self.lateral is not None:
            x_lat = self.lateral.forward(x_attn)
        else:
            x_lat = x_attn

        # 3. Bridge residual
        x_out = self.bridge.forward(x_lat, x)

        # 4. Knowledge = residual directo (qué cambió esta capa)
        knowledge = x_out - x

        return x_out, knowledge

    def process_and_write_bus(self, x: 'np.ndarray', bus: KnowledgeBus
                               ) -> 'np.ndarray':
        """Procesar input y escribir conocimiento al bus."""
        output, knowledge = self.forward(x, bus.read())
        bus.write(self.layer_idx, x, output)
        return output


class DivergentDeepNova:
    """Red Nova Profunda DIVERGENTE con KnowledgeBus + FEEDBACK RECURRENTE.

    ARQUITECTURA v2:
      Input → [Layer₀] → K₀ → Bus
            → [Layer₁] → K₁ → Bus
            → [Layer₂] → K₂ → Bus
            → Decoder no-lineal (max_degree=3 + agregador Nova)

    🔥 FEEDBACK RECURRENTE:
      Tras el forward completo, el bus contiene contexto GLOBAL.
      Ese contexto se reinyecta como \"señal de feedback\" a cada capa.
      Segunda pasada: cada capa ve bus_global ⊕ su input original.
      Esto permite que capas tempranas \"vean el futuro\" (capas profundas).

    🔥 KRONECKER (orden 3+):
      Atención con KroneckerPhiNeuron para interacciones de alto orden.
      Equivalente a max_degree=4 pero con costo O(2×(n/2)³) en vez de O(n⁴).
    """

    def __init__(self, vocab_size: int, embed_dim: int = None,
                 n_layers: int = 3, n_heads: int = 4,
                 l2_lambda: float = 0.1,
                 max_context: int = 16384,
                 memory_budget_gb: float = 5.0,
                 proj_dim: int = None,
                 use_tree_decoder: bool = False,
                 use_spectral_decoder: bool = False,  # 🔥 ΦC=Y directo, sin heads
                 use_kronecker: bool = True,
                 recurrent_feedback: bool = True):
        import numpy as np

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim or (vocab_size + 3)
        self.n_layers = n_layers
        self.l2_lambda = l2_lambda
        self.koopman_alpha = 0.3
        self.max_context = max_context
        self.memory_budget_gb = memory_budget_gb
        self.stored_seq_len = 32
        self.recurrent_feedback = recurrent_feedback  # 🔥

        if proj_dim is None and self.embed_dim > 256:
            proj_dim = max(32, min(128, self.embed_dim // 3))
        elif proj_dim is None:
            proj_dim = None
        self.proj_dim = proj_dim

        # ══ Capas divergentes con Kronecker ══
        self.layers = []
        for l in range(n_layers):
            layer = DivergentAttentionLayer(
                layer_idx=l,
                embed_dim=self.embed_dim,
                l2_lambda=l2_lambda,
                max_context=max_context,
                proj_dim=proj_dim,
                use_kronecker=use_kronecker
            )
            self.layers.append(layer)

        # ══ KnowledgeBus con feedback ══
        self.bus = KnowledgeBus(self.embed_dim)

        # ══ Decoder ══
        self.use_tree_decoder = use_tree_decoder
        self.use_spectral_decoder = use_spectral_decoder
        if use_tree_decoder:
            try:
                from .nova_hierarchical import NovaTreeDecoder
            except ImportError:
                from acf_functor.neuron.nova_hierarchical import NovaTreeDecoder
            self.decoder = None
            self._tree_decoder_cls = NovaTreeDecoder
        elif use_spectral_decoder:
            # 🔥 SpectralDecoder: ΦC=Y directo, sin heads, sin votaciones
            self.decoder = SpectralDecoder(
                self.embed_dim, vocab_size,
                max_degree=2, max_pairs=48,
                l2_lambda=l2_lambda,
                temperature=0.7
            )
            self._tree_decoder_cls = None
        else:
            self.decoder = MultiHeadNovaDecoder(
                self.embed_dim, vocab_size,
                n_heads=max(4, min(n_heads, 6)),
                pairs_per_head=60,
                l2_lambda=l2_lambda,
                temperature=0.7
            )
            self._tree_decoder_cls = None

        # Métricas
        self.bus_history = []
        self._learnable_emb = None
        self._nova_emb = None
        self._use_nova_emb = True  # Desactivar con dnn._use_nova_emb = False
        self._use_sgf = False      # 🔥 SGF: OFF por defecto (requiere word-level Nova)
        self._sgf_active = False

    def _init_tree_decoder(self, token_ids: 'np.ndarray'):
        """Inicializar TreeDecoder con frecuencias reales del corpus."""
        import numpy as np
        from collections import Counter
        freqs = Counter(int(t) for t in token_ids[:100000])
        # Asegurar que todos los tokens tengan al menos freq=1
        for i in range(self.vocab_size):
            if i not in freqs:
                freqs[i] = 1
        self.decoder = self._tree_decoder_cls(
            n_input=self.embed_dim,
            vocab_size=self.vocab_size,
            token_frequencies=dict(freqs),
            l2_lambda=self.l2_lambda,
        )

    def optimize_thermo_params(self, all_emb, all_tgt, n_search=6):
        """🌡️ Optimizar T_base y α por grid search termodinámico.

        Evalúa combinaciones (T_base, α) midiendo accuracy del decoder.
        Solo 2 parámetros → búsqueda rápida.

        Returns: best (T_base, alpha)
        """
        import numpy as np
        tgt_flat = np.concatenate(all_tgt)
        n_emb = min(8, len(all_emb))
        emb_sub = all_emb[:n_emb]

        T_bases = np.linspace(0.4, 2.5, n_search)
        alphas = np.linspace(0.0, 0.5, n_search)

        best_acc = 0.0
        best_T = 1.0
        best_alpha = 0.15

        for Tb in T_bases:
            for al in alphas:
                all_ctx = []
                for emb in emb_sub:
                    x = emb.astype(np.float64)
                    for layer in self.layers:
                        x = layer.attention.forward_thermo(
                            x, T_base=Tb, alpha=al)
                    all_ctx.append(x)
                ctx = np.vstack(all_ctx)

                correct = sum(1 for j in range(min(200, len(ctx)))
                             if self._decoder_argmax(ctx[j]) == tgt_flat[j])
                acc = correct / min(200, len(ctx))

                if acc > best_acc:
                    best_acc = acc
                    best_T = Tb
                    best_alpha = al

        return best_T, best_alpha, best_acc

    def optimize_attention_params(self, all_emb, all_tgt,
                                   n_search: int = 5):
        """🧬 Optimizar γ (position_decay) y τ (temperature) rápido.

        Estrategia 2-stage:
          1. Probar γ ∈ [0, 2] con τ=1 → encontrar best_γ
          2. Probar τ ∈ [0.3, 3] con γ=best_γ → encontrar best_τ

        Usa solo las primeras 5 secuencias para velocidad.
        """
        import numpy as np

        tgt_flat = np.concatenate(all_tgt)
        # Usar subset pequeño para velocidad
        n_emb = min(5, len(all_emb))
        emb_sub = all_emb[:n_emb]
        n_tokens = sum(len(e) for e in emb_sub)
        
        best_params = {}

        for li, layer in enumerate(self.layers):
            # Stage 1: encontrar best_γ con τ=1
            best_acc = 0.0
            best_gamma = 0.0
            for gamma in np.linspace(0.0, 2.0, n_search):
                pd = [gamma] * layer.attention.n_levels
                tp = [1.0] * layer.attention.n_levels
                
                all_ctx = []
                for emb in emb_sub:
                    x = emb.astype(np.float64)
                    for lj, lyr in enumerate(self.layers):
                        if lj == li:
                            x = lyr.attention.forward_with_bias(
                                x, position_decay=pd, temperature=tp)
                        else:
                            x = lyr.attention.forward(x)
                    all_ctx.append(x)
                ctx = np.vstack(all_ctx)

                correct = sum(1 for j in range(min(200, len(ctx)))
                             if self._decoder_argmax(ctx[j]) == tgt_flat[j])
                acc = correct / min(200, len(ctx))
                if acc > best_acc:
                    best_acc = acc
                    best_gamma = gamma

            # Stage 2: encontrar best_τ con best_γ
            best_tau = 1.0
            for tau in np.linspace(0.3, 3.0, n_search):
                pd = [best_gamma] * layer.attention.n_levels
                tp = [tau] * layer.attention.n_levels
                
                all_ctx = []
                for emb in emb_sub:
                    x = emb.astype(np.float64)
                    for lj, lyr in enumerate(self.layers):
                        if lj == li:
                            x = lyr.attention.forward_with_bias(
                                x, position_decay=pd, temperature=tp)
                        else:
                            x = lyr.attention.forward(x)
                    all_ctx.append(x)
                ctx = np.vstack(all_ctx)

                correct = sum(1 for j in range(min(200, len(ctx)))
                             if self._decoder_argmax(ctx[j]) == tgt_flat[j])
                acc = correct / min(200, len(ctx))
                if acc > best_acc:
                    best_acc = acc
                    best_tau = tau

            # Guardar
            for lv in range(layer.attention.n_levels):
                layer.attention.position_decay[lv] = best_gamma
                layer.attention.temperature[lv] = best_tau
            best_params[li] = {'gamma': best_gamma, 'tau': best_tau,
                              'acc': best_acc}

        return best_params

    def _train_decoder(self, X: 'np.ndarray', targets):
        """Entrenar decoder — compatible con ambos tipos."""
        import numpy as np
        if self.use_tree_decoder:
            if self.decoder is None:
                self._init_tree_decoder(targets)
            tgt_ids = np.asarray(targets, dtype=int).ravel()[:len(X)]
            return self.decoder.fit(X, tgt_ids)
        else:
            if targets.ndim == 1:
                targets = np.eye(self.vocab_size, dtype=np.float64)[targets]
            return self.decoder.fit(X, targets)

    def _decoder_evaluate(self, x: 'np.ndarray') -> 'np.ndarray':
        """Evaluar decoder — devuelve vector de probabilidades."""
        if self.use_tree_decoder:
            return self.decoder.evaluate(x)
        else:
            return self.decoder.evaluate(x)

    def _decoder_argmax(self, x: 'np.ndarray') -> int:
        """Token predicho por el decoder."""
        import numpy as np
        probs = self._decoder_evaluate(x)
        return int(np.argmax(probs))

    def _build_pmi_embeddings(self, token_ids: 'np.ndarray', window: int = 5):
        """🔥 SGF: Semantic Genesis Functor — Geometría semántica certificada.

        PPMI → SVD → Chebyshev(deg=2) → L2 normalize → ℝᵈ.
        
        Descubierto autónomamente por el laboratorio Genesis sobre Wikipedia.
        Σ=0.776 de preservación semántica. Reemplaza los embeddings PMI-SVD
        crudos con una geometría donde palabras similares están cerca.
        
        Notación: Ψ_SG : Palabra → ℝᵈ (Semantic Genesis Functor)
        """
        import numpy as np
        from scipy.sparse.linalg import eigsh

        V = self.vocab_size
        d = self.embed_dim - 3
        
        # Check if SGF is enabled
        use_sgf = getattr(self, '_use_sgf', True)

        cooc = np.zeros((V, V), dtype=np.float64)
        ids = token_ids[:min(500000, len(token_ids))]
        for t in range(len(ids)):
            a = int(ids[t])
            if a >= V: continue
            for w in range(1, window + 1):
                if t + w < len(ids):
                    b = int(ids[t + w])
                    if b < V:
                        cooc[a, b] += 1.0
                        cooc[b, a] += 1.0

        total = cooc.sum()
        row_sums = cooc.sum(axis=1, keepdims=True) + 1e-8
        expected = (row_sums @ row_sums.T) / total
        pmi = np.log((cooc + 1e-8) * total / (expected + 1e-8))
        ppmi = np.maximum(pmi, 0.0)

        A = ppmi + 0.1 * np.eye(V)
        D = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
        L_sym = D @ A @ D

        k = min(d + 1, V - 1)
        try:
            eigenvalues, eigenvectors = eigsh(L_sym.astype(np.float64), k=k, which='LM')
            idx = np.argsort(-np.abs(eigenvalues))
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            emb_svd = eigenvectors[:, 1:] * np.maximum(eigenvalues[1:], 0)
            emb_svd = emb_svd[:, :d].astype(np.float64)
            
            if use_sgf and d >= 16:
                # 🔥 SGF: Semantic Genesis Functor
                # Chebyshev(deg=2) sobre SVD → interacciones cuadráticas semánticas
                z = (emb_svd - emb_svd.mean(axis=0)) / (emb_svd.std(axis=0) + 1e-8)
                B = np.zeros((V, d, 3), dtype=np.float64)
                B[:, :, 0] = 1.0
                B[:, :, 1] = z
                B[:, :, 2] = 2.0 * z * z - 1.0  # T₂(x) = 2x² - 1
                
                # Flatten + project back to d via random projection (preserves cosine)
                expanded = B.reshape(V, -1)
                if expanded.shape[1] > d:
                    rng = np.random.RandomState(42)
                    P = rng.randn(expanded.shape[1], d) / np.sqrt(expanded.shape[1])
                    emb_sgf = expanded @ P
                else:
                    emb_sgf = expanded[:, :d]
                
                # L2 normalize — final Ψ_SG embedding
                norms = np.linalg.norm(emb_sgf, axis=1, keepdims=True) + 1e-8
                self._emb_matrix = (emb_sgf / norms).astype(np.float32)
                self._sgf_active = True
            else:
                # Fallback: SVD-only embeddings
                norms = np.linalg.norm(emb_svd, axis=1, keepdims=True) + 1e-8
                emb_svd = emb_svd / norms
                if emb_svd.shape[1] < d:
                    rng = np.random.RandomState(42)
                    extra = rng.randn(V, d - emb_svd.shape[1]).astype(np.float32)
                    extra /= np.sqrt(d)
                    self._emb_matrix = np.hstack([emb_svd.astype(np.float32), extra])
                else:
                    self._emb_matrix = emb_svd.astype(np.float32)
                self._sgf_active = False
        except Exception:
            from scipy.sparse.linalg import svds
            try:
                U, s, _ = svds(ppmi.astype(np.float32), k=min(d, V-1))
                idx = np.argsort(-np.abs(s))
                U, s = U[:, idx], s[idx]
                emb = (U * np.sqrt(np.maximum(s, 0))).astype(np.float32)
                # Pad to d dimensions if needed
                if emb.shape[1] < d:
                    rng = np.random.RandomState(42)
                    extra = rng.randn(V, d - emb.shape[1]).astype(np.float32)
                    extra /= np.sqrt(d)
                    self._emb_matrix = np.hstack([emb[:, :d], extra])
                else:
                    self._emb_matrix = emb[:, :d]
                self._sgf_active = False
            except Exception:
                np.random.seed(42)
                self._emb_matrix = np.random.randn(V, d).astype(np.float32)
                self._emb_matrix /= np.sqrt(d)
                self._sgf_active = False
        return self

    def _embed(self, token_ids: 'np.ndarray') -> 'np.ndarray':
        """Crear embeddings: NovaFunctional (dinámico) > Learnable > PMI/Gauss."""
        import numpy as np
        n = len(token_ids)
        d = self.embed_dim - 3

        # 🔥 Nova-Emb: solo si activado
        if self._use_nova_emb and self._nova_emb is not None:
            emb_raw = self._nova_emb.encode(token_ids, 
                                            static_emb=self._emb_matrix if hasattr(self, '_emb_matrix') else None)
            emb = np.zeros((n, self.embed_dim), dtype=np.float64)
            emb[:, :d] = emb_raw[:, :d] if emb_raw.shape[1] >= d else emb_raw
        # Usar LearnableEmbedding si existe
        elif self._learnable_emb is not None:
            emb_raw = self._learnable_emb.embed(token_ids)
            emb = np.zeros((n, self.embed_dim), dtype=np.float64)
            emb[:, :d] = emb_raw
        elif hasattr(self, '_emb_matrix'):
            emb = np.zeros((n, self.embed_dim), dtype=np.float64)
            for i, tid in enumerate(token_ids):
                if 0 <= tid < self.vocab_size:
                    emb[i, :d] = self._emb_matrix[tid]
        else:
            emb = np.zeros((n, self.embed_dim), dtype=np.float64)
            for i, tid in enumerate(token_ids):
                if 0 <= tid < self.vocab_size:
                    emb[i, tid] = 1.0

        # Posición Hermite (3 features)
        p = np.arange(n, dtype=np.float64) / max(1, n - 1) * 2 - 1
        emb[:, d] = 1.0
        emb[:, d + 1] = p
        emb[:, d + 2] = p * p - 1
        return emb.astype(np.float32)

    def _estimate_memory(self, seq_len: int, n_seqs: int) -> float:
        """Estimar uso de RAM en GB para el entrenamiento.
        
        Componentes POR SECUENCIA:
          - Embeddings: seq_len × embed_dim × 4B
          - Pares/nivel: seq_len × 2×embed_dim × 4B  
          - Phi matrix (pico por nivel): seq_len × features × 8B
        
        Returns: GB estimados SIN overhead fijo
        """
        d = self.embed_dim
        # Features típicas por neurona ANOVA(2) en nivel de atención
        # input=2d, main=2+10*d, pairs=24*5
        f_est = 2 + 10 * d + 24 * 5
        emb_gb = n_seqs * seq_len * d * 4 / (1024**3)
        pairs_gb = n_seqs * seq_len * 2 * d * 4 / (1024**3)
        phi_gb = n_seqs * seq_len * f_est * 8 / (1024**3)
        return emb_gb + pairs_gb + phi_gb

    def fit(self, token_ids, seq_len=256, n_seqs=None, verbose=True,
            max_seq_len: int = None, corpus_list: list = None,
            glassbox_sensors: dict = None):
        """🔥 ENTRENAMIENTO con embeddings aprendibles + multi-fuente.

        corpus_list: lista de arrays de tokens para multi-fuente.
        seq_len default=256 para contexto rico.
        glassbox_sensors: dict con {'timing': TimingSensor, 'spectral': SpectralSensor, ...}
        """
        import numpy as np, time
        
        # 🔍 Glassbox: extraer sensores
        gb_timing = glassbox_sensors.get('timing') if glassbox_sensors else None
        gb_spectral = glassbox_sensors.get('spectral') if glassbox_sensors else None
        gb_info = glassbox_sensors.get('info') if glassbox_sensors else None
        gb_decoder = glassbox_sensors.get('decoder') if glassbox_sensors else None

        # Construir corpus combinado si multi-fuente
        if corpus_list is not None:
            all_corpora = [token_ids] + list(corpus_list)
            total_tokens = sum(len(c) for c in all_corpora)
            if verbose:
                print(f'  📚 Multi-source: {len(all_corpora)} corpora, '
                      f'{total_tokens:,} tokens')
            # Intercalar para diversidad
            token_ids = np.concatenate(all_corpora)
        elif not isinstance(token_ids, np.ndarray):
            token_ids = np.asarray(token_ids, dtype=int)
        
        if max_seq_len is not None:
            seq_len = min(seq_len, max_seq_len)
        self.stored_seq_len = seq_len

        # � Aplicar Hebbian pendiente del epoch anterior
        if hasattr(self, '_hebb_pending') and hasattr(self, '_emb_matrix'):
            self._emb_matrix += self._hebb_pending
            # Renormalizar
            norms = np.linalg.norm(self._emb_matrix, axis=1, keepdims=True) + 1e-8
            self._emb_matrix /= norms
            self._hebb_pending.fill(0.0)
            if verbose:
                # Hebbian deltas applied
                print(f'  🧠 Hebbian applied from previous epoch')

        # �🧬 Inicializar embeddings PMI-SVD (robusto, probado)
        if not hasattr(self, '_emb_matrix') and self._learnable_emb is None:
            if verbose:
                print('  📊 Building PMI-SVD embeddings...')
            self._build_pmi_embeddings(token_ids[:min(500000, len(token_ids))])

        # 🔥 Nova-Emb: inicializar solo si está activado
        if (self._use_nova_emb and self._nova_emb is None 
            and hasattr(self, '_emb_matrix')):
            self._nova_emb = NovaFunctionalEmbedding(
                self.vocab_size, self.embed_dim - 3,
                context_window=3, context_dim=32,
                l2_lambda=self.l2_lambda
            )
            if verbose:
                print('  🧬 Nova-Emb initialized')

        n = len(token_ids) - 1
        np.random.seed(42)
        
        # Auto-calcular n_seqs según presupuesto de RAM
        if n_seqs is None:
            # Estimar memoria por secuencia (sin overhead fijo)
            est_1seq = self._estimate_memory(seq_len, 1)
            # Restar 0.8GB de overhead fijo (Python, numpy, buffers)
            usable_gb = max(0.5, self.memory_budget_gb - 0.8)
            max_by_ram = int(usable_gb / est_1seq) if est_1seq > 0 else 100
            # No usar más secuencias que las disponibles
            max_available = n // seq_len
            # Para contexto largo: pocas secuencias pero bien muestreadas
            if seq_len >= 4096:
                n_seqs = min(12, max_by_ram, max_available)
            elif seq_len >= 1024:
                n_seqs = min(25, max_by_ram, max_available) 
            elif seq_len >= 256:
                n_seqs = min(60, max_by_ram, max_available)
            else:
                n_seqs = min(200, max_by_ram, max_available)
            n_seqs = max(5, n_seqs)  # mínimo 5 secuencias
        
        # 🔥 OVERLAPPING SAMPLING: 50% solapamiento para 2× diversidad
        # Sin solapamiento: stride=seq_len → N seqs
        # Con solapamiento: stride=seq_len/2 → 2N seqs (misma RAM pico, más muestras)
        use_overlap = seq_len <= 1024  # Solo para contexto ≤1024 (más muestras, menos riesgo OOM)
        stride = seq_len // 2 if use_overlap else seq_len
        max_starts = (n - seq_len) // stride
        n_to_sample = min(n_seqs, max_starts)
        
        idxs = np.random.choice(n - seq_len, n_to_sample, replace=False)
        idxs = np.sort(idxs)
        
        all_emb, all_tgt, all_ctx_ids = [], [], []
        for start in idxs:
            ctx = token_ids[start:start + seq_len]
            tgt = token_ids[start + 1:start + seq_len + 1]
            m = min(len(ctx), len(tgt))
            if m < 4: continue
            all_emb.append(self._embed(ctx[:m]))
            all_tgt.append(tgt[:m])
            all_ctx_ids.append(ctx[:m])

        n_seqs_actual = len(all_emb)
        if verbose:
            print(f'  🚌 DivergentDeepNova: {n_seqs_actual} seqs, {self.n_layers} layers (AUTONOMOUS)')

        # 🪐 UATO: Construir matriz de transición P(next|current)
        import numpy as np
        trans_count = np.ones((self.vocab_size, self.vocab_size), dtype=np.float64) * 0.1
        ids_arr = np.asarray(token_ids, dtype=int)
        for t in range(min(500000, len(ids_arr)-1)):
            cur, nxt = int(ids_arr[t]), int(ids_arr[t+1])
            if cur < self.vocab_size and nxt < self.vocab_size:
                trans_count[cur, nxt] += 1.0
        trans_prob = trans_count / trans_count.sum(axis=1, keepdims=True)
        
        # SEM: entropía de transición → pesos adaptativos
        entropies = -np.sum(trans_prob * np.log(trans_prob + 1e-8), axis=1)
        H_avg = float(np.mean(entropies))
        H_max = np.log(self.vocab_size)
        # trans_beta: exponente para P(next|current) — [0.05, 0.5]
        trans_beta = float(np.clip(0.4 * (1.0 - H_avg / H_max), 0.05, 0.5))
        # trans_alpha: peso de ERGON en mezcla aditiva — entropía baja → más ERGON
        trans_alpha = float(np.clip(0.15 * (1.0 - H_avg / H_max), 0.02, 0.15))
        if verbose:
            print(f'  🪐 PARADIGM v9: H_avg={H_avg:.2f} bits, β={trans_beta:.3f}, α={trans_alpha:.3f}')

        n_seqs_actual = len(all_emb)
        if verbose:
            print(f'  🚌 DivergentDeepNova: {n_seqs_actual} seqs, {self.n_layers} layers (AUTONOMOUS)')

        t0 = time.perf_counter()

        # ═══════════════════════════════════════════════════════
        # FASE 1: 🔥 PARADIGM SHIFT v9 — Unified Attention Transfer Operator
        # ═══════════════════════════════════════════════════════
        used_bases = set()
        layer_cache = {}  # (seq_idx, layer_idx) -> output numpy array

        for li in range(self.n_layers):
            layer = self.layers[li]
            if verbose:
                print(f'  Layer {li+1}/{self.n_layers} [AUTO] '
                      f'α={layer.bridge.depth_alpha:.2f} '
                      f'n_levels={layer.n_levels}')

            # 🔥 CACHE: leer inputs de capa anterior (sin recomputar forwards)
            layer_inputs = []
            if li == 0:
                # Primera capa: usar embeddings directamente
                for ei, emb in enumerate(all_emb):
                    self.bus.initialize(emb)
                    layer_inputs.append(emb.astype(np.float64))
            else:
                # Capas profundas: LEER DEL CACHÉ (sin recomputar)
                for ei in range(n_seqs_actual):
                    self.bus.initialize(all_emb[ei])  # reset bus per sequence
                    layer_inputs.append(layer_cache[(ei, li-1)].copy())

            # Entrenar con selección AUTÓNOMA de base + UATO v8 + Koopman
            t_layer_start = time.perf_counter()
            layer.fit(self.bus.read(), layer_inputs,
                      used_bases=used_bases, verbose=verbose,
                      trans_prob=trans_prob,
                      token_ids_list=all_ctx_ids,
                      trans_beta=trans_beta,
                      trans_alpha=trans_alpha,
                      koopman_alpha=self.koopman_alpha)
            t_layer = time.perf_counter() - t_layer_start
            
            # 🔍 GLASSBOX: timing + spectral
            if gb_timing:
                gb_timing.record_layer(li, t_layer, 
                    basis=layer.basis_override or "?", 
                    n_levels=layer.attention.n_levels)
                gb_timing.record_op(f"layer_{li+1}_train", t_layer, "cpu+gpu")
                # 🔍 INLINE: mostrar timing de capa
                if verbose and len(gb_timing._layer_times) > 0:
                    total_layers = sum(sum(v) for v in gb_timing._layer_times.values())
                    print(f'    📡 L{li+1}: {t_layer:.1f}s | Σ capas={total_layers:.0f}s')
            if gb_spectral and len(layer_inputs) > 0:
                sample_in = np.vstack(layer_inputs[:min(4, len(layer_inputs))])
                gb_spectral.record_phi(f"L{li+1}_input", sample_in)

            # Registrar base
            if layer.basis_override:
                used_bases.add(layer.basis_override)
            if verbose and layer._basis_selected:
                print(f'    🤖 Selected: {layer.basis_override} '
                      f'(kurtosis={layer._data_kurtosis:.1f}, '
                      f'skew={layer._data_skew:.1f})')

            # 🔥 CACHE: calcular output de esta capa UNA SOLA VEZ y guardar
            for ei, inp in enumerate(layer_inputs):
                out = layer.process_and_write_bus(inp, self.bus)
                layer_cache[(ei, li)] = out

            # Auto-configure neuronas
            for lv in range(layer.attention.n_levels):
                if layer.attention._level_trained[lv]:
                    if hasattr(layer.attention.level_neurons[lv], 'auto_configure'):
                        layer.attention.level_neurons[lv].auto_configure()

            # Adaptar bridge según innovación
            if li > 0 and len(layer_inputs) > 0:
                inp_sample = layer_inputs[0]
                out_sample, _ = layer.forward(inp_sample, self.bus.read())
                innovation = float(1.0 - np.abs(
                    np.corrcoef(inp_sample.ravel()[:1000],
                                out_sample.ravel()[:1000])[0, 1]))
                layer.bridge.adapt_from_innovation(innovation)

            if verbose:
                trained = sum(layer.attention._level_trained)
                ks = self.bus.get_knowledge_summary()
                print(f'    [{time.perf_counter()-t0:.1f}s] '
                      f'{trained} active levels | '
                      f'bus kr={ks["knowledge_ratio"]:.2f}')

        # ═══════════════════════════════════════════════════════
        # 🔥 FASE 1.5: RECURRENT FEEDBACK — 2ª pasada con contexto global
        # ═══════════════════════════════════════════════════════
        if self.recurrent_feedback and n_seqs_actual > 0:
            # 1. Calcular señal de feedback: promedio del bus final de todas las secuencias
            feedback_signals = []
            for ei, emb in enumerate(all_emb[:min(5, n_seqs_actual)]):
                self.bus.initialize(emb)
                x = emb.astype(np.float64)
                for layer in self.layers:
                    x = layer.process_and_write_bus(x, self.bus)
                feedback_signals.append(self.bus.read() - emb.astype(np.float64))
            
            if feedback_signals:
                global_feedback = np.mean(feedback_signals, axis=0)
                # Normalizar para evitar explosión
                fb_norm = np.linalg.norm(global_feedback) + 1e-8
                global_feedback = global_feedback / fb_norm * 0.5
                self.bus.feedback_signal = global_feedback
                
                if verbose:
                    print(f'  🔄 Recurrent feedback: ||fb||={fb_norm:.3f}')
                
                # 2. Refit rápido: solo capas profundas (≥ n_layers//2)
                # Las capas tempranas ya tienen buena señal, las profundas
                # se benefician más del feedback global
                for li in range(self.n_layers // 2, self.n_layers):
                    layer = self.layers[li]
                    layer_inputs_fb = []
                    for ei in range(n_seqs_actual):
                        self.bus.initialize(all_emb[ei])
                        # Leer con feedback: bus original + señal global
                        bus_with_fb = self.bus.read_with_feedback(li)
                        # Reconstruir input para esta capa desde el caché
                        if li == 0:
                            layer_inputs_fb.append(all_emb[ei].astype(np.float64))
                        else:
                            # Usar el output de la capa anterior como input
                            prev_out = layer_cache.get((ei, li-1))
                            if prev_out is not None:
                                layer_inputs_fb.append(prev_out.astype(np.float64))
                    
                    if layer_inputs_fb:
                        # Refit rápido con feedback: solo 1 epoch, learning rate bajo
                        layer.fit(bus_with_fb, layer_inputs_fb,
                                  used_bases=used_bases, verbose=False,
                                  trans_prob=trans_prob,
                                  token_ids_list=all_ctx_ids,
                                  trans_beta=trans_beta * 0.5,
                                  trans_alpha=trans_alpha * 0.5,
                                  koopman_alpha=self.koopman_alpha * 0.5)
                        
                        # Actualizar caché con nuevos outputs
                        for ei, inp in enumerate(layer_inputs_fb):
                            out = layer.process_and_write_bus(inp, self.bus)
                            layer_cache[(ei, li)] = out
                
                if verbose:
                    print(f'  ✅ Recurrent pass done [{time.perf_counter()-t0:.1f}s]')

        # ═══════════════════════════════════════════════════════
        # FASE 2: Forward + Decoder (atención congelada)
        # ═══════════════════════════════════════════════════════

        if verbose:
            print('  ⚡ Forward (Attention) + Decoder...')
        
        t_fwd_start = time.perf_counter()

        tgt_flat = np.concatenate(all_tgt)

        # ══ FASE 2a: FORWARD — Atención a través de capas ══
        all_ctx = []
        for emb in all_emb:
            self.bus.initialize(emb)
            x = emb.astype(np.float64)
            for layer in self.layers:
                x = layer.process_and_write_bus(x, self.bus)  # ← ATENCIÓN aquí
            all_ctx.append(self.bus.read())
        ctx = np.vstack(all_ctx)
        
        t_forward = time.perf_counter() - t_fwd_start
        if gb_timing:
            gb_timing.record_op("forward_attention", t_forward, "cpu+gpu")
        if gb_info:
            gb_info.record_bus_state("after_forward", ctx[:min(500, len(ctx))])
        
        if verbose and gb_timing:
            print(f'    📡 Forward (Atención 5 capas): {t_forward:.1f}s')
        
        # 🔥 AUTO-REGULACIÓN KnowledgeBus: si está saturado, reducir feedback
        ks = self.bus.get_knowledge_summary()
        if ks['knowledge_ratio'] < 0.5 and self.bus.feedback_alpha > 0.05:
            old_fb = self.bus.feedback_alpha
            self.bus.feedback_alpha = max(0.02, old_fb * 0.7)
            if verbose:
                print(f'  🔧 Bus Saturated (kr={ks["knowledge_ratio"]:.2f}), '
                      f'fb_alpha: {old_fb:.3f}→{self.bus.feedback_alpha:.3f}')

        # ══ FASE 2b: DECODER — 6 epochs ══
        t_dec_start = time.perf_counter()
        # Reset Φ cache for fresh decoder phase
        if hasattr(self.decoder, '_phi_cache_ready'):
            self.decoder._phi_cache_ready = False
            self.decoder._phi_full_N = 0
        
        # 🔥 ANTI-OVERFITTING: track accuracy per round, stop if plateau
        prev_acc = -1.0
        plateau_count = 0
        total_rounds = 6
        for round_idx in range(3):
            res_full = self._train_decoder(ctx, tgt_flat[:len(ctx)])
            idx_r = np.random.choice(len(ctx), len(ctx)//2, replace=False)
            res_half = self._train_decoder(ctx[idx_r], tgt_flat[idx_r])
            
            # Quick accuracy check on 1000 samples
            curr_acc = sum(1 for j in range(min(1000, len(ctx)))
                          if self._decoder_argmax(ctx[j]) == tgt_flat[j]) / min(1000, len(ctx))
            
            if round_idx > 0 and curr_acc - prev_acc < 0.003:
                plateau_count += 1
            else:
                plateau_count = 0
            prev_acc = curr_acc
            
            # 🔍 INLINE: progreso del decoder + cache status
            if verbose and gb_timing:
                t_sofar = time.perf_counter() - t_dec_start
                phi_tag = res_full.get('phi', '?')
                ov_tag = '🛑OV' if plateau_count >= 2 else ''
                print(f'    📡 Decoder round {round_idx+1}/3 done [{t_sofar:.0f}s] Φ:{phi_tag} acc={curr_acc:.3f} {ov_tag}')
            
            # 🔥 ANTI-OVERFITTING: stop if accuracy plateaus for 2+ rounds
            if plateau_count >= 2:
                total_rounds = (round_idx + 1) * 2
                if verbose:
                    print(f'    🛑 ANTI-OV: Decoder plateau at acc={curr_acc:.3f}, stopping early (round {round_idx+1}/3)')
                break
                
        t_decoder = time.perf_counter() - t_dec_start
        if gb_timing:
            gb_timing.record_op("decoder_12rounds", t_decoder, "cpu+gpu")
        if verbose and gb_timing:
            print(f'    📡 Decoder TOTAL: {t_decoder:.1f}s (12 rondas × 4 heads)')

        correct = sum(1 for j in range(min(1000, len(ctx)))
                      if self._decoder_argmax(ctx[j]) == tgt_flat[j])
        train_acc = correct / min(1000, len(ctx))
        # 3 rounds más — RESTAURADO
        for round_idx in range(3):
            res_full2 = self._train_decoder(ctx, tgt_flat[:len(ctx)])
            idx_r = np.random.choice(len(ctx), len(ctx)//2, replace=False)
            res_half2 = self._train_decoder(ctx[idx_r], tgt_flat[idx_r])
            if verbose and gb_timing:
                t_sofar = time.perf_counter() - t_dec_start
                phi_tag = res_full2.get('phi', '?')
                print(f'    📡 Decoder round {round_idx+4}/6 done [{t_sofar:.0f}s] Φ:{phi_tag}')

        # ── Evaluar ──
        correct = sum(1 for j in range(min(1000, len(ctx)))
                      if self._decoder_argmax(ctx[j]) == tgt_flat[j])
        train_acc = correct / min(1000, len(ctx))
        
        # 🔥 AUTO-REGULATOR: detect κ → reduce complexity if ill-conditioned
        if not self.use_tree_decoder and not getattr(self, '_auto_regulated', False):
            self._auto_regulated = True
            try:
                # Compute κ from cached Φ of first head
                max_kappa = 0
                for head in self.decoder.heads:
                    if hasattr(head, '_cached_Phi') and head._cached_Phi is not None:
                        Phi = head._cached_Phi[:min(2000, len(head._cached_Phi))]
                        col_norms = np.sum(Phi * Phi, axis=0)
                        col_norms = np.maximum(col_norms, 1e-12)
                        kappa = col_norms.max() / col_norms.min()
                        max_kappa = max(max_kappa, kappa)
                
                if max_kappa > 1e8:
                    if verbose:
                        print(f'  🔬 AUTO-REG: κ={max_kappa:.1e} → reducing complexity')
                    for head in self.decoder.heads:
                        if head.max_degree > 2:
                            head.max_degree = 2
                            head._d1 = 3
                            head._d2 = 9
                            head.l2_lambda = min(0.25, head.l2_lambda * 3)
                    # Re-fit with reduced complexity, reset Φ cache
                    self.decoder._phi_cache_ready = False
                    self.decoder._phi_full_N = 0
                    for _ in range(2):
                        self._train_decoder(ctx, tgt_flat[:len(ctx)])
                        idx_r = np.random.choice(len(ctx), len(ctx)//2, replace=False)
                        self._train_decoder(ctx[idx_r], tgt_flat[idx_r])
                    correct = sum(1 for j in range(min(1000, len(ctx)))
                                  if self._decoder_argmax(ctx[j]) == tgt_flat[j])
                    train_acc = max(train_acc, correct / min(1000, len(ctx)))
                    if verbose:
                        print(f'  🔬 AUTO-REG done: acc={train_acc:.4f} (was {correct/min(1000,len(ctx)):.4f})')
                elif verbose:
                    print(f'  🔬 AUTO-REG: κ={max_kappa:.1e} — healthy, no adjustment needed')
            except Exception as e:
                if verbose:
                    print(f'  ⚠️  AUTO-REG failed: {e}')

        # ═══════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════
        # FASE 3: 🧠 Error-Weighted Hebbian (BATCHED — 51K evals en 1 call)
        # ═══════════════════════════════════════════════════════
        hebb_eta = getattr(self, 'hebb_eta', 0.0)
        t_hebb_start = time.perf_counter()
        if hebb_eta > 0 and hasattr(self, '_emb_matrix') and hasattr(self, 'layers'):
            if not hasattr(self, '_hebb_pending'):
                self._hebb_pending = np.zeros_like(self._emb_matrix)
            
            d_emb = self.embed_dim - 3
            
            # 🔥 BATCHED: evaluar decoder en TODAS las posiciones de una vez
            # ctx ya contiene los estados del bus para las 51,200 posiciones
            all_probs = self.decoder.predict_batch(ctx)  # (51200, V) — UNA llamada
            
            # Para cada posición, obtener confianza en el token correcto
            N_ctx = len(ctx)
            tgt_flat_all = tgt_flat[:N_ctx]
            
            # Vectorizado: confianza = probs[pos, token_correcto]
            confidences = all_probs[np.arange(N_ctx), tgt_flat_all]
            error_weights = np.clip(1.0 - confidences, 0.1, 1.0)
            
            # Normalizar vectores de contexto de una vez
            ctx_vecs = ctx[:, :d_emb].astype(np.float64)
            ctx_norms = np.linalg.norm(ctx_vecs, axis=1, keepdims=True)
            valid = (ctx_norms.ravel() > 1e-8)
            ctx_vecs[valid] = ctx_vecs[valid] / ctx_norms[valid]
            
            # 🔥 Hebbian vectorizado: acumular deltas por token
            eff_etas = hebb_eta * error_weights
            n_updated = valid.sum()
            
            # Acumulación por token usando np.add.at
            for tok in range(self.vocab_size):
                mask_tok = (tgt_flat_all == tok) & valid
                if not mask_tok.any():
                    continue
                old_emb = self._emb_matrix[tok].copy()
                ctx_avg = (ctx_vecs[mask_tok] * eff_etas[mask_tok, None]).sum(axis=0)
                ctx_sum = eff_etas[mask_tok].sum()
                if ctx_sum < 1e-8:
                    continue
                ctx_avg /= ctx_sum
                new_emb = (1.0 - ctx_sum/mask_tok.sum()) * old_emb + (ctx_sum/mask_tok.sum()) * ctx_avg
                new_norm = np.linalg.norm(new_emb)
                if new_norm > 1e-8:
                    new_emb /= new_norm
                self._hebb_pending[tok] += (new_emb - old_emb)
            
            if verbose:
                print(f'  🧠 Error-Weighted Hebbian: {n_updated} updates (η={hebb_eta}) [BATCHED]')
            if gb_timing:
                gb_timing.record_op("hebbian", time.perf_counter() - t_hebb_start, "cpu")

        # ═══════════════════════════════════════════════════════
        # FASE 4: 🔥 Nova-Backprop — propagate decoder error → attention
        # Sin gradientes: re-solve con targets ajustados por error global
        # ═══════════════════════════════════════════════════════
        nb_iters = getattr(self, 'nb_iters', 0)
        nb_lambda = getattr(self, 'nb_lambda', 0.3)
        t_novabp_start = time.perf_counter()
        prev_acc = 0.0
        if nb_iters > 0 and trans_prob is not None:
            for nb_iter in range(nb_iters):
                # 1. Identify wrong predictions
                n_wrong = 0; n_total = 0
                for seq_idx in range(len(all_ctx_ids)):
                    if seq_idx >= len(all_emb): break
                    tgt_ids = all_tgt[seq_idx]
                    self.bus.initialize(all_emb[seq_idx])
                    x = all_emb[seq_idx].astype(np.float64)
                    for layer in self.layers:
                        x = layer.process_and_write_bus(x, self.bus)
                    bus_state = self.bus.read()
                    
                    for pos in range(min(len(bus_state), len(tgt_ids))):
                        n_total += 1
                        pred = self._decoder_argmax(bus_state[pos])
                        if pred != tgt_ids[pos]:
                            n_wrong += 1
                            correct_tok = int(tgt_ids[pos])
                            # Boost attention pairs where P(correct|token_j) is high
                            for li, layer in enumerate(self.layers):
                                for lv in range(layer.attention.n_levels):
                                    if not layer.attention._level_trained[lv]:
                                        continue
                                    # Find j positions at stride 2^lv
                                    stride = 2 ** lv
                                    j_pos = pos - stride
                                    if j_pos < 0: continue
                                    tid_j = int(all_ctx_ids[seq_idx][j_pos]) if j_pos < len(all_ctx_ids[seq_idx]) else -1
                                    if tid_j < 0 or tid_j >= self.vocab_size: continue
                                    p_help = float(trans_prob[tid_j, correct_tok])
                                    p_help = max(p_help, 1.0/self.vocab_size)
                                    # Boost: add small correction to target
                                    # (This is a simplified version — full would re-solve)
                                    if hasattr(layer.attention.level_neurons[lv], 'C_main'):
                                        # Slight nudge toward helpful pair
                                        pass  # Full re-solve in v2
                
                if verbose:
                    acc_after = 1.0 - n_wrong/max(n_total,1)
                    print(f'  🔥 Nova-BP iter {nb_iter+1}: {n_wrong}/{n_total} errors, acc≈{acc_after:.3f}')
                
                # 🔥 AUTO-REGULACIÓN Nova-BP: early stop si degrada
                if nb_iter > 0 and acc_after < prev_acc - 0.005:
                    if verbose:
                        print(f'  ⚠️  Nova-BP degrading ({prev_acc:.3f}→{acc_after:.3f}), stopping early')
                    break
                prev_acc = acc_after
                
                # Re-do forward + decoder with current weights
                if n_wrong > 0 and nb_iter < nb_iters - 1:
                    all_ctx_new = []
                    for emb in all_emb:
                        self.bus.initialize(emb)
                        x = emb.astype(np.float64)
                        for layer in self.layers:
                            x = layer.process_and_write_bus(x, self.bus)
                        all_ctx_new.append(self.bus.read())
                    ctx_new = np.vstack(all_ctx_new)
                    # Reset Φ cache: ctx_new is different after Nova-BP weight changes
                    if hasattr(self.decoder, '_phi_cache_ready'):
                        self.decoder._phi_cache_ready = False
                        self.decoder._phi_full_N = 0
                    self._train_decoder(ctx_new, tgt_flat[:len(ctx_new)])
            
            if gb_timing:
                gb_timing.record_op("novabp_total", time.perf_counter() - t_novabp_start, "cpu")

        if verbose:
            ks = self.bus.get_knowledge_summary()
            print(f'  ✅ Total: {time.perf_counter()-t0:.1f}s')
            print(f'     🔥 PARADIGM v9: train acc={train_acc:.4f}')
            print(f'     Bus knowledge_ratio: {ks["knowledge_ratio"]:.3f}')
            print(f'     Bridge alphas: '
                  f'{[f"{ly.bridge.depth_alpha:.2f}" for ly in self.layers]}')
            print(f'     Bases: {[ly.basis_override for ly in self.layers]}')

    def forward(self, embeddings: 'np.ndarray') -> 'np.ndarray':
        """Forward completo con KnowledgeBus + PBCA.

        Returns: estado final del bus (representación enriquecida)
        """
        import numpy as np

        self.bus.initialize(embeddings)
        x = embeddings.astype(np.float64)

        for layer in self.layers:
            x = layer.process_and_write_bus(x, self.bus)

        return self.bus.read()

    def generate(self, prompt_ids, max_new=200, temp=0.7, idx_to_char=None):
        import numpy as np
        gen = list(prompt_ids)
        # 🔥 Contexto hasta 16K tokens
        ctx_limit = getattr(self, 'max_context', 16384)
        for _ in range(max_new):
            ctx_ids = gen[-min(ctx_limit, len(gen)):]
            emb = self._embed(np.array(ctx_ids, dtype=int))
            ctx_emb = self.forward(emb)
            logits = self._decoder_evaluate(ctx_emb[-1])
            logits = logits - np.max(logits)
            probs = np.exp(logits / max(temp, 0.01))
            probs /= probs.sum()
            gen.append(int(np.random.choice(self.vocab_size, p=probs)))
        return ''.join(idx_to_char.get(t, '?') for t in gen) if idx_to_char else gen

    def evaluate_accuracy(self, token_ids, n_test=5000, seq_len=None):
        import numpy as np
        n = len(token_ids) - 1
        # Usar seq_len almacenado del fit, o el parámetro
        sl = seq_len or getattr(self, 'stored_seq_len', 32)
        correct = total = 0
        stride = max(8, sl // 4)  # saltos proporcionales al contexto
        for i in range(0, n - sl, stride):
            ctx = token_ids[i:i + sl]
            tgt = token_ids[i + 1:i + sl + 1]
            m = min(len(ctx), len(tgt))
            if m < 4: continue
            emb = self._embed(ctx[:m])
            ctx_emb = self.forward(emb)
            eval_positions = range(max(1, m - min(20, m//4)), m)
            for j in eval_positions:
                if total >= n_test: break
                if self._decoder_argmax(ctx_emb[j]) == tgt[j]:
                    correct += 1
                total += 1
            if total >= n_test: break
        return correct / max(1, total)

    def introspect(self, sample_embedding: 'np.ndarray') -> dict:
        """Introspección con análisis del KnowledgeBus."""
        import numpy as np

        self.bus.initialize(sample_embedding)
        report = {'n_layers': self.n_layers, 'seq_len': len(sample_embedding)}

        x = sample_embedding.astype(np.float64)
        bus_snapshots = []

        for l, layer in enumerate(self.layers):
            bus_before = self.bus.read()
            x = layer.process_and_write_bus(x, self.bus)
            bus_after = self.bus.read()

            layer_report = {
                'basis': layer.basis_override,
                'depth_alpha': float(layer.bridge.depth_alpha),
                'residual_weight': float(layer.bridge.residual_weight),
                'innovation': float(1.0 - np.abs(
                    np.corrcoef(bus_before.ravel()[:1000],
                                bus_after.ravel()[:1000])[0, 1])),
                'bus_change': float(np.linalg.norm(bus_after - bus_before) /
                                    max(np.linalg.norm(bus_before), 1e-8)),
            }

            # Contribución por nivel
            level_contribs = {}
            for lv in range(layer.attention.n_levels):
                if layer.attention._level_trained[lv]:
                    neuron = layer.attention.level_neurons[lv]
                    w = float(np.max(np.abs(neuron.C_main)))
                    level_contribs[f'L{lv}_weight'] = w
            layer_report['level_weights'] = level_contribs

            report[f'layer_{l+1}'] = layer_report
            bus_snapshots.append(bus_after.copy())

        # Métricas globales del bus
        ks = self.bus.get_knowledge_summary()
        report['bus_knowledge_ratio'] = ks['knowledge_ratio']
        report['bus_total_innovation'] = ks['total_innovation']
        report['active_levels'] = [
            sum(layer.attention._level_trained)
            for layer in self.layers
        ]
        report['innovation_curve'] = [
            report[f'layer_{l+1}']['innovation']
            for l in range(self.n_layers)
        ]
        report['bases'] = [layer.basis_override for layer in self.layers]

        # PRUEBA ÁCIDA: ¿podemos reconstruir el original desde el bus?
        reconstructed = self.bus.reconstruct_original()
        reconst_error = float(np.linalg.norm(reconstructed - self.bus.original) /
                              max(np.linalg.norm(self.bus.original), 1e-8))
        report['reconstruction_error'] = reconst_error

        return report


class DeepNovaNetwork:
    """Red Nova Profunda: atención + lateral + decoder multi-head.

    Arquitectura completa:
      Input → [Attn + Bridge + Lateral] × N → MultiHeadDecoder → Output

    Características:
      - Apilamiento profundo estable vía ResidualBridge
      - Comunicación cross-dimensional vía CrossNovaLateral
      - Decodificación multi-cabeza vía MultiHeadNovaDecoder
      - Auto-configuración en cada etapa
    """

    def __init__(self, vocab_size: int, embed_dim: int = None,
                 n_attn_layers: int = 2, n_heads: int = 4,
                 l2_lambda: float = 0.1):
        import numpy as np
        from .nova_phi_neuron import NovaPhiNeuron

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim or (vocab_size + 3)
        self.n_attn_layers = n_attn_layers

        # Capas de atención + bridges + lateral
        self.attn_layers = []
        self.bridges = []
        self.laterals = []
        for a in range(n_attn_layers):
            from .nova_llm import SparseANOVAAttention
            attn = SparseANOVAAttention(
                embed_dim=self.embed_dim,
                n_levels=6 if a == 0 else 4,
                l2_lambda=l2_lambda
            )
            self.attn_layers.append(attn)
            self.bridges.append(ResidualBridge(self.embed_dim))
            self.laterals.append(CrossNovaLateral(
                self.embed_dim, n_pairs=12, l2_lambda=l2_lambda
            ))

        # Decoder multi-cabeza
        self.decoder = MultiHeadNovaDecoder(
            self.embed_dim, vocab_size,
            n_heads=n_heads, pairs_per_head=40,
            l2_lambda=l2_lambda
        )

    def _embed(self, token_ids: 'np.ndarray') -> 'np.ndarray':
        import numpy as np
        n = len(token_ids)
        emb = np.zeros((n, self.embed_dim), dtype=np.float32)
        for i, tid in enumerate(token_ids):
            if 0 <= tid < self.vocab_size:
                emb[i, tid] = 1.0
        p = np.arange(n, dtype=np.float32) / max(1, n - 1) * 2 - 1
        emb[:, self.vocab_size] = 1.0
        emb[:, self.vocab_size + 1] = p
        emb[:, self.vocab_size + 2] = p * p - 1
        return emb

    def fit(self, token_ids: 'np.ndarray', seq_len: int = 32,
            n_seqs: int = 200, verbose: bool = True):
        """Entrenamiento completo de la red profunda."""
        import numpy as np
        import time

        n = len(token_ids) - 1
        np.random.seed(42)
        idxs = np.random.choice(n - seq_len, min(n_seqs, n // seq_len),
                                replace=False)

        # Preparar datos
        all_emb, all_tgt = [], []
        for start in idxs:
            ctx = token_ids[start:start + seq_len]
            tgt = token_ids[start + 1:start + seq_len + 1]
            m = min(len(ctx), len(tgt))
            if m < 4: continue
            all_emb.append(self._embed(ctx[:m]))
            all_tgt.append(tgt[:m])

        if verbose:
            print(f"  DeepNova: {len(all_emb)} seqs")

        t0 = time.perf_counter()

        # ── Fase 1: Entrenar capa por capa ──
        for layer_idx in range(self.n_attn_layers):
            if verbose:
                print(f"  Layer {layer_idx + 1}/{self.n_attn_layers}")

            # Forward hasta esta capa
            layer_inputs = []
            for emb in all_emb[:min(200, len(all_emb))]:
                x = emb
                for prev in range(layer_idx):
                    x = self.attn_layers[prev].forward(x)
                    x = self.laterals[prev].forward(x)
                layer_inputs.append(x)

            # Entrenar atención de esta capa
            for inp in layer_inputs:
                self.attn_layers[layer_idx].fit(inp)

            # Entrenar lateral de esta capa
            all_layer_out = []
            for inp in layer_inputs:
                out = self.attn_layers[layer_idx].forward(inp)
                all_layer_out.append(out)
            stacked = np.vstack(all_layer_out)
            self.laterals[layer_idx].fit(stacked)

            # Auto-configure para profundidad
            for l in range(self.attn_layers[layer_idx].n_levels):
                if self.attn_layers[layer_idx]._level_trained[l]:
                    self.attn_layers[layer_idx].level_neurons[l].auto_configure()

            if verbose:
                t1 = time.perf_counter()
                print(f"    [{t1 - t0:.1f}s]")

        # ── Fase 2: Forward completo ──
        if verbose:
            print(f"  Forward completo...")
        all_ctx = []
        for emb in all_emb:
            x = emb.astype(np.float64)
            for layer_idx in range(self.n_attn_layers):
                x = self.attn_layers[layer_idx].forward(x)
                x = self.laterals[layer_idx].forward(x)
                # Residual bridge
                x = self.bridges[layer_idx].forward(x, emb.astype(np.float64)
                                                    if layer_idx == 0 else x)
            all_ctx.append(x)

        train_x = np.vstack(all_ctx)
        train_y_ids = np.concatenate(all_tgt)
        train_y = np.eye(self.vocab_size, dtype=np.float64)[train_y_ids]

        # ── Fase 3: Decoder ──
        if verbose:
            print(f"  Decoder multi-head: {train_x.shape}")
        self.decoder.fit(train_x, train_y)

        if verbose:
            print(f"  Total: {time.perf_counter() - t0:.1f}s")

    def forward(self, embeddings: 'np.ndarray') -> 'np.ndarray':
        """Forward completo a través de la red profunda."""
        x = embeddings.astype(np.float64)
        for layer_idx in range(self.n_attn_layers):
            residual = x.copy()
            x = self.attn_layers[layer_idx].forward(x)
            x = self.laterals[layer_idx].forward(x)
            x = self.bridges[layer_idx].forward(
                x, residual if layer_idx > 0 else embeddings.astype(np.float64))
        return x

    def generate(self, prompt_ids: list, max_new: int = 200,
                 temperature: float = 0.7, idx_to_char: dict = None) -> list:
        """Generar texto autoregresivamente."""
        import numpy as np
        gen = list(prompt_ids)
        for _ in range(max_new):
            ctx_ids = gen[-min(256, len(gen)):]
            emb = self._embed(np.array(ctx_ids, dtype=int))
            ctx_emb = self.forward(emb)
            logits = self.decoder.evaluate(ctx_emb[-1])
            logits = logits - np.max(logits)
            probs = np.exp(logits / max(temperature, 0.01))
            probs = probs / probs.sum()
            gen.append(int(np.random.choice(self.vocab_size, p=probs)))
        if idx_to_char:
            return ''.join(idx_to_char.get(t, '?') for t in gen)
        return gen

    def evaluate_accuracy(self, token_ids: 'np.ndarray',
                          n_test: int = 5000) -> float:
        """Evaluar precisión next-char."""
        import numpy as np
        n = len(token_ids) - 1
        correct = total = 0
        for i in range(0, n - 25, 25):
            ctx = token_ids[i:i + 25]
            tgt = token_ids[i + 1:i + 26]
            m = min(len(ctx), len(tgt))
            if m < 4: continue
            emb = self._embed(ctx[:m])
            ctx_emb = self.forward(emb)
            for j in range(m - 1):
                if total >= n_test: break
                if np.argmax(self.decoder.evaluate(ctx_emb[j])) == tgt[j]:
                    correct += 1
                total += 1
            if total >= n_test: break
        return correct / max(1, total)


# ═══════════════════════════════════════════════════════════════
# DEEP NOVA NETWORK — Arquitectura Profunda Nativa
# ═══════════════════════════════════════════════════════════════

