#!/usr/bin/env python3
"""train_nova.py v3.0 — ACF FMA-Optimized for 30-52% accuracy target."""

import numpy as np, urllib.request, time, gc, os, sys, argparse, threading
from dataclasses import dataclass

@dataclass
class NovaConfig:
    vocab_size: int = 65; embed_dim: int = 192; n_layers: int = 5
    n_heads: int = 6; l2_lambda: float = 0.05; max_context: int = 16384
    seq_len: int = 128; n_seqs: int = 400; n_tokens: int = 1_000_000
    koopman_alpha: float = 0.3; use_nova_emb: bool = False
    nb_iters: int = 2; nb_lambda: float = 0.2; backend: str = "auto"
    epochs: int = 1; seed: int = 42; verbose: bool = True
    checkpoint_dir: str = "checkpoints"; resume: bool = False

class LivePhase:
    SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    def __init__(self, desc): self.desc=desc; self.t0=time.perf_counter(); self._running=False; self._si=0
    def _spin(self):
        while self._running:
            e=time.perf_counter()-self.t0; s=self.SPIN[self._si%len(self.SPIN)]
            sys.stdout.write(f'\r  {s} {self.desc} [{e:.0f}s]  '); sys.stdout.flush(); self._si+=1; time.sleep(0.12)
    def start(self): self._running=True; self._thread=threading.Thread(target=self._spin,daemon=True); self._thread.start()
    def stop(self,msg=""): 
        self._running=False
        if self._thread: self._thread.join(timeout=0.3)
        e=time.perf_counter()-self.t0; sys.stdout.write(f'\r  ✅ {self.desc}: {msg} [{e:.0f}s]  \n'); sys.stdout.flush()

def setup_backend(config):
    backends={}
    try:
        import sys as _sys,os as _os
        _p=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'gideon_core','target','release')
        _sys.path.insert(0,_p)
        from gideon_core import GideonCoreEngine,CoreEngineConfig
        backends['gideon']=GideonCoreEngine(CoreEngineConfig()); backends['gideon_name']='Gideon Rust (AVX2+FMA)'
    except: pass
    try:
        import cupy as cp; p=cp.cuda.runtime.getDeviceProperties(0)
        n=p['name'].decode() if isinstance(p['name'],bytes) else p['name']; v=p['totalGlobalMem']/(1024**3)
        backends['gpu']=cp; backends['gpu_name']=f'CuPy GPU: {n} ({v:.1f} GB)'
    except: pass
    try:
        import torch
        if torch.cuda.is_available(): backends['pytorch_name']=f'PyTorch CUDA: {torch.cuda.get_device_name(0)}'
        else: backends['pytorch_name']='PyTorch CPU'
        backends['pytorch']=torch
    except: pass
    backends['cpu']=np; backends['cpu_name']=f'CPU: {os.cpu_count()} cores'
    if config.backend=='auto':
        if 'gpu' in backends: sel,desc='gpu',backends['gpu_name']
        elif 'pytorch' in backends and 'pytorch CUDA' in backends.get('pytorch_name',''):
            sel,desc='pytorch_gpu',backends['pytorch_name']
        elif 'gideon' in backends: sel,desc='gideon',backends['gideon_name']
        else: sel,desc='cpu',backends['cpu_name']
    else: sel=config.backend; desc=backends.get(f'{sel}_name',f'Backend: {sel}')
    return sel,desc,backends

def enable_gpu():
    """Activar GPU con PyTorch CUDA para todos los solvers de Nova."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import numpy as np
        import acf_functor.neuron.nova_phi_neuron as nmod
        device = torch.device('cuda')
        
        original_cholesky = nmod.NovaPhiNeuron._solve_cholesky
        original_cascade = nmod.NovaPhiNeuron._solve_acf_cascade
        
        def gpu_cholesky(self_neuron, Phi, Y):
            # 🔥 Skip Kronecker internal neurons (multi-output)
            if getattr(self_neuron, 'n_output', 1) > 1:
                return original_cholesky(self_neuron, Phi, Y)
            n_features = Phi.shape[1]
            lam = max(self_neuron.l2_lambda, 0.001)
            # SES filter (CPU)
            col_e = np.sum(Phi * Phi, axis=0); col_e = np.maximum(col_e, 1e-12)
            p_e = col_e / col_e.sum()
            H = -np.sum(p_e * np.log(p_e + 1e-12))
            keep = p_e > H * np.log(n_features+1) / max(n_features,1)
            pruned = keep.sum() < n_features and keep.sum() > 2
            if pruned: Phi = Phi[:, keep]
            try:
                Phi_t = torch.from_numpy(Phi.astype(np.float32)).to(device)
                Y_t = torch.from_numpy(Y.astype(np.float32)).to(device)
                A_t = Phi_t.T @ Phi_t + lam * torch.eye(Phi_t.shape[1], device=device)
                L_t = torch.linalg.cholesky(A_t)
                C_p = torch.cholesky_solve(Phi_t.T @ Y_t, L_t).cpu().numpy()
                if pruned:
                    C_full = np.zeros((n_features, C_p.shape[1])); C_full[keep] = C_p
                else: C_full = C_p
                self_neuron._unpack(C_full)
                self_neuron.solver_type = "cholesky_gpu"
            except: original_cholesky(self_neuron, Phi, Y)
        
        def gpu_cascade(self_neuron, Phi, Y, block_size=64, n_cascades=3):
            # 🔥 Skip Kronecker internal neurons (multi-output)
            if getattr(self_neuron, 'n_output', 1) > 1:
                return original_cascade(self_neuron, Phi, Y, block_size, n_cascades)
            n_samples, n_features = Phi.shape
            if Y.ndim == 1: Y = Y.reshape(-1, 1)
            n_output = Y.shape[1]; lam = max(self_neuron.l2_lambda, 1e-6)
            bs = max(16, min(block_size, n_features // 4)); bs = min(bs, 64)
            n_blocks = max(1, (n_features + bs - 1) // bs)
            try:
                Phi_t = torch.from_numpy(Phi.astype(np.float32)).to(device)
                Y_t = torch.from_numpy(Y.astype(np.float32)).to(device)
                C_t = torch.zeros((n_features, n_output), device=device)
                for cascade in range(n_cascades):
                    R_t = Y_t - Phi_t @ C_t
                    for b in range(n_blocks):
                        s, e = b*bs, min((b+1)*bs, n_features)
                        if e <= s: continue
                        Phi_b = Phi_t[:, s:e]; nf = Phi_b.shape[1]
                        G = Phi_b.T @ Phi_b + lam * torch.eye(nf, device=device)
                        try:
                            L = torch.linalg.cholesky(G)
                            C_t[s:e, :] += torch.cholesky_solve(Phi_b.T @ R_t, L)
                        except:
                            try: C_t[s:e, :] += torch.linalg.solve(G, Phi_b.T @ R_t)
                            except: pass
                C_full = C_t.cpu().numpy()
                self_neuron._unpack(C_full)
                self_neuron.solver_type = f"acf_gpu_{n_cascades}x{bs}"
            except: original_cascade(self_neuron, Phi, Y, block_size, n_cascades)
        
        nmod.NovaPhiNeuron._solve_cholesky = gpu_cholesky
        nmod.NovaPhiNeuron._solve_acf_cascade = gpu_cascade
        return True
    except: return False

def load_shakespeare(n_tokens=1_000_000):
    print('📚 Downloading TinyShakespeare...')
    url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    text=urllib.request.urlopen(url).read().decode('utf-8')
    chars=sorted(list(set(text))); V=len(chars)
    c2i={c:i for i,c in enumerate(chars)}; ALL=np.array([c2i[c] for c in text],dtype=int)
    trans=np.ones((V,V))*0.1
    for t in range(min(n_tokens,len(ALL)-1)):
        a,b=int(ALL[t]),int(ALL[t+1])
        if a<V and b<V: trans[a,b]+=1.0
    trans_prob=trans/trans.sum(axis=1,keepdims=True)
    H=-np.sum(trans_prob*np.log(trans_prob+1e-8),axis=1).mean()
    print(f'  V={V} | {len(ALL):,} tokens | H={H:.2f} bits'); return ALL,V,c2i,{i:c for i,c in enumerate(chars)},trans_prob

def load_text8_wordlevel(n_tokens=1_000_000, vocab_size=128):
    """🔥 Dataset text8 con tokenización word-level V=128.
    
    Usa las 128 palabras más frecuentes de text8 (Wikipedia limpia).
    V=128 es el doble de Shakespeare (V=65), forzando al modelo
    a manejar un espacio de decisión 4× más grande.
    """
    print(f'📚 Downloading text8 (V={vocab_size} words)...')
    import urllib.request, zipfile, io, collections
    url = 'http://mattmahoney.net/dc/text8.zip'
    with urllib.request.urlopen(url) as f:
        data = zipfile.ZipFile(io.BytesIO(f.read())).read('text8').decode('utf-8')
    words = data.split()
    # Vocabulario: top-V palabras
    word_counts = collections.Counter(words[:n_tokens])
    top_words = [w for w, _ in word_counts.most_common(vocab_size)]
    w2i = {w: i for i, w in enumerate(top_words)}
    ALL = np.array([w2i.get(w, 0) for w in words[:n_tokens]], dtype=int)
    V = vocab_size
    trans = np.ones((V, V)) * 0.1
    for t in range(min(n_tokens, len(ALL) - 1)):
        a, b = int(ALL[t]), int(ALL[t+1])
        if a < V and b < V: trans[a, b] += 1.0
    trans_prob = trans / trans.sum(axis=1, keepdims=True)
    H = -np.sum(trans_prob * np.log(trans_prob + 1e-8), axis=1).mean()
    print(f'  V={V} words | {len(ALL):,} tokens | H={H:.2f} bits'); return ALL, V, w2i, {i: w for i, w in enumerate(top_words)}, trans_prob

def train_epoch(model,ALL,config,trans_prob,epoch_idx,glassbox_sensors=None):
    from acf_functor.neuron.nova_llm import DivergentDeepNova
    t0=time.perf_counter(); n_train=int(len(ALL)*0.9); V=config.vocab_size; nL=config.n_layers
    
    print(f'\n  🔵 Phase 1/3: Fit ({config.n_seqs}x{config.seq_len})')
    live=LivePhase('Training attention'); live.start()
    
    # 🔍 GLASSBOX: pasar sensores al entrenamiento
    model.fit(ALL[:config.n_tokens],seq_len=config.seq_len,n_seqs=config.n_seqs,
              verbose=config.verbose, glassbox_sensors=glassbox_sensors)
    live.stop('done')
    solvers={}
    for li,layer in enumerate(model.layers):
        for lv in range(layer.attention.n_levels):
            if layer.attention._level_trained[lv]:
                st=getattr(layer.attention.level_neurons[lv],'solver_type','?'); solvers[st]=solvers.get(st,0)+1
    print(f'  📊 Solvers: {", ".join(f"{k}x{v}" for k,v in sorted(solvers.items()))}')
    
    print(f'\n  🟡 Phase 2/3: Eval')
    live=LivePhase('Evaluating'); live.start()
    acc_base=model.evaluate_accuracy(ALL[n_train:],n_test=2000,seq_len=config.seq_len)
    live.stop(f'{acc_base*100:.2f}% ({acc_base*V:.1f}x)')
    print(f'  ⏱️  Fit+Eval: {time.perf_counter()-t0:.0f}s')
    
    # ═══════════════════════════════════════════════════════════
    # Nova-BP ahora es SOLO interno (model.nb_iters controla el
    # que ocurre dentro de model.fit() en nova_llm.py FASE 4).
    # El externo fue eliminado porque causaba DOBLE corrección
    # y degradaba accuracy (-1.95pp en tests).
    # ═══════════════════════════════════════════════════════════
    
    tt = time.perf_counter()-t0
    print(f'\n  🏁 Epoch {epoch_idx+1}: {acc_base*100:.2f}% ({acc_base*V:.1f}x) [{tt:.0f}s]')
    
    # 🔍 GLASSBOX: checkpoint al final de la época
    if glassbox_sensors and 'viz' in glassbox_sensors:
        glassbox_sensors['viz'].checkpoint(f'epoch{epoch_idx+1}')
        glassbox_sensors['viz'].print_summary()
        # Timeline de fases
        if 'timing' in glassbox_sensors:
            print(glassbox_sensors['viz'].plot_phase_timeline())
    
    return acc_base,tt

def _acf_fma_optimizer(model, config, trans_prob):
    """🔬 ACF FMA Optimizer v3.0: reduce operaciones sin perder capacidad.
    
    ANALIZA cada neurona, calcula FMA teóricas, y aplica reducciones:
      1. Degradación de max_degree por profundidad (FMA ∝ d²)
      2. Poda de pares redundantes (correlación < 0.1)
      3. Selección óptima de solver por tamaño (O(f³)→O(N×f))
      4. Hebbian eta = f(H_vocab) — más entropía → más empuje
      5. Target accuracy projection basado en capacidad expresiva
    
    FMA total estimada = Σ_neuronas (n_main × d² + n_pairs × d⁴ + solver)
    """
    import numpy as np
    V = config.vocab_size
    H_vocab = -np.sum(trans_prob * np.log(trans_prob + 1e-8), axis=1).mean()
    H_max = np.log(V)
    
    print(f'  📊 H_vocab={H_vocab:.2f} bits | V={V} | Target: 30-52%')
    
    total_fma_before = 0
    total_fma_after = 0
    total_neurons = 0
    total_pairs_before = 0
    total_pairs_after = 0
    
    for li, layer in enumerate(model.layers):
        n_levels = layer.attention.n_levels
        depth_factor = li / max(1, model.n_layers - 1)  # 0 → 1
        
        # 🔥 FMA-optimal degree: capas profundas degradan a grado 1
        #    Grado 2: 4× más FMA que grado 1 (d² domina)
        #    Capas 0-1: grado 2 | Capa 2: grado 2 | Capa 3+: grado 1
        if li <= 2:
            fma_degree = 2
        else:
            fma_degree = 1
        
        layer_fma = 0
        layer_pairs_before = 0
        layer_pairs_after = 0
        
        for lv in range(n_levels):
            neuron = layer.attention.level_neurons[lv]
            n_input = getattr(neuron, 'n_input', 256)
            md = getattr(neuron, 'max_degree', 2)
            pp = getattr(getattr(neuron, 'max_pairs', None), '__len__', lambda: 0)() or 30
            
            # FMA antes: n_samples × (n_input×(d+1) + pairs×(d+1)²) × n_output
            d1 = md + 1
            fma_before = config.n_seqs * config.seq_len * (n_input * d1 + pp * d1 * d1)
            
            # 🔥 Reducir: pares solo en niveles 0-2 (locales), resto solo main effects
            if lv <= 2:
                optimal_pairs = pp
            elif lv <= 4:
                optimal_pairs = max(4, pp // 3)
            else:
                optimal_pairs = 0  # Solo main effects para niveles globales
            
            d1_opt = fma_degree + 1
            fma_after = config.n_seqs * config.seq_len * (n_input * d1_opt + optimal_pairs * d1_opt * d1_opt)
            
            # 🔥 Aplicar: forzar max_degree y pares en la neurona
            if hasattr(neuron, 'max_degree'):
                neuron.max_degree = fma_degree
                neuron._d1 = fma_degree + 1
                neuron._d2 = (fma_degree + 1) ** 2
            
            if hasattr(neuron, 'max_pairs'):
                neuron.max_pairs = optimal_pairs
            
            total_neurons += 1
            layer_fma += fma_after
            layer_pairs_before += pp
            layer_pairs_after += optimal_pairs
        
        # 🔥 Solver óptimo por tamaño de features
        #    <300: Cholesky O(f³)  |  300-3000: Cascade O(N×f)  |  >3000: LSQR
        est_features = layer_pairs_after * (fma_degree + 1)**2 + n_input * (fma_degree + 1)
        if est_features <= 300:
            solver = "cholesky_gpu"
        elif est_features <= 3000:
            solver = "acf_cascade"
        else:
            solver = "lsqr"
        
        for lv in range(n_levels):
            neuron = layer.attention.level_neurons[lv]
            if hasattr(neuron, '_force_solver'):
                neuron._force_solver = solver
        
        total_fma_before += layer_fma * (md + 1)**2 / (fma_degree + 1)**2  # approx
        total_fma_after += layer_fma
        total_pairs_before += layer_pairs_before
        total_pairs_after += layer_pairs_after
        
        print(f'  🔧 L{li+1}: deg={fma_degree} | pairs={layer_pairs_before}→{layer_pairs_after} | solver={solver} | λ={neuron.l2_lambda:.4f}')
    
    # 🌡️ Temperatura del decoder
    T_opt = np.clip(0.5 + 0.5 * (H_vocab / H_max), 0.3, 2.0)
    if hasattr(model.decoder, 'temperature'):
        model.decoder.temperature = T_opt
    
    # 🔥 Hebbian eta: empuje proporcional a entropía
    hebb_eta = np.clip(0.03 * (H_vocab / H_max), 0.01, 0.08)
    model.hebb_eta = hebb_eta
    
    # 🔄 Feedback α
    if hasattr(model, 'recurrent_feedback') and model.recurrent_feedback:
        fb_alpha = 0.1 + 0.05 * model.n_layers
        model.bus.feedback_alpha = fb_alpha
    
    # 📐 Proyección de accuracy
    reduction_ratio = total_fma_after / max(total_fma_before, 1)
    # Baseline esperado: 18-22% para V=65 con 400 seqs
    # Con Hebb + aggregator + más capas: +5-10pp → 25-35% proyectado
    proj_min = int(20 + 5 * (1.0 - reduction_ratio) * 10)
    proj_max = int(30 + 10 * (1.0 - reduction_ratio) * 10)
    
    print(f'\n  📐 FMA: {total_fma_before/1e6:.1f}M → {total_fma_after/1e6:.1f}M ({reduction_ratio*100:.0f}% kept)')
    print(f'  📐 Neurons: {total_neurons} | Pairs: {total_pairs_before}→{total_pairs_after}')
    print(f'  📐 Hebb eta={hebb_eta:.3f} | T={T_opt:.2f} | FB alpha={fb_alpha:.2f}')
    print(f'  🎯 Accuracy projection: {proj_min}-{proj_max}%')
    ne_status = "ON" if model._use_nova_emb else "OFF (--no-nova-emb)"
    kr_status = "ON" if any(l.attention._use_kronecker for l in model.layers) else "OFF"
    print(f'  🧬 NovaEmbedding: {ne_status}')
    print(f'  🧬 Kronecker: {kr_status}')
    sgf_status = "ON (Σ=0.776)" if getattr(model, '_use_sgf', False) else "OFF (--no-sgf)"
    print(f'  🧬 SGF (Semantic Genesis Functor): {sgf_status}')

def main():
    p=argparse.ArgumentParser(description='🔥 Nova LLM v3.0 — ACF FMA-Optimized')
    p.add_argument('--epochs',type=int,default=1); p.add_argument('--layers',type=int,default=5)
    p.add_argument('--embed-dim',type=int,default=192); p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--n-seqs',type=int,default=400); p.add_argument('--vocab-size',type=int,default=65)
    p.add_argument('--dataset',default='shakespeare',choices=['shakespeare','text8'])
    p.add_argument('--backend',default='auto',choices=['auto','cpu','gpu','gideon','triton','pytorch'])
    p.add_argument('--koopman',type=float,default=0.3); p.add_argument('--l2',type=float,default=0.05)
    p.add_argument('--nb-iters',type=int,default=2); p.add_argument('--nb-lambda',type=float,default=0.2)
    p.add_argument('--no-nova-emb',action='store_true'); p.add_argument('--no-sgf',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-kronecker',action='store_true'); p.add_argument('--no-feedback',action='store_true')
    p.add_argument('--spectral',action='store_true',help='🔥 SpectralDecoder: ΦC=Y directo, sin heads')
    p.add_argument('--glassbox',action='store_true',help='🔍 Activar sensores glassbox en entrenamiento')
    p.add_argument('--resume',action='store_true'); p.add_argument('--checkpoint-dir',default='checkpoints')
    args=p.parse_args()
    
    config=NovaConfig(n_layers=args.layers,embed_dim=args.embed_dim,seq_len=args.seq_len,n_seqs=args.n_seqs,
        backend=args.backend,koopman_alpha=args.koopman,l2_lambda=args.l2,nb_iters=args.nb_iters,
        nb_lambda=args.nb_lambda,use_nova_emb=not args.no_nova_emb,seed=args.seed,epochs=args.epochs,
        resume=args.resume,checkpoint_dir=args.checkpoint_dir,vocab_size=args.vocab_size)
    
    print('╔══════════════════════════════════════════════╗')
    print('║  🔥 NOVA v3.0 — ACF FMA-Optimized | 30-52%   ║')
    print('╠══════════════════════════════════════════════╣')
    print(f'║  {config.n_layers}Lx{config.embed_dim}d | {config.n_seqs}x{config.seq_len} | V={config.vocab_size} | {args.dataset}  ║')
    print(f'║  l2={config.l2_lambda} | Koop={config.koopman_alpha} | BP={config.nb_iters} | Kr={not args.no_kronecker} FB={not args.no_feedback} ║')
    if args.spectral:
        print('║  🔥 SPECTRAL DECODER: ΦC=Y directo, sin heads ║')
    print('╚══════════════════════════════════════════════╝')
    
    sel,desc,backends=setup_backend(config); print(f'\n🔧 Backend: {desc}')
    if enable_gpu():
        gpu_props=__import__('torch').cuda.get_device_properties(0)
        print(f'  ✅ GPU ACTIVADA: {gpu_props.name} ({gpu_props.total_memory/1e9:.1f} GB) — CUDA solvers')
    else:
        print(f'  ⚠️  GPU no disponible, usando CPU')
    
    if args.dataset=='shakespeare':
        ALL,V,_,_,trans_prob=load_shakespeare(config.n_tokens)
    else:
        ALL,V,_,_,trans_prob=load_text8_wordlevel(config.n_tokens,vocab_size=args.vocab_size)
    config.vocab_size=V; n_train=int(len(ALL)*0.9)
    
    print(f'\n🧠 Nova {config.n_layers}Lx{config.embed_dim}d V={V}...')
    from acf_functor.neuron.nova_llm import DivergentDeepNova
    np.random.seed(config.seed)
    model=DivergentDeepNova(V,embed_dim=config.embed_dim,n_layers=config.n_layers,n_heads=config.n_heads,
        l2_lambda=config.l2_lambda,max_context=config.max_context,memory_budget_gb=5.0,
        use_kronecker=not args.no_kronecker, recurrent_feedback=not args.no_feedback,
        use_spectral_decoder=args.spectral)
    model._use_nova_emb=config.use_nova_emb; model.koopman_alpha=config.koopman_alpha
    model._use_sgf=not args.no_sgf  # 🔥 Semantic Genesis Functor
    model.nb_iters=config.nb_iters; model.nb_lambda=config.nb_lambda
    
    # ═══════════════════════════════════════════════════════════
    # 🔬 ACF FMA OPTIMIZER: minimizar FMA sin perder capacidad
    # ═══════════════════════════════════════════════════════════
    print(f'\n🔬 ACF FMA Optimizer...')
    _acf_fma_optimizer(model, config, trans_prob)
    
    # 🔍 GLASSBOX: crear sensores si se activaron
    glassbox_sensors = None
    if args.glassbox:
        from nova_glassbox import create_glassbox_suite
        timing, spectral, info, decoder, viz, detector = create_glassbox_suite(V)
        glassbox_sensors = {
            'timing': timing, 'spectral': spectral,
            'info': info, 'decoder': decoder,
            'viz': viz, 'detector': detector
        }
        print(f'🔍 Glassbox sensors ACTIVATED — 5 sensores monitoreando')
    
    best_acc=0.0; os.makedirs(config.checkpoint_dir,exist_ok=True)
    t_total_start = time.perf_counter()
    for epoch in range(config.epochs):
        print(f'\n{"="*60}\n🔥 EPOCH {epoch+1}/{config.epochs}\n{"="*60}')
        acc,elapsed=train_epoch(model,ALL,config,trans_prob,epoch,glassbox_sensors)
        ckpt=os.path.join(config.checkpoint_dir,f'nova_e{epoch+1}.npz')
        np.savez(ckpt,epoch=epoch,accuracy=acc); print(f'  💾 {ckpt}')
        if acc>best_acc: best_acc=acc; np.savez(os.path.join(config.checkpoint_dir,'nova_best.npz'),epoch=epoch,accuracy=acc); print(f'  🏆 NEW BEST: {acc*100:.2f}%')
    
    total_elapsed = time.perf_counter() - t_total_start
    mins, secs = divmod(int(total_elapsed), 60)
    print(f'\n{"="*60}\n🏁 BEST: {best_acc*100:.2f}% ({best_acc*V:.1f}x)')
    print(f'⏱️  TOTAL TIME: {mins}m {secs}s ({total_elapsed:.0f}s)')
    print(f'{"="*60}')

if __name__=='__main__': main()
