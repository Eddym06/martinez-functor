"""
nova_face_ui.py — UI Nova Face + Webcam Tracker
Backend: NovaConv2D ANOVA(2). Sin CNN, sin PyTorch.
"""

import sys, os, time, warnings
import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acf_functor.neuron.nova_vision_v2 import NovaConv2D
import cv2, gradio as gr

try:
    from pillow_heif import register_heif_opener; register_heif_opener()
except: pass

class FaceDetector:
    def __init__(self):
        self.c = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    def detect(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim==3 else img
        return self.c.detectMultiScale(g, 1.1, 5, minSize=(30,30))
    def extract_face(self, img, sz=64):
        fs = self.detect(img)
        if len(fs)==0: return None
        x,y,w,h = fs[0]
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim==3 else img
        return cv2.resize(g[y:y+h,x:x+w], (sz,sz)).astype(np.float64)/255.0

class NovaFaceEngine:
    def __init__(self, fs=64, fd=32):
        self.fs=fs; self.fd=fd
        self.conv = NovaConv2D(1,fd,7,3,2,max_degree=2,l2_lambda=0.1,max_pairs=40)
        self._t=False; self._d=FaceDetector()
    def train_on_olivetti(self):
        import pickle
        cp = os.path.join(os.path.dirname(os.path.abspath(__file__)),'.nova_face_cache.pkl')
        if os.path.exists(cp):
            with open(cp,'rb') as f: self.conv = pickle.load(f)['conv']
            if not hasattr(self.conv, '_gpu_cache') or self.conv._gpu_cache is None:
                self.conv._build_gpu_cache()
            self._t=True; return {'total_features':0}
        from sklearn.datasets import fetch_olivetti_faces
        X = fetch_olivetti_faces(shuffle=True,random_state=42).images.astype(np.float64)
        if self.fs!=64: X=np.array([cv2.resize(x,(self.fs,self.fs)) for x in X])
        t0=time.perf_counter(); r=self.conv.fit(X)
        print(f"[trained {time.perf_counter()-t0:.0f}s]")
        self._t=True
        with open(cp,'wb') as f: pickle.dump({'conv':self.conv},f)
        return r
    def extract_features(self, face):
        if face.shape!=(self.fs,self.fs): face=cv2.resize(face,(self.fs,self.fs))
        return self.conv.forward(face).mean(axis=(1,2))
    def process(self, img):
        face=self._d.extract_face(img,self.fs)
        if face is None: return None,None,True,"No cara"
        return face,self.extract_features(face),False,"OK"
    def compare(self, ref, qry):
        nr,nq=np.linalg.norm(ref)+1e-10,np.linalg.norm(qry)+1e-10
        c=float(np.dot(ref,qry)/(nr*nq)); d=1.0-float(np.linalg.norm(ref-qry))/(nr+nq)
        s=0.6*max(0,c)+0.4*max(0,d); return s,s>0.55,d,False

_SESSION={"ref":None}

def create_ui(engine):
    def proc_ref(img):
        if img is None: return None,"Sube foto"
        face,feats,_,msg=engine.process(img)
        if face is None: return img,msg
        _SESSION["ref"]=feats
        # Mostrar cara a COLOR con interpolación suave
        fs = engine._d.detect(img)
        if len(fs)>0:
            x,y,w,h = fs[0]
            if img.ndim==3: color_crop = img[y:y+h, x:x+w]
            else: color_crop = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB)
            return Image.fromarray(color_crop).resize((192,192), Image.LANCZOS),f"OK|{len(feats)}d"
        return Image.fromarray((face*255).astype(np.uint8)).resize((192,192), Image.LANCZOS),f"OK|{len(feats)}d"
    def proc_single(img):
        ref=_SESSION.get("ref")
        if ref is None: return img,"Referencia primero"
        if img is None: return None,"Sube foto"
        face,feats,_,msg=engine.process(img)
        if face is None: return img,msg
        sc,match,_,_=engine.compare(ref,feats)
        d=img.copy()
        if d.ndim==2: d=cv2.cvtColor(d,cv2.COLOR_GRAY2RGB)
        color=(0,255,0) if match else (255,0,0)
        cv2.putText(d,f"{'MATCH' if match else 'NO'} {sc:.0%}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        for(x,y,w,h) in engine._d.detect(img): cv2.rectangle(d,(x,y),(x+w,y+h),color,2)
        return d,f"{'MATCH' if match else 'NO'}|{sc:.0%}"
    def scan(files):
        ref=_SESSION.get("ref")
        if ref is None: return [],"Referencia primero"
        if not files: return [],"Sube fotos"
        res=[]
        for item in files:
            try: img=np.array(Image.open(item if isinstance(item,str) else item.name).convert('RGB')) if not isinstance(item,np.ndarray) else item
            except: res.append((Image.fromarray(np.zeros((64,64,3),dtype=np.uint8)),"Error")); continue
            face,feats,_,msg=engine.process(img)
            if face is None: res.append((Image.fromarray(np.zeros((64,64,3),dtype=np.uint8)),msg[:20])); continue
            sc,match,_,_=engine.compare(ref,feats)
            # Mostrar cara a COLOR con interpolación suave
            fs = engine._d.detect(img)
            if len(fs)>0:
                x,y,w,h = fs[0]
                if img.ndim==3: color_crop = img[y:y+h, x:x+w]
                else: color_crop = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB)
                face_img = Image.fromarray(color_crop).resize((128,128), Image.LANCZOS)
            else:
                face_img = Image.fromarray((face*255).astype(np.uint8)).resize((128,128), Image.LANCZOS)
            res.append((face_img,f"{'MATCH' if match else 'No'} {sc:.0%}"))
        return res,f"Escaneadas {len(files)}"
    def _auto_track(frame, state):
        if frame is None: return None,0.0,"Camara apagada. Haz clic en el icono del recuadro Webcam.",state
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY); H,W=gray.shape
        if state is None or not state.get('trained'):
            s=64; y1,x1=(H-s)//2,(W-s)//2
            pf=gray[y1:y1+s,x1:x1+s].astype(np.float64)/255.0
            conv=NovaConv2D(1,12,5,2,2,max_degree=2,l2_lambda=0.1,max_pairs=15)
            conv.fit(np.tile(pf[None,:,:],(8,1,1))+np.random.randn(8,s,s)*0.02)
            ref=conv.forward(pf).mean(axis=(1,2))
            state={'ref_features':ref,'conv':conv,'roi_size':s,'trained':True}
            d=frame.copy(); cv2.rectangle(d,(x1,y1),(x1+s,y1+s),(255,255,0),3)
            cv2.putText(d,"REF OK - MUEVETE!",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            return d,1.0,"Capturado! Muevete...",state
        s=state['roi_size'];conv=state['conv'];ref=state['ref_features']
        fm=conv.forward(gray.astype(np.float64)/255.0)
        rn=ref/(np.linalg.norm(ref)+1e-10);fn=fm/(np.linalg.norm(fm,axis=0)+1e-10)[None,:,:]
        sm=np.tensordot(rn,fn,axes=([0],[0]));idx=np.argmax(sm)
        by,bx=np.unravel_index(idx,sm.shape);sc=float(sm[by,bx])
        cx=int(bx*2+s//2);cy=int(by*2+s//2)
        cx=max(s//2,min(W-s//2,cx));cy=max(s//2,min(H-s//2,cy))
        d=frame.copy();color=(0,255,0)if sc>0.5 else(0,165,255)
        cv2.rectangle(d,(cx-s//2,cy-s//2),(cx+s//2,cy+s//2),color,2)
        cv2.circle(d,(cx,cy),6,color,-1)
        cv2.putText(d,f"Score:{sc:.0%}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
        return d,float(sc),f"Score:{sc:.0%}|({cx},{cy})",state
    with gr.Blocks(title="Nova Vision",theme=gr.themes.Soft()) as app:
        gr.Markdown("# Nova Vision — Face + Tracker")
        ts=gr.State(None)
        with gr.Tabs():
            with gr.TabItem("Face Recognition"):
                with gr.Row():
                    with gr.Column():
                        ri=gr.Image(label="Referencia",type="numpy")
                        rb=gr.Button("Fijar referencia",variant="primary")
                        rd=gr.Image(label="Cara",width=128,height=128)
                        rinfo=gr.Markdown("")
                    with gr.Column():
                        si=gr.Image(label="Comparar",type="numpy")
                        so=gr.Image(label="Resultado")
                        sinfo=gr.Markdown("")
                gr.Markdown("---")
                gi=gr.File(label="Galeria",file_count="multiple",file_types=None)
                sb=gr.Button("Escanear",variant="primary")
                go=gr.Gallery(label="Resultados",columns=4,height="auto",object_fit="contain")
                ginfo=gr.Markdown("")
            with gr.TabItem("Webcam Tracker"):
                gr.Markdown("### Tracking en vivo\n1. Haz clic en el icono del recuadro Webcam\n2. Muestra tu puno centrado\n3. Muevete!")
                with gr.Row():
                    with gr.Column():
                        wi=gr.Image(sources=["webcam"],streaming=True,label="Webcam",type="numpy")
                        rst=gr.Button("Reset")
                    with gr.Column():
                        to=gr.Image(label="Tracking")
                        tsc=gr.Number(label="Score",value=0.0,precision=4)
                        tst=gr.Markdown("Enciende la camara...")
        rb.click(proc_ref,[ri],[rd,rinfo])
        si.change(proc_single,[si],[so,sinfo])
        sb.click(scan,[gi],[go,ginfo])
        wi.stream(_auto_track,[wi,ts],[to,tsc,tst,ts],time_limit=60,concurrency_limit=1,stream_every=0.3)
        rst.click(lambda:(None,0.0,"Listo.",None),[],[to,tsc,tst,ts])
    return app

def main():
    engine=NovaFaceEngine(face_size=32,feature_dim=16)
    engine.train_on_olivetti()
    create_ui(engine).launch(server_name="127.0.0.1",server_port=7860,share=False)

if __name__=="__main__":
    main()
