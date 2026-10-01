import os, json, time, uuid, hashlib, sqlite3
import numpy as np
from PIL import Image
import imagehash
from sklearn.ensemble import IsolationForest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

# ---------- utils ----------
def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

# ---------- 2.1 DATA INTEGRITY ----------
def phash(p):
    return imagehash.phash(Image.open(p).convert("RGB"))

def find_duplicates(paths, max_dist=5):
    hs = [(p, phash(p)) for p in paths]
    out = []
    for i in range(len(hs)):
        for j in range(i+1, len(hs)):
            d = hs[i][1] - hs[j][1]
            if d <= max_dist: out.append((hs[i][0], hs[j][0], int(d)))
    return out

def label_anomalies(samples):
    """samples: list of dict {id, contributor, label}"""
    from collections import Counter, defaultdict
    per = defaultdict(Counter)
    for s in samples: per[s["contributor"]][s["label"]] += 1
    flags = []
    for c, cnt in per.items():
        total = sum(cnt.values())
        for lbl, n in cnt.items():
            share = n/total
            if share > 0.85 and total > 20:
                flags.append({"contributor": c, "reason": f"label {lbl} dominates {share:.0%}", "severity": "medium"})
    return flags

def ood_flags(paths, ref_paths=None, contamination=0.05):
    def feat(p):
        im = np.asarray(Image.open(p).convert("RGB").resize((32,32))).flatten()/255.0
        return im
    X = np.array([feat(p) for p in paths])
    if ref_paths and len(ref_paths) >= 20:
        Xr = np.array([feat(p) for p in ref_paths]); clf = IsolationForest(contamination=contamination, random_state=0).fit(Xr)
    else:
        clf = IsolationForest(contamination=contamination, random_state=0).fit(X)
    pred = clf.predict(X)
    return [paths[i] for i, p in enumerate(pred) if p == -1]

def trigger_scan(paths):
    """Detect small bright/high-contrast corner patches."""
    flags = []
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"))
        h, w, _ = im.shape
        corner = im[h-24:h, w-24:w]
        # high contrast small patch = suspicious
        if corner.std() > 60 and corner.mean() > 180:
            flags.append(p)
    return flags

def source_risk(samples, dup_pairs, ood_paths, trig_paths):
    from collections import defaultdict
    score = defaultdict(lambda: {"dup":0,"ood":0,"trig":0,"n":0})
    for s in samples: score[s["contributor"]]["n"] += 1
    id2c = {s["id"]: s["contributor"] for s in samples}
    for a,b,_ in dup_pairs:
        for x in (a,b):
            if x in id2c: score[id2c[x]]["dup"] += 1
    for p in ood_paths:
        if p in id2c: score[id2c[p]]["ood"] += 1
    for p in trig_paths:
        if p in id2c: score[id2c[p]]["trig"] += 1
    out = []
    for c, v in score.items():
        n = max(v["n"],1)
        risk = (v["dup"]*1.0 + v["ood"]*1.5 + v["trig"]*2.5)/n
        sev = "low" if risk < 0.05 else "medium" if risk < 0.15 else "high"
        out.append({"contributor": c, "risk_score": round(risk,3), "severity": sev,
                    "dup": v["dup"], "ood": v["ood"], "trig": v["trig"], "n": v["n"]})
    return out

# ---------- 2.2 MODEL INTEGRITY ----------
def model_digest(path): return sha256_file(path)

def behavioural_fingerprint(model_path, image_paths, access="blackbox"):
    """Runs ONNX or TorchScript on ref battery, returns output signature."""
    try:
        if model_path.endswith(".onnx"):
            import onnxruntime as ort
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            outs = []
            for p in image_paths[:10]:
                x = np.asarray(Image.open(p).convert("RGB").resize((224,224))).astype(np.float32)/255.0
                x = np.transpose(x, (2,0,1))[None,:,:,:]
                outs.append(sess.run(None, {name: x})[0].flatten()[:16])
            arr = np.array(outs)
            return {"access": access, "mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist(),
                    "signature": sha256_bytes(arr.tobytes())}
        elif model_path.endswith((".pt",".torchscript")):
            import torch
            m = torch.jit.load(model_path).eval()
            outs = []
            for p in image_paths[:10]:
                x = torch.tensor(np.asarray(Image.open(p).convert("RGB").resize((224,224))).astype(np.float32)/255.0)
                x = x.permute(2,0,1).unsqueeze(0)
                with torch.no_grad(): outs.append(m(x).flatten()[:16].numpy())
            arr = np.array(outs)
            return {"access": access, "mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist(),
                    "signature": sha256_bytes(arr.tobytes())}
    except Exception as e:
        return {"access": access, "error": str(e), "signature": None}

def compare_fingerprint(a, b):
    if not a.get("signature") or not b.get("signature"):
        return {"match": False, "reason": "signature unavailable"}
    return {"match": a["signature"] == b["signature"],
            "reason": "identical" if a["signature"]==b["signature"] else "behavioural drift detected"}

# ---------- 2.3 PROVENANCE ----------
def gen_keypair():
    sk = Ed25519PrivateKey.generate()
    return sk, sk.public_key()

def sign_inference(image_path, model_path, preprocess, output, seq, prev_hash, sk):
    rec = {
        "seq": seq, "timestamp": time.time(), "nonce": uuid.uuid4().hex,
        "input_hash": sha256_file(image_path),
        "model_digest": sha256_file(model_path),
        "preprocess": preprocess, "output": output, "prev_hash": prev_hash,
    }
    rec["record_hash"] = sha256_bytes(canon(rec))
    rec["signature"] = sk.sign(bytes.fromhex(rec["record_hash"])).hex()
    return rec

def verify_inference(rec, pk):
    body = {k: rec[k] for k in ("seq","timestamp","nonce","input_hash","model_digest","preprocess","output","prev_hash")}
    if sha256_bytes(canon(body)) != rec["record_hash"]:
        return {"ok": False, "reason": "record_hash mismatch (tampered)"}
    try:
        pk.verify(bytes.fromhex(rec["signature"]), bytes.fromhex(rec["record_hash"]))
    except Exception:
        return {"ok": False, "reason": "signature invalid"}
    return {"ok": True, "reason": "verified"}

def verify_chain(records, pk):
    seen_nonces = set()
    for i, r in enumerate(records):
        v = verify_inference(r, pk)
        if not v["ok"]: return {"ok": False, "at": i, "reason": v["reason"]}
        if r["nonce"] in seen_nonces: return {"ok": False, "at": i, "reason": "replay detected (nonce reuse)"}
        seen_nonces.add(r["nonce"])
        if i > 0 and r["prev_hash"] != records[i-1]["record_hash"]:
            return {"ok": False, "at": i, "reason": "chain broken"}
    return {"ok": True, "length": len(records)}

# ---------- 2.4 DRIFT ----------
def drift_score(ref_paths, tgt_paths):
    from scipy.stats import ks_2samp
    def feat(p):
        return np.asarray(Image.open(p).convert("L").resize((32,32))).flatten()/255.0
    R = np.array([feat(p) for p in ref_paths]); T = np.array([feat(p) for p in tgt_paths])
    stats = []
    for i in range(R.shape[1]):
        s,_ = ks_2samp(R[:,i], T[:,i]); stats.append(s)
    score = float(np.mean(stats))
    label = "low" if score<0.15 else "moderate" if score<0.35 else "high"
    interpretation = "operational drift" if score<0.35 else "suspicious shift — manual review"
    return {"score": round(score,3), "level": label, "interpretation": interpretation}

# ---------- 2.5 AUDIT ----------
def audit_append(log_path, event, prev_hash="0"*64):
    event = dict(event); event["prev_hash"] = prev_hash
    event["hash"] = sha256_bytes(canon(event))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f: f.write(json.dumps(event, sort_keys=True)+"\n")
    return event["hash"]

def audit_verify(log_path):
    prev = "0"*64
    with open(log_path) as f:
        for i, line in enumerate(f):
            e = json.loads(line)
            if e["prev_hash"] != prev: return {"ok": False, "at": i, "reason": "chain broken"}
            h = e.pop("hash"); 
            if sha256_bytes(canon(e)) != h: return {"ok": False, "at": i, "reason": "event tampered"}
            prev = h
    return {"ok": True}

def merkle_root(hashes):
    if not hashes: return None
    layer = [bytes.fromhex(h) for h in hashes]
    while len(layer) > 1:
        if len(layer)%2: layer.append(layer[-1])
        layer = [hashlib.sha256(layer[i]+layer[i+1]).digest() for i in range(0,len(layer),2)]
    return layer[0].hex()

# ---------- REPORT ----------
def build_report(asset, findings, coverage, limitations, audit_root):
    return {"report_id": uuid.uuid4().hex, "asset": asset, "findings": findings,
            "coverage": coverage, "limitations": limitations, "audit_root": audit_root,
            "generated_at": time.time()}