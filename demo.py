import os, json, random, numpy as np
from PIL import Image, ImageDraw
from cviaf import *
from cryptography.hazmat.primitives import serialization

random.seed(0); np.random.seed(0)
BASE = "demo_data"
os.makedirs(f"{BASE}/images", exist_ok=True)
os.makedirs(f"{BASE}/models", exist_ok=True)
os.makedirs(f"{BASE}/inferences", exist_ok=True)
os.makedirs(f"{BASE}/keys", exist_ok=True) 
os.makedirs("reports", exist_ok=True)

# --- 1) 200 synthetic images, 3 contributors ---
samples = []
paths = []
for i in range(200):
    c = ["A","B","C"][i % 3]
    lbl = random.choice(["car","person","truck"])
    img = Image.new("RGB",(64,64),(random.randint(0,255),)*3)
    d = ImageDraw.Draw(img); d.rectangle([10,10,40,40], fill=(random.randint(0,255),)*3)
    p = f"{BASE}/images/{c}_{i}.png"; img.save(p)
    samples.append({"id": p, "contributor": c, "label": lbl, "batch": f"{c}_b1"})
    paths.append(p)

# --- 2) INJECT ATTACKS into contributor A ---
# 2a) near-duplicates
for k in range(15):
    src = Image.open(paths[k]); src.save(f"{BASE}/images/A_dup_{k}.png")
    samples.append({"id": f"{BASE}/images/A_dup_{k}.png","contributor":"A","label":"car","batch":"A_b1"})
    paths.append(f"{BASE}/images/A_dup_{k}.png")
# 2b) label flips
for s in samples[:10]:
    if s["contributor"]=="A": s["label"] = "person" if s["label"]=="car" else "car"
# 2c) OOD — random noise images
for k in range(6):
    n = (np.random.rand(64,64,3)*255).astype(np.uint8)
    p = f"{BASE}/images/A_ood_{k}.png"; Image.fromarray(n).save(p)
    samples.append({"id":p,"contributor":"A","label":"truck","batch":"A_b1"}); paths.append(p)
# 2d) trigger patch — bright corner
for k in range(8):
    src = Image.open(paths[k]).convert("RGB"); d = ImageDraw.Draw(src)
    d.rectangle([56,56,63,63], fill=(255,255,0))
    p = f"{BASE}/images/A_trig_{k}.png"; src.save(p)
    samples.append({"id":p,"contributor":"A","label":"car","batch":"A_b1"}); paths.append(p)

json.dump(samples, open(f"{BASE}/samples.json","w"), indent=2)
print(f"[demo] {len(samples)} samples, {len(paths)} images")

# --- 3) Tiny dummy ONNX model (identity-ish) ---
try:
    import onnx
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,3,224,224])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,16])
    w = helper.make_tensor("W", TensorProto.FLOAT, [3*224*224,16], np.random.randn(3*224*224,16).astype(np.float32).flatten())
    node = helper.make_node("MatMul", ["input_flat","W"], ["output"])
    flat = helper.make_node("Flatten", ["input"], ["input_flat"])
    graph = helper.make_graph([flat,node], "tiny", [x], [y], [w])
    m = helper.make_model(graph); m.opset_import[0].version = 13
    onnx.save(m, f"{BASE}/models/tiny.onnx")
    print("[demo] tiny.onnx created")
except Exception as e:
    print("[demo] ONNX create failed:", e)
    open(f"{BASE}/models/tiny.onnx","wb").write(os.urandom(1024))

# --- 4) Signed inference records + tampering ---
sk, pk = gen_keypair()
open(f"{BASE}/keys/priv.pem","wb").write(sk.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption()))
open(f"{BASE}/keys/pub.pem","wb").write(pk.public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo))

records = []; prev = "0"*64
for i in range(10):
    r = sign_inference(paths[i], f"{BASE}/models/tiny.onnx", {"resize":[224,224]}, {"class":"car","conf":0.9}, i, prev, sk)
    records.append(r); prev = r["record_hash"]

# tamper 1: change output
records[3]["output"] = {"class":"person","conf":0.1}
# replay 2: duplicate record
records.append(dict(records[1])); records[-1]["seq"] = 10

json.dump(records, open(f"{BASE}/inferences/records.json","w"), indent=2)
print("[demo] inferences + tampering done")