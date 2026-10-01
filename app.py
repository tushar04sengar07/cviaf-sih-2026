import streamlit as st, json, os
from cviaf import *

st.set_page_config(page_title="CVIAF", layout="wide")
st.title("CVIAF — CV Integrity Assurance Framework")

BASE = "demo_data"
samples = json.load(open(f"{BASE}/samples.json"))
paths = [s["id"] for s in samples]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data","Model","Provenance","Drift","Report"])

with tab1:
    st.subheader("Training-Data Integrity")
    dups = find_duplicates(paths)
    ood  = ood_flags(paths)
    trig = trigger_scan(paths)
    risk = source_risk(samples, dups, ood, trig)
    st.write(f"Near-duplicates: **{len(dups)}** | OOD: **{len(ood)}** | Trigger-like: **{len(trig)}**")
    st.dataframe(risk)
    st.caption("Source-level risk aggregation across contributors.")

with tab2:
    st.subheader("Model Integrity")
    mp = f"{BASE}/models/tiny.onnx"
    d = model_digest(mp)
    fp = behavioural_fingerprint(mp, paths, access="blackbox")
    st.write("Digest:", d[:32]+"...")
    st.json({k:v for k,v in fp.items() if k!="mean" and k!="std"})
    st.caption("Access: black-box. Fingerprint against reference battery.")

with tab3:
    st.subheader("Inference Provenance")
    recs = json.load(open(f"{BASE}/inferences/records.json"))
    sk, pk = gen_keypair()
    # reload pub key
    from cryptography.hazmat.primitives import serialization
    pk = serialization.load_pem_public_key(open(f"{BASE}/keys/pub.pem","rb").read())
    chain = verify_chain(recs, pk)
    st.write("Chain status:", "✅ VERIFIED" if chain["ok"] else f"❌ FAILED at {chain.get('at')}: {chain.get('reason')}")
    for i,r in enumerate(recs):
        v = verify_inference(r, pk)
        st.write(f"#{i} seq={r['seq']} → {'✅' if v['ok'] else '❌ '+v['reason']}")

with tab4:
    st.subheader("Distribution Shift")
    ref = paths[:50]; tgt = paths[50:100]
    st.json(drift_score(ref, tgt))
    st.caption("Low = consistent; high = suspicious shift, manual review.")

with tab5:
    st.subheader("Assurance Report")
    # rebuild everything
    dups = find_duplicates(paths); ood = ood_flags(paths); trig = trigger_scan(paths)
    risk = source_risk(samples, dups, ood, trig)
    findings = []
    for r in risk:
        if r["severity"] != "low":
            findings.append({
                "capability":"training_data_integrity",
                "reason": f"Contributor {r['contributor']} risk {r['risk_score']} ({r['severity']})",
                "evidence": r, "confidence": 0.9, "severity": r["severity"],
                "recommended_disposition": "quarantine" if r["severity"]=="high" else "review",
                "affected_assets": [r["contributor"]]
            })
    # audit
    log = "reports/audit.jsonl"
    if os.path.exists(log): os.remove(log)
    prev = "0"*64
    for ev in [{"event":"data_audit","n":len(samples)}, {"event":"model_audit","digest":model_digest(f"{BASE}/models/tiny.onnx")}, {"event":"inference_audit"}]:
        prev = audit_append(log, ev, prev)
    hashes = [json.loads(l)["hash"] for l in open(log)]
    root = merkle_root(hashes)
    rep = build_report(
        asset={"type":"dataset","id":"demo"},
        findings=findings,
        coverage={"supported":["label_flip","near_dup","ood","simple_patch_backdoor","model_digest","inference_tamper","replay","drift"],
                  "unsupported":["adaptive_attacks","clean_label_poisoning"]},
        limitations=["Black-box trigger search limited","No retraining performed"],
        audit_root=root
    )
    st.json(rep)
    st.download_button("Download report.json", json.dumps(rep, indent=2), file_name="report.json")