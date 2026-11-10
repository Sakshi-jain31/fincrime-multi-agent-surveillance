import os
import re
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, TypedDict, Optional
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

# LangChain / LangGraph / Gemini / Chroma / Langfuse
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler

# Load environment variables
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# ==============================================================================
# LAYER 3 (EPISODIC MEMORY) & SQLITE RELATIONAL STORE
# ==============================================================================
DB_PATH = Path(__file__).resolve().parent / "fincrime.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id TEXT PRIMARY KEY,
            kind TEXT,
            name TEXT,
            country TEXT,
            created_at TEXT,
            risk_base INTEGER,
            kyc_status TEXT,
            pep_score REAL,
            sanctions_score REAL,
            docs TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            customer_id TEXT,
            amount REAL,
            currency TEXT,
            counterparty_country TEXT,
            tx_type TEXT,
            timestamp TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS investigations (
            case_id TEXT PRIMARY KEY,
            customer_id TEXT,
            transaction_id TEXT,
            risk_score REAL,
            decision TEXT,
            sar_narrative TEXT,
            audit_trail TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    
    # Seed baseline data if empty
    cur.execute("SELECT COUNT(*) FROM customers")
    if cur.fetchone()[0] == 0:
        seed_customers = [
            ("CUST-001", "individual", "Viktor Bout", "RU", "2023-01-15", 85, "verified", 0.9, 0.95, "Passport_RU.pdf"),
            ("CUST-002", "corporate", "Acme Global Logistics", "US", "2021-06-10", 15, "verified", 0.05, 0.0, "Cert_Incorp.pdf"),
            ("CUST-003", "individual", "Elena Rostova", "IR", "2024-02-01", 70, "pending", 0.75, 0.2, "ID_Card.pdf")
        ]
        cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", seed_customers)
        
        seed_txs = [
            ("TX-101", "CUST-002", 4500.00, "USD", "US", "wire", "2026-11-01T10:00:00"),
            ("TX-102", "CUST-002", 5200.00, "USD", "DE", "wire", "2026-11-02T14:30:00"),
            ("TX-103", "CUST-002", 4800.00, "USD", "GB", "ach", "2026-11-03T09:15:00"),
            ("TX-104", "CUST-001", 9800.00, "USD", "KP", "wire", "2026-11-05T23:45:00")
        ]
        cur.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?)", seed_txs)
        conn.commit()
    conn.close()

# ==============================================================================
# DETERMINISTIC SCREENING RULES (SANCTIONS, PEP, HIGH-RISK JURISDICTIONS)
# ==============================================================================
HIGH_RISK_COUNTRIES = {"KP", "IR", "SY", "MM", "RU", "BY", "AF", "VE"}
SANCTIONED_ENTITIES = {"viktor bout", "kim jong", "islamic revolutionary guard", "wagner group"}

def run_deterministic_screening(customer: Dict, tx: Dict) -> Dict:
    flags = []
    rule_score = 0.0

    # 1. Sanctions List Check
    cust_name_lower = customer.get("name", "").lower()
    if any(sanctioned in cust_name_lower for sanctioned in SANCTIONED_ENTITIES) or customer.get("sanctions_score", 0) >= 0.8:
        flags.append("OFAC/EU Sanctions Match")
        rule_score += 50.0

    # 2. PEP Check
    if customer.get("pep_score", 0) >= 0.7:
        flags.append("High-Risk Politically Exposed Person (PEP)")
        rule_score += 20.0

    # 3. High-Risk Jurisdiction Check (Customer or Counterparty)
    cust_country = customer.get("country", "").upper()
    cp_country = tx.get("counterparty_country", "").upper()
    if cust_country in HIGH_RISK_COUNTRIES or cp_country in HIGH_RISK_COUNTRIES:
        flags.append(f"FATF High-Risk Jurisdiction ({cust_country} -> {cp_country})")
        rule_score += 25.0

    # 4. Structuring / Threshold Avoidance ($9,000 - $9,999 USD)
    amount = float(tx.get("amount", 0))
    if 9000.0 <= amount < 10000.0:
        flags.append("Potential CTR Structuring ($9k-$10k threshold)")
        rule_score += 15.0

    return {
        "flags": flags,
        "rule_score": min(rule_score, 100.0)
    }

# ==============================================================================
# UNSUPERVISED ML ANOMALY DETECTION (ISOLATION FOREST)
# ==============================================================================
def detect_tx_anomaly(customer_id: str, current_tx: Dict) -> Dict:
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT amount, timestamp FROM transactions WHERE customer_id = ?", conn, params=(customer_id,))
    conn.close()

    # Fallback if insufficient historical baseline
    if len(df) < 3:
        is_anomaly = float(current_tx["amount"]) > 25000.0
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": -0.65 if is_anomaly else 0.15,
            "reason": "Rule-fallback: Amount exceeds baseline threshold (insufficient history)" if is_anomaly else "Normal baseline"
        }

    # Feature engineering: Amount & Hour of Day
    df["hour"] = df["timestamp"].apply(lambda x: parser.parse(x).hour)
    features = df[["amount", "hour"]].values

    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    iso_forest.fit(features)

    curr_hour = parser.parse(current_tx["timestamp"]).hour
    curr_feat = np.array([[float(current_tx["amount"]), curr_hour]])
    
    pred = iso_forest.predict(curr_feat)[0]  # -1 for anomaly, 1 for normal
    score = float(iso_forest.decision_function(curr_feat)[0])

    return {
        "is_anomaly": bool(pred == -1),
        "anomaly_score": round(score, 4),
        "reason": "Statistical outlier in amount/time distribution" if pred == -1 else "Within historical behavior profile"
    }

# ==============================================================================
# LAYER 2 (SEMANTIC MEMORY) - CHROMA RAG FOR COMPLIANCE POLICIES
# ==============================================================================
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_compliance_db"

def init_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=API_KEY
    )
    vectorstore = Chroma(
        collection_name="aml_policies",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Seed compliance policies if collection is empty
    if len(vectorstore.get()["ids"]) == 0:
        policies = [
            Document(
                page_content="POLICY AML-01 (Sanctions): Any positive or high-confidence match against OFAC SDN or EU Consolidated Sanctions lists requires immediate transaction blocking, account freeze, and mandatory SAR/STR filing within 24 hours.",
                metadata={"policy_id": "AML-01", "category": "Sanctions"}
            ),
            Document(
                page_content="POLICY AML-02 (Structuring): Transactions between $9,000 and $9,999 designed to evade Currency Transaction Report (CTR) $10,000 thresholds must be flagged for Enhanced Due Diligence (EDD) and aggregated over a 14-day rolling window.",
                metadata={"policy_id": "AML-02", "category": "Structuring"}
            ),
            Document(
                page_content="POLICY AML-03 (PEP & High-Risk Jurisdictions): Politically Exposed Persons (PEPs) transacting with FATF grey/black-listed jurisdictions require senior compliance officer sign-off and source-of-wealth verification.",
                metadata={"policy_id": "AML-03", "category": "PEP/Jurisdiction"}
            ),
            Document(
                page_content="POLICY AML-04 (Behavioral Anomalies): Unexplained spikes in transaction volume (>3x historical average) or off-hours wire transfers require automated RFI (Request for Information) unless accompanied by verified trade invoices.",
                metadata={"policy_id": "AML-04", "category": "Anomaly"}
            )
        ]
        vectorstore.add_documents(policies)
    return vectorstore

# ==============================================================================
# LANGGRAPH MULTI-AGENT COMPLIANCE WORKFLOW
# ==============================================================================
class ComplianceState(TypedDict):
    case_id: str
    customer: Dict
    transaction: Dict
    screening_results: Dict
    anomaly_results: Dict
    retrieved_policies: List[str]
    composite_risk_score: float
    decision: str
    sar_narrative: str
    audit_trail: List[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=API_KEY)

def node_screening_agent(state: ComplianceState) -> ComplianceState:
    res = run_deterministic_screening(state["customer"], state["transaction"])
    audit = state.get("audit_trail", [])
    audit.append(f"[ScreeningAgent] Completed rule checks. Score: {res['rule_score']}, Flags: {res['flags']}")
    return {**state, "screening_results": res, "audit_trail": audit}

def node_anomaly_agent(state: ComplianceState) -> ComplianceState:
    res = detect_tx_anomaly(state["customer"]["id"], state["transaction"])
    audit = state["audit_trail"]
    audit.append(f"[AnomalyAgent] IsolationForest status: {res['is_anomaly']} (score: {res['anomaly_score']})")
    return {**state, "anomaly_results": res, "audit_trail": audit}

def node_policy_rag_agent(state: ComplianceState) -> ComplianceState:
    flags = state["screening_results"]["flags"]
    anomaly_reason = state["anomaly_results"]["reason"]
    query = f"Compliance rules for flags: {', '.join(flags)}. Anomaly context: {anomaly_reason}."
    
    docs = vector_store.similarity_search(query, k=2)
    policies = [f"[{d.metadata.get('policy_id')}] {d.page_content}" for d in docs]
    
    audit = state["audit_trail"]
    audit.append(f"[PolicyRAGAgent] Retrieved {len(policies)} relevant regulatory policies.")
    return {**state, "retrieved_policies": policies, "audit_trail": audit}

def node_adjudicator_agent(state: ComplianceState) -> ComplianceState:
    rule_score = state["screening_results"]["rule_score"]
    anomaly_bonus = 25.0 if state["anomaly_results"]["is_anomaly"] else 0.0
    base_risk = float(state["customer"].get("risk_base", 0)) * 0.2
    
    composite_score = min(round(rule_score + anomaly_bonus + base_risk, 2), 100.0)
    
    if composite_score >= 75.0 or "OFAC/EU Sanctions Match" in state["screening_results"]["flags"]:
        decision = "BLOCK_AND_FILE_SAR"
    elif composite_score >= 40.0:
        decision = "HOLD_FOR_EDD"
    else:
        decision = "APPROVE"

    prompt = f"""
    You are a Senior Financial Crime Compliance Officer. Generate a concise, regulatory-grade 
    Suspicious Activity Report (SAR) / Investigation Summary based on the following evidence:
    
    Customer: {json.dumps(state['customer'])}
    Transaction: {json.dumps(state['transaction'])}
    Deterministic Flags: {state['screening_results']['flags']}
    ML Anomaly Detection: {json.dumps(state['anomaly_results'])}
    Applicable Policies: {json.dumps(state['retrieved_policies'])}
    Composite Risk Score: {composite_score}/100
    Final Decision: {decision}
    
    Structure the output with: 1) Executive Summary, 2) Key Risk Indicators, 3) Policy Mapping, 4) Recommended Action.
    """
    
    response = llm.invoke(prompt)
    narrative = response.content

    audit = state["audit_trail"]
    audit.append(f"[AdjudicatorAgent] Final Decision: {decision} (Composite Risk: {composite_score})")
    
    return {
        **state,
        "composite_risk_score": composite_score,
        "decision": decision,
        "sar_narrative": narrative,
        "audit_trail": audit
    }

def build_compliance_graph():
    workflow = StateGraph(ComplianceState)
    workflow.add_node("screening", node_screening_agent)
    workflow.add_node("anomaly", node_anomaly_agent)
    workflow.add_node("policy_rag", node_policy_rag_agent)
    workflow.add_node("adjudicator", node_adjudicator_agent)

    workflow.set_entry_point("screening")
    workflow.add_edge("screening", "anomaly")
    workflow.add_edge("anomaly", "policy_rag")
    workflow.add_edge("policy_rag", "adjudicator")
    workflow.add_edge("adjudicator", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Initialize DB, Vector Store, and Compiled Graph on startup
init_db()
vector_store = init_vector_store()
compliance_graph = build_compliance_graph()

# ==============================================================================
# FLASK API ENDPOINTS & SHARED INVESTIGATION PIPELINE
# ==============================================================================
def execute_investigation_pipeline(customer_id: str, tx_data: Dict) -> Tuple[Dict, int]:
    conn = get_db_connection()
    cust_row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    if not cust_row:
        conn.close()
        return {"error": f"Customer {customer_id} not found"}, 404

    customer = dict(cust_row)
    unique_suffix = int(datetime.utcnow().timestamp() * 1000)
    tx_id = tx_data.get("id", f"TX-{unique_suffix}")
    tx_data["id"] = tx_id
    tx_data["customer_id"] = customer_id
    tx_data.setdefault("timestamp", datetime.utcnow().isoformat())

    # Persist incoming transaction
    conn.execute(
        """INSERT OR REPLACE INTO transactions 
           (id, customer_id, amount, currency, counterparty_country, tx_type, timestamp) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            tx_id,
            customer_id,
            float(tx_data["amount"]),
            tx_data.get("currency", "USD"),
            tx_data.get("counterparty_country", "US"),
            tx_data.get("tx_type", "wire"),
            tx_data["timestamp"]
        )
    )
    conn.commit()

    # Prepare LangGraph execution
    case_id = f"CASE-{int(datetime.utcnow().timestamp() * 1000)}"
    initial_state: ComplianceState = {
        "case_id": case_id,
        "customer": customer,
        "transaction": tx_data,
        "screening_results": {},
        "anomaly_results": {},
        "retrieved_policies": [],
        "composite_risk_score": 0.0,
        "decision": "",
        "sar_narrative": "",
        "audit_trail": []
    }

    callbacks = []
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        callbacks.append(CallbackHandler())

    config = {
        "configurable": {"thread_id": case_id},
        "callbacks": callbacks
    }

    try:
        final_state = compliance_graph.invoke(initial_state, config=config)
    except Exception as e:
        conn.close()
        return {"error": f"Agent workflow execution failed: {str(e)}"}, 500

    # Save investigation outcome to SQLite
    conn.execute(
        "INSERT INTO investigations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            case_id,
            customer_id,
            tx_id,
            final_state["composite_risk_score"],
            final_state["decision"],
            final_state["sar_narrative"],
            json.dumps(final_state["audit_trail"]),
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()

    return {
        "case_id": case_id,
        "customer_id": customer_id,
        "transaction_id": tx_id,
        "composite_risk_score": final_state["composite_risk_score"],
        "decision": final_state["decision"],
        "screening_flags": final_state["screening_results"]["flags"],
        "anomaly_detection": final_state["anomaly_results"],
        "matched_policies": final_state["retrieved_policies"],
        "sar_narrative": final_state["sar_narrative"],
        "audit_trail": final_state["audit_trail"]
    }, 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()}), 200


@app.route("/api/v1/customers", methods=["GET"])
def list_customers():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM customers").fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows]), 200


@app.route("/api/v1/investigate", methods=["POST"])
def investigate_transaction():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    customer_id = payload.get("customer_id")
    tx_data = payload.get("transaction")
    if not customer_id or not tx_data:
        return jsonify({"error": "Both 'customer_id' and 'transaction' objects are required"}), 400

    result, status = execute_investigation_pipeline(customer_id, tx_data)
    return jsonify(result), status


@app.route("/api/v1/investigations", methods=["GET"])
def list_investigations():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM investigations ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows]), 200


@app.route("/api/v1/investigations/<case_id>", methods=["GET"])
def get_investigation(case_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM investigations WHERE case_id = ?", (case_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Investigation case not found"}), 404

    result = dict(row)
    result["audit_trail"] = json.loads(result["audit_trail"])
    return jsonify(result), 200


@app.route("/api/v1/seed-demo", methods=["POST"])
def seed_demo_c101():
    conn = get_db_connection()
    conn.execute(
        "INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("C101", "corporate", "Horizon Shell Trading Ltd", "IR", "2025-10-12", 88, "flagged", 0.85, 0.92, "Registry_IR.pdf")
    )
    conn.commit()
    conn.close()

    case_data, status = execute_investigation_pipeline("C101", {
        "id": "TX-C101-DEMO",
        "amount": 9850.00,
        "currency": "USD",
        "counterparty_country": "KP",
        "tx_type": "wire"
    })
    if status != 200:
        return jsonify(case_data), status

    return jsonify({
        "message": "Seeded customer C101 and executed SAR investigation.",
        "case": case_data
    }), 200


@app.route("/api/v1/run-detectors", methods=["POST"])
def batch_run_detectors():
    conn = get_db_connection()
    uninvestigated = conn.execute("""
        SELECT t.* FROM transactions t
        LEFT JOIN investigations i ON t.id = i.transaction_id
        WHERE i.case_id IS NULL
    """).fetchall()
    conn.close()

    processed = 0
    for row in uninvestigated:
        tx = dict(row)
        _, status = execute_investigation_pipeline(tx["customer_id"], tx)
        if status == 200:
            processed += 1

    return jsonify({"processed": processed}), 200


@app.route("/api/v1/nlp-sql", methods=["POST"])
def nlp_to_sql():
    user_query = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    sql_prompt = f"""
    Convert the user's compliance question into a single, read-only SQLite SELECT query.
    Tables:
    - customers (id, kind, name, country, created_at, risk_base, kyc_status, pep_score, sanctions_score, docs)
    - transactions (id, customer_id, amount, currency, counterparty_country, tx_type, timestamp)
    - investigations (case_id, customer_id, transaction_id, risk_score, decision, created_at)
    
    Rules: Return ONLY raw SQL without markdown fences. Only SELECT statements are permitted.
    Question: {user_query}
    """
    raw_sql = llm.invoke(sql_prompt).content.strip()
    clean_sql = re.sub(r"^```sql|^```|```$", "", raw_sql, flags=re.MULTILINE).strip()

    if not clean_sql.upper().startswith("SELECT") or any(
        kw in clean_sql.upper() for kw in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    ):
        return jsonify({"error": "Blocked unsafe query", "sql": clean_sql}), 400

    try:
        conn = get_db_connection()
        rows = conn.execute(clean_sql).fetchall()
        conn.close()
        return jsonify({"sql": clean_sql, "rows": [dict(r) for r in rows]}), 200
    except Exception as e:
        return jsonify({"error": str(e), "sql": clean_sql}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
