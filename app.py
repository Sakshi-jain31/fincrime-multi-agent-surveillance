import os
import re
import io
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, TypedDict, Optional

import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, Response

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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS unified_runs (
            run_id TEXT PRIMARY KEY,
            filename TEXT,
            rows_processed INTEGER,
            winning_strategy TEXT,
            wf_sharpe REAL,
            wf_return_pct REAL,
            framl_alerts_count INTEGER,
            executive_summary TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS framl_ledger (
            row_id TEXT PRIMARY KEY,
            run_id TEXT,
            timestamp TEXT,
            symbol TEXT,
            customer_id TEXT,
            order_amount REAL,
            counterparty_country TEXT,
            tx_count_1h INTEGER,
            amount_ratio_avg REAL,
            fraud_prob REAL,
            aml_score REAL,
            unified_risk_score REAL,
            decision TEXT,
            flags TEXT
        )
    """)
    conn.commit()
    conn.close()

# ==============================================================================
# DETERMINISTIC SCREENING & ML DETECTORS (AML + LAB 2 FRAUD + LAB 3 QUANT)
# ==============================================================================
HIGH_RISK_COUNTRIES = {"KP", "IR", "SY", "MM", "RU", "BY", "AF", "VE"}
SANCTIONED_ENTITIES = {"viktor bout", "kim jong", "islamic revolutionary guard", "wagner group", "horizon shell"}

def run_deterministic_screening(customer: Dict, tx: Dict) -> Dict:
    flags = []
    rule_score = 0.0

    cust_name_lower = customer.get("name", "").lower()
    if any(s in cust_name_lower for s in SANCTIONED_ENTITIES) or float(customer.get("sanctions_score", 0)) >= 0.8:
        flags.append("OFAC/EU Sanctions Match")
        rule_score += 50.0

    if float(customer.get("pep_score", 0)) >= 0.7:
        flags.append("High-Risk PEP")
        rule_score += 20.0

    cust_country = str(customer.get("country", "")).upper()
    cp_country = str(tx.get("counterparty_country", "")).upper()
    if cust_country in HIGH_RISK_COUNTRIES or cp_country in HIGH_RISK_COUNTRIES:
        flags.append(f"FATF Jurisdiction ({cust_country}->{cp_country})")
        rule_score += 25.0

    amount = float(tx.get("amount", 0))
    if 9000.0 <= amount < 10000.0:
        flags.append("CTR Structuring ($9k-$10k)")
        rule_score += 15.0

    return {"flags": flags, "rule_score": min(rule_score, 100.0)}

def detect_tx_anomaly(customer_id: str, current_tx: Dict) -> Dict:
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT amount, timestamp FROM transactions WHERE customer_id = ?", conn, params=(customer_id,))
    conn.close()

    if len(df) < 3:
        is_anomaly = float(current_tx["amount"]) > 25000.0
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": -0.65 if is_anomaly else 0.15,
            "reason": "Rule-fallback: Amount > $25k" if is_anomaly else "Normal baseline"
        }

    df["hour"] = df["timestamp"].apply(lambda x: parser.parse(str(x)).hour)
    features = df[["amount", "hour"]].values

    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    iso_forest.fit(features)

    curr_hour = parser.parse(str(current_tx["timestamp"])).hour
    curr_feat = np.array([[float(current_tx["amount"]), curr_hour]])
    pred = iso_forest.predict(curr_feat)
    score = float(iso_forest.decision_function(curr_feat))

    return {
        "is_anomaly": bool(pred == -1),
        "anomaly_score": round(score, 4),
        "reason": "Statistical outlier in amount/time" if pred == -1 else "Within historical profile"
    }

# --- LAB 2: LOGISTIC REGRESSION FRAUD & VELOCITY SCORER ---
def build_fraud_classifier() -> LogisticRegression:
    # Deterministic synthetic calibration dataset: [amount_ratio_avg, tx_count_1h, is_night, ip_country_mismatch]
    rng = np.random.default_rng(42)
    X_legit = np.column_stack([
        rng.normal(1.0, 0.35, 200).clip(0.1, 3.0),
        rng.poisson(1.2, 200).clip(1, 4),
        rng.binomial(1, 0.15, 200),
        rng.binomial(1, 0.05, 200)
    ])
    X_fraud = np.column_stack([
        rng.normal(5.5, 1.5, 80).clip(3.2, 12.0),
        rng.poisson(5.5, 80).clip(3, 15),
        rng.binomial(1, 0.70, 80),
        rng.binomial(1, 0.80, 80)
    ])
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * 200 + [1] * 80)
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    return clf

FRAUD_CLF = build_fraud_classifier()

# --- LAB 3: QUANT STRATEGY SWEEP & 70/30 WALK-FORWARD ENGINE ---
def run_lab3_quant_engine(df: pd.DataFrame) -> Dict:
    required_ohlcv = {"open", "high", "low", "close"}
    if not required_ohlcv.issubset(set(df.columns)) or len(df) < 15:
        return {
            "executed": False,
            "reason": "Insufficient OHLCV rows (<15 bars) for walk-forward backtest.",
            "winning_strategy": "N/A",
            "wf_sharpe": 0.0,
            "wf_return_pct": 0.0,
            "sweep_results": []
        }

    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    high = pd.to_numeric(df["high"], errors="coerce").ffill().bfill()
    low = pd.to_numeric(df["low"], errors="coerce").ffill().bfill()
    returns = close.pct_change().fillna(0.0)

    # 70/30 Walk-Forward Split
    split_idx = int(len(df) * 0.70)
    in_ret = returns.iloc[:split_idx]
    out_ret = returns.iloc[split_idx:]

    def eval_signal(sig: pd.Series, ret_series: pd.Series) -> Tuple[float, float]:
        strat_ret = sig.shift(1).fillna(0.0) * ret_series
        std = strat_ret.std()
        sharpe = float((strat_ret.mean() / std) * np.sqrt(252)) if std > 1e-9 else 0.0
        cum_ret = float(((1 + strat_ret).prod() - 1) * 100.0)
        return round(sharpe, 3), round(cum_ret, 2)

    # Candidate 1: SMA Crossover (5 vs 15)
    sma_fast = close.rolling(5, min_periods=1).mean()
    sma_slow = close.rolling(15, min_periods=1).mean()
    sig_sma = (sma_fast > sma_slow).astype(float)

       # Candidate 2: RSI Mean-Reversion (14-bar)
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    sig_rsi = np.where(rsi < 35, 1.0, np.where(rsi > 70, 0.0, np.nan))
    sig_rsi = pd.Series(sig_rsi, index=close.index).ffill().fillna(0.0)

    # Candidate 3: Breakout + ATR Volatility Filter
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(10, min_periods=1).mean()
    upper_band = high.rolling(10, min_periods=1).max().shift(1).bfill()
    sig_brk = ((close > upper_band) | (atr > atr.median())).astype(float)

    candidates = [
        ("SMA Crossover (5/15)", sig_sma),
        ("RSI Mean-Reversion (14)", sig_rsi),
        ("Breakout + ATR Stop", sig_brk)
    ]

    sweep_results = []
    best_name, best_sig, best_is_sharpe = "", None, -999.0

    for name, sig in candidates:
        is_sharpe, is_ret = eval_signal(sig.iloc[:split_idx], in_ret)
        oos_sharpe, oos_ret = eval_signal(sig.iloc[split_idx:], out_ret)
        sweep_results.append({
            "strategy": name,
            "in_sample_sharpe": is_sharpe,
            "in_sample_return_pct": is_ret,
            "walk_forward_sharpe": oos_sharpe,
            "walk_forward_return_pct": oos_ret
        })
        if is_sharpe > best_is_sharpe:
            best_is_sharpe = is_sharpe
            best_name = name
            best_sig = sig

    wf_sharpe, wf_ret = eval_signal(best_sig.iloc[split_idx:], out_ret)
    return {
        "executed": True,
        "bars_analyzed": len(df),
        "split": f"{split_idx} IS / {len(df) - split_idx} OOS",
        "winning_strategy": best_name,
        "wf_sharpe": wf_sharpe,
        "wf_return_pct": wf_ret,
        "sweep_results": sweep_results
    }

# ==============================================================================
# LAYER 2 (SEMANTIC MEMORY) - CHROMA RAG FOR COMPLIANCE POLICIES
# ==============================================================================
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_compliance_db"

def init_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=API_KEY)
    vectorstore = Chroma(
        collection_name="aml_policies",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    if len(vectorstore.get()["ids"]) == 0:
        policies = [
            Document(
                page_content="POLICY AML-01 (Sanctions): Positive/high-confidence match against OFAC SDN or EU Consolidated Sanctions lists requires immediate transaction blocking, account freeze, and mandatory SAR filing within 24 hours.",
                metadata={"policy_id": "AML-01", "category": "Sanctions"}
            ),
            Document(
                page_content="POLICY AML-02 (Structuring): Transactions between $9,000 and $9,999 designed to evade CTR $10,000 thresholds must be flagged for Enhanced Due Diligence (EDD).",
                metadata={"policy_id": "AML-02", "category": "Structuring"}
            ),
            Document(
                page_content="POLICY FRAML-03 (Account Takeover & Velocity): Orders >=4x historical 30-day average combined with IP country mismatch or high 1h velocity require step-up authentication and human /review.",
                metadata={"policy_id": "FRAML-03", "category": "Fraud"}
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
    audit.append(f"[ScreeningAgent] Score: {res['rule_score']}, Flags: {res['flags']}")
    return {**state, "screening_results": res, "audit_trail": audit}

def node_anomaly_agent(state: ComplianceState) -> ComplianceState:
    res = detect_tx_anomaly(state["customer"]["id"], state["transaction"])
    audit = state["audit_trail"]
    audit.append(f"[AnomalyAgent] IsolationForest outlier: {res['is_anomaly']} ({res['anomaly_score']})")
    return {**state, "anomaly_results": res, "audit_trail": audit}

def node_policy_rag_agent(state: ComplianceState) -> ComplianceState:
    flags = state["screening_results"]["flags"]
    query = f"Compliance rules for flags: {', '.join(flags)}. {state['anomaly_results']['reason']}"
    docs = vector_store.similarity_search(query, k=2)
    policies = [f"[{d.metadata.get('policy_id')}] {d.page_content}" for d in docs]
    audit = state["audit_trail"]
    audit.append(f"[PolicyRAGAgent] Retrieved {len(policies)} policies.")
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
    Generate a concise, regulatory-grade Suspicious Activity Report (SAR) summary using ONLY the exact metrics below:
    Customer: {json.dumps(state['customer'])}
    Transaction: {json.dumps(state['transaction'])}
    Flags: {state['screening_results']['flags']}
    Anomaly: {json.dumps(state['anomaly_results'])}
    Policies: {json.dumps(state['retrieved_policies'])}
    Composite Risk Score: {composite_score}/100 | Decision: {decision}
    Sections: 1) Executive Summary, 2) Risk Indicators, 3) Policy Mapping, 4) Action.
    """
    narrative = llm.invoke(prompt).content
    audit = state["audit_trail"]
    audit.append(f"[AdjudicatorAgent] Decision: {decision} (Risk: {composite_score})")
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
    return workflow.compile(checkpointer=MemorySaver())

init_db()
vector_store = init_vector_store()
compliance_graph = build_compliance_graph()

# ==============================================================================
# SHARED SINGLE-TRANSACTION PIPELINE
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

    callbacks = [CallbackHandler()] if (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")) else []
    try:
        final_state = compliance_graph.invoke(initial_state, config={"configurable": {"thread_id": case_id}, "callbacks": callbacks})
    except Exception as e:
        conn.close()
        return {"error": f"Agent workflow failed: {str(e)}"}, 500

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

# ==============================================================================
# FLASK ROUTES: PORTAL, UNIFIED CSV/XLSX UPLOADER, FRAML & QUANT ENGINES
# ==============================================================================
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
        return jsonify({"error": "Case not found"}), 404
    res = dict(row)
    res["audit_trail"] = json.loads(res["audit_trail"])
    return jsonify(res), 200

@app.route("/api/v1/investigate", methods=["POST"])
def investigate_transaction():
    payload = request.get_json(silent=True) or {}
    customer_id = payload.get("customer_id")
    tx_data = payload.get("transaction")
    if not customer_id or not tx_data:
        return jsonify({"error": "Both 'customer_id' and 'transaction' are required"}), 400
    result, status = execute_investigation_pipeline(customer_id, tx_data)
    return jsonify(result), status

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
        "id": f"TX-C101-{int(datetime.utcnow().timestamp())}",
        "amount": 9850.00,
        "currency": "USD",
        "counterparty_country": "KP",
        "tx_type": "wire"
    })
    if status != 200:
        return jsonify(case_data), status
    return jsonify({"message": "Seeded customer C101 and executed SAR investigation.", "case": case_data}), 200

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

# --- SAMPLE UNIFIED CSV TEMPLATE DOWNLOAD ---
@app.route("/api/v1/sample-unified-csv", methods=["GET"])
def download_sample_unified_csv():
    rng = np.random.default_rng(101)
    rows = []
    base_price = 185.0
    accounts = ["CUST-002", "CUST-002", "CUST-003", "C101", "CUST-001"]
    countries = ["US", "DE", "GB", "IR", "KP"]

    for i in range(30):
        ts = f"2026-11-{(i // 3) + 1:02d}T{(9 + (i * 3)) % 24:02d}:15:00"
        change = rng.normal(0.4, 2.2)
        open_p = round(base_price, 2)
        close_p = round(max(120.0, open_p + change), 2)
        high_p = round(max(open_p, close_p) + abs(rng.normal(1.0, 0.8)), 2)
        low_p = round(min(open_p, close_p) - abs(rng.normal(1.0, 0.8)), 2)
        vol = int(rng.integers(12000, 85000))
        base_price = close_p

        # Inject specific FRAML anomalies on rows 12, 22, 27
        if i == 12:
            acct, amount, cp_country, ip_mis = "C101", 9850.00, "KP", 1
        elif i == 22:
            acct, amount, cp_country, ip_mis = "CUST-001", 48500.00, "RU", 1
        elif i == 27:
            acct, amount, cp_country, ip_mis = "CUST-003", 9450.00, "IR", 0
        else:
            idx = i % len(accounts)
            acct = accounts[idx]
            amount = round(float(rng.uniform(2200, 6500)), 2)
            cp_country = countries[idx]
            ip_mis = 0

        hour_val = int(ts.split("T")[1].split(":")[0])
        rows.append({
            "timestamp": ts,
            "symbol": "NVDA",
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": vol,
            "account_id": acct,
            "order_amount": amount,
            "hour_of_day": hour_val,
            "ip_country_mismatch": ip_mis,
            "counterparty_country": cp_country
        })

    df_sample = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    df_sample.to_csv(csv_buffer, index=False)
    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=unified_trade_aml_ledger.csv"}
    )

# --- UNIFIED CSV/EXCEL INGESTION & 3-ENGINE PIPELINE ---
@app.route("/api/v1/upload-unified", methods=["POST"])
def upload_unified_ledger():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported format. Upload .csv or .xlsx"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = df["date"]
    if "customer_id" in df.columns and "account_id" not in df.columns:
        df["account_id"] = df["customer_id"]
    if "amount" in df.columns and "order_amount" not in df.columns:
        df["order_amount"] = df["amount"]

    # Fill missing telemetry columns gracefully so OHLCV-only or TX-only files both work
    if "timestamp" not in df.columns:
        df["timestamp"] = [datetime.utcnow().isoformat() for _ in range(len(df))]
    if "symbol" not in df.columns:
        df["symbol"] = "SYNTH-EQ"
    if "account_id" not in df.columns:
        sim_accts = ["CUST-002", "CUST-003", "C101", "CUST-001"]
        df["account_id"] = [sim_accts[i % len(sim_accts)] for i in range(len(df))]
    if "order_amount" not in df.columns:
        if "close" in df.columns and "volume" in df.columns:
            df["order_amount"] = (pd.to_numeric(df["close"], errors="coerce").fillna(100) * 50).round(2)
        else:
            df["order_amount"] = 5000.0
    if "counterparty_country" not in df.columns:
        df["counterparty_country"] = "US"
    if "ip_country_mismatch" not in df.columns:
        df["ip_country_mismatch"] = 0

    df["parsed_ts"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp.utcnow())
    df = df.sort_values("parsed_ts").reset_index(drop=True)
    df["hour_of_day"] = df["parsed_ts"].dt.hour
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce").fillna(0.0)

    # 1. Rolling Velocity & Spend Baseline Features
    df["acct_avg_amt"] = df.groupby("account_id")["order_amount"].transform("mean").replace(0, 1.0)
    df["amount_ratio_avg"] = (df["order_amount"] / df["acct_avg_amt"]).round(2)

    tx_counts_1h = []
    for idx, row in df.iterrows():
        window_start = row["parsed_ts"] - pd.Timedelta(hours=1)
        cnt = len(df[(df["account_id"] == row["account_id"]) & (df["parsed_ts"] >= window_start) & (df["parsed_ts"] <= row["parsed_ts"])])
        tx_counts_1h.append(cnt)
    df["tx_count_1h"] = tx_counts_1h

    # 2. Stage 0: Quant Strategy Backtest (Lab 3)
    quant_report = run_lab3_quant_engine(df)

    # 3. Stage 1 & 2: FRAML Surveillance (Lab 2 Fraud + FinCrime AML)
    conn = get_db_connection()
    cust_rows = conn.execute("SELECT * FROM customers").fetchall()
    cust_map = {r["id"]: dict(r) for r in cust_rows}

    iso_preds = np.ones(len(df))
    if len(df) >= 5:
        iso = IsolationForest(contamination=0.15, random_state=42)
        iso_preds = iso.fit_predict(df[["order_amount", "hour_of_day"]].values)

    run_id = f"RUN-{int(datetime.utcnow().timestamp() * 1000)}"
    ledger_records = []
    high_risk_alerts = []

    for idx, row in df.iterrows():
        acct_id = str(row["account_id"]).strip()
        cust = cust_map.get(acct_id, {
            "id": acct_id, "name": f"Entity {acct_id}", "country": "US",
            "risk_base": 20, "pep_score": 0.1, "sanctions_score": 0.0
        })

        if acct_id not in cust_map:
            conn.execute(
                "INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (acct_id, "corporate", cust["name"], "US", datetime.utcnow().isoformat(), 20, "unverified", 0.1, 0.0, "Uploaded_Ledger.csv")
            )
            cust_map[acct_id] = cust

        tx_obj = {
            "amount": float(row["order_amount"]),
            "counterparty_country": str(row["counterparty_country"]).upper(),
            "timestamp": row["parsed_ts"].isoformat()
        }
        aml_res = run_deterministic_screening(cust, tx_obj)
        aml_score = min(aml_res["rule_score"] + (20.0 if iso_preds[idx] == -1 else 0.0), 100.0)

        # Lab 2 Logistic Regression Fraud Probability
        is_night = 1 if (row["hour_of_day"] < 6 or row["hour_of_day"] >= 23) else 0
        ip_mis = int(row.get("ip_country_mismatch", 0))
        fraud_features = np.array([[float(row["amount_ratio_avg"]), int(row["tx_count_1h"]), is_night, ip_mis]])
        fraud_prob = float(FRAUD_CLF.predict_proba(fraud_features)[0][1])

        unified_score = round(min(100.0, max(aml_score, fraud_prob * 100.0) * 0.75 + (float(cust.get("risk_base", 0)) * 0.25)), 2)

        flags = list(aml_res["flags"])
        if iso_preds[idx] == -1:
            flags.append("IsolationForest Outlier")
        if fraud_prob >= 0.65:
            flags.append(f"Fraud Prob {round(fraud_prob * 100)}%")
        if row["amount_ratio_avg"] >= 3.5:
            flags.append(f"Spend Spike {row['amount_ratio_avg']}x")
        if ip_mis == 1:
            flags.append("IP Country Mismatch")

        if unified_score >= 75.0 or "OFAC/EU Sanctions Match" in flags:
            decision = "BLOCK_AND_FILE_SAR"
        elif unified_score >= 45.0:
            decision = "QUEUE_HUMAN_REVIEW"
        else:
            decision = "APPROVE"

        row_id = f"{run_id}-R{idx + 1}"
        rec = {
            "row_id": row_id,
            "run_id": run_id,
            "timestamp": row["parsed_ts"].strftime("%Y-%m-%d %H:%M"),
            "symbol": str(row["symbol"]),
            "customer_id": acct_id,
            "order_amount": round(float(row["order_amount"]), 2),
            "counterparty_country": str(row["counterparty_country"]).upper(),
            "tx_count_1h": int(row["tx_count_1h"]),
            "amount_ratio_avg": float(row["amount_ratio_avg"]),
            "fraud_prob": round(fraud_prob, 3),
            "aml_score": round(aml_score, 1),
            "unified_risk_score": unified_score,
            "decision": decision,
            "flags": flags
        }
        ledger_records.append(rec)

        conn.execute(
            "INSERT INTO framl_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec["row_id"], run_id, rec["timestamp"], rec["symbol"], rec["customer_id"],
                rec["order_amount"], rec["counterparty_country"], rec["tx_count_1h"],
                rec["amount_ratio_avg"], rec["fraud_prob"], rec["aml_score"],
                rec["unified_risk_score"], rec["decision"], ", ".join(flags)
            )
        )

        if decision != "APPROVE":
            high_risk_alerts.append(rec)

    # 4. Stage 3: Gemini 2.5 Flash Grounded Synthesis (Layer 6 Numeric Faithfulness)
    synthesis_prompt = f"""
    You are a Principal Quantitative & Financial Crime Risk Officer.
    Summarize the unified batch execution below using ONLY the exact numbers provided. Do NOT invent metrics.

    File: {file.filename} | Total Rows: {len(df)}
    Lab 3 Quant Backtest: {json.dumps(quant_report)}
    Total FRAML Alerts Flagged: {len(high_risk_alerts)}
    Top High-Risk Alerts: {json.dumps(high_risk_alerts[:5])}

    Provide a crisp 3-paragraph audit summary covering:
    1) Quant Strategy Walk-Forward Performance,
    2) FRAML Fraud & AML Surveillance Findings, and
    3) Immediate Compliance & Risk Controls.
    """
    try:
        exec_summary = llm.invoke(synthesis_prompt).content
    except Exception:
        exec_summary = f"Batch processed {len(df)} rows. Winning strategy: {quant_report.get('winning_strategy')} (WF Sharpe: {quant_report.get('wf_sharpe')}). High-risk FRAML alerts: {len(high_risk_alerts)}."

    conn.execute(
        "INSERT INTO unified_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, file.filename, len(df),
            quant_report.get("winning_strategy", "N/A"),
            quant_report.get("wf_sharpe", 0.0),
            quant_report.get("wf_return_pct", 0.0),
            len(high_risk_alerts),
            exec_summary,
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()

    return jsonify({
        "run_id": run_id,
        "filename": file.filename,
        "rows_processed": len(df),
        "quant_report": quant_report,
        "framl_alerts_count": len(high_risk_alerts),
        "ledger_rows": ledger_records,
        "executive_summary": exec_summary
    }), 200

@app.route("/api/v1/nlp-sql", methods=["POST"])
def nlp_to_sql():
    user_query = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    sql_prompt = f"""
    Convert the user's compliance or quant question into a single, read-only SQLite SELECT query.
    Tables:
    - customers (id, kind, name, country, created_at, risk_base, kyc_status, pep_score, sanctions_score, docs)
    - transactions (id, customer_id, amount, currency, counterparty_country, tx_type, timestamp)
    - investigations (case_id, customer_id, transaction_id, risk_score, decision, created_at)
    - unified_runs (run_id, filename, rows_processed, winning_strategy, wf_sharpe, wf_return_pct, framl_alerts_count, created_at)
    - framl_ledger (row_id, run_id, timestamp, symbol, customer_id, order_amount, counterparty_country, tx_count_1h, amount_ratio_avg, fraud_prob, aml_score, unified_risk_score, decision, flags)
    
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

