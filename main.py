import os, hmac, hashlib, json, time, sqlite3, base64, random, math
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

B = os.path.dirname(os.path.abspath(__file__))
SECRET = os.getenv("SECRET", "dev-secret-ecovest-2026").encode()
DB = os.getenv("DB_PATH", f"{B}/app.db")

# ---------- Data Ingestion & Preprocessing ----------
df = pd.read_csv(f"{B}/data/esg_dataset.csv")

# Train Machine Learning Models
ML_FEATURES = [
    'Renewable_Energy_Usage', 'Waste_Recycling_Rate', 'Carbon_Emission_Score',
    'Water_Management_Score', 'Employee_Satisfaction', 'Gender_Diversity',
    'Governance_Score', 'ESG_Controversies', 'ROE', 'Stock_Volatility'
]
X_all = df[ML_FEATURES].values
y_score = df['Overall_ESG_Score'].values
y_rec   = df['Recommendation'].values
y_risk  = df['Investment_Risk'].values

# 80/20 held-out split for Model Performance card
X_tr, X_te, ys_tr, ys_te, yr_tr, yr_te = train_test_split(
    X_all, y_score, y_rec, test_size=0.2, random_state=42
)

reg_esg  = RandomForestRegressor(n_estimators=45, max_depth=12, random_state=42).fit(X_tr, ys_tr)
clf_rec  = RandomForestClassifier(n_estimators=45, max_depth=12, random_state=42).fit(X_tr, yr_tr)
clf_risk = RandomForestClassifier(n_estimators=45, max_depth=12, random_state=42).fit(X_all, y_risk)

RF_R2  = round(float(r2_score(ys_te, reg_esg.predict(X_te))), 4)
RF_ACC = round(float(accuracy_score(yr_te, clf_rec.predict(X_te))) * 100, 2)

FEAT_IMP = {
    feat: round(float(imp) * 100, 1)
    for feat, imp in zip(ML_FEATURES, reg_esg.feature_importances_)
}

# ---------- Pre-compute KMeans Clusters ----------
KM_COLS = [
    'Carbon_Emission_Score', 'Renewable_Energy_Usage', 'Waste_Recycling_Rate',
    'Employee_Satisfaction', 'Governance_Score', 'ROE', 'Stock_Volatility'
]
_km_scaler = StandardScaler()
_km_X = _km_scaler.fit_transform(df[KM_COLS].values)
_km = KMeans(n_clusters=3, random_state=42, n_init=10)
_km.fit(_km_X)
_km_labels = _km.labels_

_cluster_esg_means = {
    i: float(df.loc[_km_labels == i, 'Overall_ESG_Score'].mean())
    for i in range(3)
}
_sorted_clusters = sorted(_cluster_esg_means.items(), key=lambda x: x[1])
_CLUSTER_NAMES = {
    _sorted_clusters[0][0]: "ESG Laggards",
    _sorted_clusters[1][0]: "Mixed",
    _sorted_clusters[2][0]: "ESG Leaders"
}

# ---------- Pre-compute Hill Climbing Scaler ----------
HC_COLS = ['Overall_ESG_Score', 'Stock_Price_Growth', 'Dividend_Yield', 'Stock_Volatility']
_hc_scaler = StandardScaler()
_hc_X = _hc_scaler.fit_transform(df[HC_COLS].values)
_hc_nn = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(_hc_X)

def _hc_score(row):
    return (0.5 * row['Overall_ESG_Score']
            + 0.2 * row['Stock_Price_Growth']
            + 0.15 * row['Dividend_Yield']
            - 0.1 * row['Stock_Volatility'])

df['_hc_score'] = df.apply(_hc_score, axis=1)

# ---------- Pre-compute ID3 Decision Tree ----------
def _entropy(vals):
    n = len(vals)
    if n == 0: return 0.0
    cnts = pd.Series(vals).value_counts()
    probs = cnts / n
    return float(-sum(p * math.log2(p) for p in probs if p > 0))

def _reu_band(v):
    if v <= 35: return "Low"
    if v <= 65: return "Medium"
    return "High"

def _wry_band(v):
    if v <= 40: return "Low"
    if v <= 65: return "Medium"
    return "High"

_dtdf = df[['Renewable_Energy_Usage', 'Waste_Recycling_Rate', 'Recommendation']].copy()
_dtdf['reu'] = _dtdf['Renewable_Energy_Usage'].apply(_reu_band)
_dtdf['wry'] = _dtdf['Waste_Recycling_Rate'].apply(_wry_band)

def _compute_decision_tree():
    root_ent = _entropy(_dtdf['Recommendation'].values)

    def info_gain(col):
        total = len(_dtdf)
        ig = root_ent
        for val in _dtdf[col].unique():
            sub = _dtdf[_dtdf[col] == val]['Recommendation'].values
            ig -= (len(sub) / total) * _entropy(sub)
        return round(ig, 4)

    ig_reu = info_gain('reu')
    ig_wry = info_gain('wry')

    leaves = []
    for reu_val in ["Low", "Medium", "High"]:
        for wry_val in ["Low", "Medium", "High"]:
            sub = _dtdf[(_dtdf['reu'] == reu_val) & (_dtdf['wry'] == wry_val)]['Recommendation']
            cnts = sub.value_counts().to_dict()
            ent = round(_entropy(sub.values), 4)
            majority = sub.mode()[0] if len(sub) > 0 else "Hold"
            leaves.append({
                "reu": reu_val,
                "wry": wry_val,
                "counts": {"Avoid": int(cnts.get("Avoid", 0)), "Hold": int(cnts.get("Hold", 0)), "Buy": int(cnts.get("Buy", 0))},
                "entropy": ent,
                "majority": majority,
                "total": int(len(sub))
            })

    return {
        "root_entropy": round(root_ent, 4),
        "ig_reu": ig_reu,
        "ig_wry": ig_wry,
        "root_feature": "Renewable_Energy_Usage" if ig_reu >= ig_wry else "Waste_Recycling_Rate",
        "leaves": leaves
    }

DT_DATA = _compute_decision_tree()

# ---------- Pre-compute Dashboard Analytics ----------
def compute_dashboard_analytics():
    total_companies = len(df)
    avg_esg = round(float(df['Overall_ESG_Score'].mean()), 2)
    rec_counts = df['Recommendation'].value_counts().to_dict()
    risk_counts = df['Investment_Risk'].value_counts().to_dict()

    sector_esg = df.groupby('Industry_Sector')['Overall_ESG_Score'].mean().round(2).to_dict()
    sorted_sectors = sorted(sector_esg.items(), key=lambda x: x[1], reverse=True)

    pillar_data = []
    for sector, group in df.groupby('Industry_Sector'):
        e_score = round(float(((group['Renewable_Energy_Usage'] + group['Waste_Recycling_Rate'] + group['Water_Management_Score'] + (100 - group['Carbon_Emission_Score'])) / 4.0).mean()), 1)
        s_score = round(float(((group['Employee_Satisfaction'] + group['Gender_Diversity'] * 2.0) / 2.0).mean()), 1)
        g_score = round(float(((group['Governance_Score'] + (100 - group['ESG_Controversies'] * 10).clip(lower=0)) / 2.0).mean()), 1)
        pillar_data.append({
            "sector": sector,
            "environmental": e_score,
            "social": s_score,
            "governance": g_score,
            "overall": round(float(group['Overall_ESG_Score'].mean()), 1)
        })

    country_esg = df.groupby('Country')['Overall_ESG_Score'].mean().round(2).to_dict()
    sorted_countries = sorted(country_esg.items(), key=lambda x: x[1], reverse=True)

    hist_counts, bin_edges = np.histogram(df['Overall_ESG_Score'], bins=[25, 35, 45, 55, 65, 75, 85, 95])
    distribution = [
        {"range": f"{int(bin_edges[i])}–{int(bin_edges[i+1])}", "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    cols = ["Company_ID", "Company_Name", "Industry_Sector", "Country", "Overall_ESG_Score", "Investment_Risk", "Recommendation", "Market_Cap_Billion", "ROE"]
    sorted_df = df.sort_values('Overall_ESG_Score')
    top_5    = json.loads(sorted_df.tail(5)[::-1][cols].to_json(orient='records'))
    bottom_5 = json.loads(sorted_df.head(5)[cols].to_json(orient='records'))

    return {
        "kpis": {
            "total_companies": total_companies,
            "avg_esg_score": avg_esg,
            "buy_count":  rec_counts.get("Buy", 0),
            "hold_count": rec_counts.get("Hold", 0),
            "avoid_count": rec_counts.get("Avoid", 0),
            "low_risk_count":    risk_counts.get("Low", 0),
            "medium_risk_count": risk_counts.get("Medium", 0),
            "high_risk_count":   risk_counts.get("High", 0),
            "total_market_cap_trillion": round(float(df['Market_Cap_Billion'].sum() / 1000.0), 2)
        },
        "recommendations_donut": rec_counts,
        "risk_breakdown": risk_counts,
        "sector_rankings": sorted_sectors,
        "pillar_by_sector": pillar_data,
        "country_rankings": sorted_countries,
        "score_distribution": distribution,
        "top_champions": top_5,
        "bottom_laggards": bottom_5,
        "feature_importances": FEAT_IMP,
        "rf_r2": RF_R2,
        "rf_accuracy": RF_ACC
    }

DASHBOARD_DATA = compute_dashboard_analytics()

# ---------- Database and Auth Setup ----------
def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

with db() as c:
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE,
            pw TEXT,
            salt TEXT
        );
        CREATE TABLE IF NOT EXISTS hist (
            id INTEGER PRIMARY KEY,
            uid INT,
            ts REAL,
            inp TEXT,
            out TEXT
        );
        CREATE TABLE IF NOT EXISTS wl (
            uid INT,
            cid INT,
            PRIMARY KEY(uid, cid)
        );
    """)

def hpw(p: str, s: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", p.encode(), s.encode(), 100000).hex()

def mk_token(uid: int) -> str:
    m = f"{uid}:{int(time.time()) + 86400 * 7}".encode()
    return base64.urlsafe_b64encode(m).decode() + "." + hmac.new(SECRET, m, "sha256").hexdigest()

def get_current_user(authorization: Optional[str] = Header(None)) -> int:
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Authentication required")
        b, s = authorization.split(" ")[1].split(".")
        m = base64.urlsafe_b64decode(b)
        assert hmac.compare_digest(s, hmac.new(SECRET, m, "sha256").hexdigest())
        uid, exp = map(int, m.split(b":"))
        assert exp > time.time()
        return uid
    except Exception:
        raise HTTPException(401, "Invalid or expired session. Please log in.")

# ---------- FastAPI Application ----------
app = FastAPI(
    title="EcoVest AI – ESG Investment Advisor & Sustainability Intelligence",
    description="Intelligent ESG scoring, investment recommendation simulation, and corporate sustainability analytics."
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.endswith(".html") or request.url.path == "/" or request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

class Cred(BaseModel):
    email: str
    password: str

class SimulatorInput(BaseModel):
    renewable_energy: float = Field(65.0, ge=0.0, le=100.0)
    waste_recycling: float = Field(60.0, ge=0.0, le=100.0)
    carbon_emission_score: float = Field(45.0, ge=0.0, le=100.0)
    water_management: float = Field(70.0, ge=0.0, le=100.0)
    employee_satisfaction: float = Field(75.0, ge=0.0, le=100.0)
    gender_diversity: float = Field(35.0, ge=0.0, le=100.0)
    governance_score: float = Field(75.0, ge=0.0, le=100.0)
    esg_controversies: int = Field(1, ge=0, le=10)
    roe: float = Field(18.0, ge=-30.0, le=60.0)
    stock_volatility: float = Field(25.0, ge=5.0, le=80.0)
    simulation_label: Optional[str] = Field("Custom Simulated Company")

class CompanyEvaluationInput(BaseModel):
    company_name: str = Field(..., description="Name of company to evaluate")
    industry_sector: str = Field("IT", description="Industry sector")
    country: str = Field("India", description="Country / Headquarters")
    market_cap_billion: Optional[float] = Field(25.0, description="Estimated market cap")
    renewable_energy: float = Field(50.0, ge=0.0, le=100.0)
    waste_recycling: float = Field(50.0, ge=0.0, le=100.0)
    carbon_emission_score: float = Field(50.0, ge=0.0, le=100.0)
    water_management: float = Field(50.0, ge=0.0, le=100.0)
    employee_satisfaction: float = Field(65.0, ge=0.0, le=100.0)
    gender_diversity: float = Field(30.0, ge=0.0, le=100.0)
    governance_score: float = Field(65.0, ge=0.0, le=100.0)
    esg_controversies: int = Field(1, ge=0, le=10)
    roe: float = Field(15.0, ge=-50.0, le=100.0)
    stock_volatility: float = Field(25.0, ge=5.0, le=90.0)

# ---------- Auth Endpoints ----------
@app.post("/api/register")
def register(c: Cred):
    if len(c.password) < 6 or "@" not in c.email:
        raise HTTPException(400, "Valid email and 6+ character password required")
    s = os.urandom(8).hex()
    try:
        with db() as d:
            uid = d.execute(
                "INSERT INTO users(email, pw, salt) VALUES(?, ?, ?)",
                (c.email.lower().strip(), hpw(c.password, s), s)
            ).lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Email already registered")
    return {"token": mk_token(uid), "email": c.email.lower().strip()}

@app.post("/api/login")
def login(c: Cred):
    u = db().execute("SELECT * FROM users WHERE email=?", (c.email.lower().strip(),)).fetchone()
    if not u or not hmac.compare_digest(u["pw"], hpw(c.password, u["salt"])):
        raise HTTPException(401, "Invalid email or password")
    return {"token": mk_token(u["id"]), "email": u["email"]}

# ---------- Core Dashboard Endpoint ----------
@app.get("/api/dashboard")
def get_dashboard():
    """Returns complete executive dashboard metrics, distributions, pillar comparisons, and rankings."""
    return DASHBOARD_DATA

# ---------- AI Techniques Lab: K-Means Clusters ----------
@app.get("/api/clusters")
def get_clusters():
    """K-Means k=3 on ESG features. Returns cluster labels, means, and scatter sample."""
    cluster_info = []
    sample_rows = []

    for raw_id, name in _CLUSTER_NAMES.items():
        mask = _km_labels == raw_id
        sub = df[mask]
        size = int(mask.sum())
        means = {col: round(float(sub[col].mean()), 2) for col in KM_COLS}
        means['Overall_ESG_Score'] = round(float(sub['Overall_ESG_Score'].mean()), 2)
        cluster_info.append({
            "cluster_id": raw_id,
            "label": name,
            "size": size,
            "means": means
        })

        # Sample up to 200 points per cluster for scatter
        samp = sub.sample(min(200, size), random_state=42)
        for _, row in samp.iterrows():
            sample_rows.append({
                "company": row['Company_Name'],
                "roe": round(float(row['ROE']), 2),
                "esg": round(float(row['Overall_ESG_Score']), 2),
                "cluster": name
            })

    cluster_info.sort(key=lambda x: x['means']['Overall_ESG_Score'])
    return {"clusters": cluster_info, "scatter": sample_rows}

# ---------- AI Techniques Lab: Decision Tree ----------
@app.get("/api/decision-tree")
def get_decision_tree():
    """ID3-style decision tree on discretised ESG bands."""
    return DT_DATA

# ---------- AI Techniques Lab: Hill Climbing ----------
@app.post("/api/hillclimb")
def run_hillclimb():
    """Hill climbing local search starting from a random weak company."""
    threshold = float(df['_hc_score'].quantile(0.15))
    weak = df[df['_hc_score'] <= threshold]
    start_idx = int(weak.sample(1, random_state=random.randint(0, 9999)).index[0])

    path = []
    current_idx = start_idx
    visited = set()

    for step in range(50):
        if current_idx in visited:
            break
        visited.add(current_idx)

        row = df.iloc[current_idx]
        path.append({
            "step": step + 1,
            "company_id": int(row['Company_ID']),
            "company_name": str(row['Company_Name']),
            "sector": str(row['Industry_Sector']),
            "score": round(float(row['_hc_score']), 4),
            "esg": round(float(row['Overall_ESG_Score']), 2),
            "growth": round(float(row['Stock_Price_Growth']), 2),
            "div_yield": round(float(row['Dividend_Yield']), 2),
            "volatility": round(float(row['Stock_Volatility']), 2),
            "recommendation": str(row['Recommendation'])
        })

        z_point = _hc_X[current_idx].reshape(1, -1)
        distances, indices = _hc_nn.kneighbors(z_point)
        neighbours = [i for i in indices[0] if i != current_idx]

        best_idx = current_idx
        best_score = float(df.iloc[current_idx]['_hc_score'])
        for nb in neighbours:
            nb_score = float(df.iloc[nb]['_hc_score'])
            if nb_score > best_score:
                best_score = nb_score
                best_idx = nb

        if best_idx == current_idx:
            break
        current_idx = best_idx

    return {"path": path, "steps": len(path)}

# ---------- Simulator Prediction Endpoint ----------
@app.post("/api/simulate")
def run_simulation(inp: SimulatorInput, authorization: Optional[str] = Header(None)):
    feat_vector = np.array([[
        inp.renewable_energy,
        inp.waste_recycling,
        inp.carbon_emission_score,
        inp.water_management,
        inp.employee_satisfaction,
        inp.gender_diversity,
        inp.governance_score,
        inp.esg_controversies,
        inp.roe,
        inp.stock_volatility
    ]])

    pred_esg  = round(float(reg_esg.predict(feat_vector)[0]), 2)
    pred_rec  = str(clf_rec.predict(feat_vector)[0])
    pred_risk = str(clf_risk.predict(feat_vector)[0])

    env_score = round((inp.renewable_energy + inp.waste_recycling + inp.water_management + (100.0 - inp.carbon_emission_score)) / 4.0, 1)
    soc_score = round((inp.employee_satisfaction + min(100.0, inp.gender_diversity * 2.0)) / 2.0, 1)
    gov_score = round((inp.governance_score + max(0.0, 100.0 - inp.esg_controversies * 10.0)) / 2.0, 1)
    fin_score = round(min(100.0, max(0.0, (inp.roe + 20.0) * 1.5 + (80.0 - inp.stock_volatility) * 0.5)), 1)

    strengths = []
    red_flags = []
    action_items = []

    if inp.renewable_energy >= 70:
        strengths.append(f"Strong clean energy adoption ({inp.renewable_energy}% Renewable Energy Usage)")
    else:
        red_flags.append(f"Low renewable energy reliance ({inp.renewable_energy}%)")
        action_items.append("Transition power purchasing agreements to 75%+ solar/wind to lower operational grid footprint.")

    if inp.carbon_emission_score <= 35:
        strengths.append(f"Low operational carbon intensity (Score: {inp.carbon_emission_score})")
    elif inp.carbon_emission_score >= 60:
        red_flags.append(f"High carbon emission score ({inp.carbon_emission_score}) penalizing rating")
        action_items.append("Deploy energy efficiency retrofits and electrification to lower carbon emission score below 40.")

    if inp.waste_recycling >= 70:
        strengths.append(f"High circular economy recycling rate ({inp.waste_recycling}%)")
    else:
        action_items.append("Implement closed-loop waste recycling to target >65% recovery rate.")

    if inp.governance_score >= 80:
        strengths.append(f"Exemplary corporate governance oversight (Score: {inp.governance_score})")
    elif inp.governance_score <= 60:
        red_flags.append(f"Sub-optimal governance score ({inp.governance_score})")
        action_items.append("Strengthen board independence, anti-corruption policies, and ESG disclosure transparency.")

    if inp.esg_controversies >= 5:
        red_flags.append(f"High controversy exposure ({inp.esg_controversies} recorded incidents)")
        action_items.append("Conduct compliance audits to resolve active labor/environmental controversies.")

    if inp.stock_volatility <= 25 and inp.roe >= 15:
        strengths.append(f"Healthy financial stability (ROE: {inp.roe}%, Volatility: {inp.stock_volatility}%)")

    result = {
        "label": inp.simulation_label,
        "predicted_esg_score": pred_esg,
        "recommendation": pred_rec,
        "investment_risk": pred_risk,
        "pillar_scores": {
            "environmental": env_score,
            "social": soc_score,
            "governance": gov_score,
            "financial_stability": fin_score
        },
        "radar_profile": {
            "labels": [
                "Clean Energy", "Waste Circularity", "Carbon Efficiency",
                "Water Stewardship", "Social & Workforce", "Governance Quality", "Financial Return"
            ],
            "values": [
                round(inp.renewable_energy, 1),
                round(inp.waste_recycling, 1),
                round(100.0 - inp.carbon_emission_score, 1),
                round(inp.water_management, 1),
                round(inp.employee_satisfaction, 1),
                round(inp.governance_score, 1),
                round(min(100.0, max(0.0, (inp.roe + 20.0) * 1.5)), 1)
            ],
            "market_average": [50.2, 50.1, 49.8, 50.4, 60.1, 55.3, 52.0]
        },
        "thesis": {
            "strengths":    strengths    if strengths    else ["Balanced baseline sustainability metrics"],
            "red_flags":    red_flags    if red_flags    else ["No critical red flags detected"],
            "action_items": action_items if action_items else ["Maintain current sustainability leadership and monitor market volatility"]
        }
    }

    try:
        if authorization:
            uid = get_current_user(authorization)
            with db() as d:
                d.execute(
                    "INSERT INTO hist(uid, ts, inp, out) VALUES(?, ?, ?, ?)",
                    (uid, time.time(), inp.model_dump_json(), json.dumps(result))
                )
    except Exception:
        pass

    return result

# ---------- DEDICATED COMPANY EVALUATOR ENDPOINT ----------
@app.post("/api/evaluate")
def evaluate_company(inp: CompanyEvaluationInput, authorization: Optional[str] = Header(None)):
    feat_vector = np.array([[
        inp.renewable_energy,
        inp.waste_recycling,
        inp.carbon_emission_score,
        inp.water_management,
        inp.employee_satisfaction,
        inp.gender_diversity,
        inp.governance_score,
        inp.esg_controversies,
        inp.roe,
        inp.stock_volatility
    ]])

    pred_esg  = round(float(reg_esg.predict(feat_vector)[0]), 2)
    pred_rec  = str(clf_rec.predict(feat_vector)[0])
    pred_risk = str(clf_risk.predict(feat_vector)[0])

    percentile = round(float((df['Overall_ESG_Score'] < pred_esg).mean() * 100), 1)

    env_score = round((inp.renewable_energy + inp.waste_recycling + inp.water_management + (100.0 - inp.carbon_emission_score)) / 4.0, 1)
    soc_score = round((inp.employee_satisfaction + min(100.0, inp.gender_diversity * 2.0)) / 2.0, 1)
    gov_score = round((inp.governance_score + max(0.0, 100.0 - inp.esg_controversies * 10.0)) / 2.0, 1)
    fin_score = round(min(100.0, max(0.0, (inp.roe + 20.0) * 1.5 + (80.0 - inp.stock_volatility) * 0.5)), 1)

    def to_grade(score):
        if score >= 80: return "A+ (Excellent)"
        if score >= 70: return "A (Strong)"
        if score >= 60: return "B (Moderate)"
        if score >= 50: return "C (Average)"
        return "D (Deficient)"

    if pred_rec == "Buy":
        verdict_title = "INVESTMENT WORTHY — HIGH CONVICTION BUY"
        verdict_desc = f"{inp.company_name} demonstrates exceptional sustainability leadership, top-decile ESG governance ({pred_esg}/100), and robust financial health. It is strongly recommended for institutional ESG and long-term capital appreciation portfolios."
        investor_action = "PROCEED TO ALLOCATE: Add to sustainable core portfolio with a 12-to-36 month investment horizon."
    elif pred_rec == "Hold":
        verdict_title = "PROCEED WITH CAUTION — HOLD / MONITOR"
        verdict_desc = f"{inp.company_name} exhibits moderate sustainability readiness ({pred_esg}/100). While foundational ESG practices are active, key vulnerabilities in clean energy or emissions require ongoing monitoring before increasing capital exposure."
        investor_action = "MONITOR QUARTERLY: Maintain current position or wait for verified green transition milestone announcements before scaling up investment."
    else:
        verdict_title = "NOT RECOMMENDED — HIGH RISK (AVOID)"
        verdict_desc = f"{inp.company_name} carries an unfavorable risk-adjusted ESG profile ({pred_esg}/100). Elevated carbon intensity, weak circularity, or controversy exposure present substantial regulatory, climate transition, and reputational liabilities."
        investor_action = "AVOID OR DIVEST: Exclude from sustainable mandates and hedge existing positions against potential ESG downgrade risks."

    strengths = []
    weaknesses = []

    if inp.renewable_energy >= 70:
        strengths.append(f"Clean Energy: Outstanding renewable energy integration at {inp.renewable_energy}%.")
    elif inp.renewable_energy <= 35:
        weaknesses.append(f"Grid Dependency: High reliance on non-renewable fossil power ({100 - inp.renewable_energy}%).")

    if inp.carbon_emission_score <= 35:
        strengths.append(f"Decarbonization: Exemplary low carbon emissions footprint (Score: {inp.carbon_emission_score}).")
    elif inp.carbon_emission_score >= 60:
        weaknesses.append(f"Emissions Hazard: Severe carbon intensity score ({inp.carbon_emission_score}) creates regulatory carbon-tax vulnerability.")

    if inp.governance_score >= 75:
        strengths.append(f"Board Oversight: High corporate governance and audit integrity ({inp.governance_score}/100).")
    else:
        weaknesses.append(f"Governance Risk: Board independence and transparency metrics below institutional threshold ({inp.governance_score}/100).")

    if inp.waste_recycling >= 70:
        strengths.append(f"Circularity: Advanced waste recycling and recovery systems ({inp.waste_recycling}%).")

    if inp.esg_controversies >= 3:
        weaknesses.append(f"Active Litigation: {inp.esg_controversies} recorded public ESG controversies increase operational and reputational risk.")

    if inp.roe >= 18 and inp.stock_volatility <= 30:
        strengths.append(f"Capital Efficiency: Robust Return on Equity ({inp.roe}%) combined with controlled stock volatility ({inp.stock_volatility}%).")

    result = {
        "company_name": inp.company_name,
        "industry_sector": inp.industry_sector,
        "country": inp.country,
        "market_cap_billion": inp.market_cap_billion,
        "esg_score": pred_esg,
        "percentile": percentile,
        "recommendation": pred_rec,
        "investment_risk": pred_risk,
        "verdict_title": verdict_title,
        "verdict_desc": verdict_desc,
        "investor_action": investor_action,
        "pillar_scores": {
            "environmental": {"score": env_score, "grade": to_grade(env_score)},
            "social":        {"score": soc_score, "grade": to_grade(soc_score)},
            "governance":    {"score": gov_score, "grade": to_grade(gov_score)},
            "financial_stability": {"score": fin_score, "grade": to_grade(fin_score)}
        },
        "strengths":  strengths  if strengths  else ["Satisfies baseline regulatory standards"],
        "weaknesses": weaknesses if weaknesses else ["No critical red flags identified"],
        "radar_profile": {
            "labels": ["Clean Energy", "Waste Circularity", "Carbon Efficiency", "Water Stewardship", "Social Equity", "Governance", "Financial Health"],
            "values": [
                round(inp.renewable_energy, 1),
                round(inp.waste_recycling, 1),
                round(100.0 - inp.carbon_emission_score, 1),
                round(inp.water_management, 1),
                round(inp.employee_satisfaction, 1),
                round(inp.governance_score, 1),
                round(min(100.0, max(0.0, (inp.roe + 20.0) * 1.5)), 1)
            ],
            "benchmark": [50.2, 50.1, 49.8, 50.4, 60.1, 55.3, 52.0]
        }
    }

    try:
        if authorization:
            uid = get_current_user(authorization)
            with db() as d:
                d.execute(
                    "INSERT INTO hist(uid, ts, inp, out) VALUES(?, ?, ?, ?)",
                    (uid, time.time(), inp.model_dump_json(), json.dumps(result))
                )
    except Exception:
        pass

    return result

# ---------- Companies Registry Endpoints ----------
COLS_EXP = [
    "Company_ID", "Company_Name", "Industry_Sector", "Country",
    "Market_Cap_Billion", "Annual_Revenue_Billion", "Net_Profit_Billion",
    "ROE", "Carbon_Emission_Score", "Renewable_Energy_Usage", "Waste_Recycling_Rate",
    "Employee_Satisfaction", "Governance_Score", "Stock_Volatility", "Dividend_Yield",
    "Overall_ESG_Score", "Investment_Risk", "Recommendation"
]

@app.get("/api/companies")
def get_companies(
    page: int = 1,
    q: str = "",
    sector: str = "",
    country: str = "",
    rec: str = "",
    risk: str = "",
    sort_by: str = "Overall_ESG_Score",
    order: str = "desc"
):
    s = df
    if q:       s = s[s['Company_Name'].str.contains(q, case=False, na=False)]
    if sector:  s = s[s['Industry_Sector'] == sector]
    if country: s = s[s['Country'] == country]
    if rec:     s = s[s['Recommendation'] == rec]
    if risk:    s = s[s['Investment_Risk'] == risk]

    ascending = (order.lower() == "asc")
    if sort_by in s.columns:
        s = s.sort_values(sort_by, ascending=ascending)

    total = len(s)
    page_size = 12
    start_idx = (page - 1) * page_size
    items = json.loads(s.iloc[start_idx: start_idx + page_size][COLS_EXP].to_json(orient="records"))

    return {"total": total, "page": page, "page_size": page_size, "items": items}

@app.get("/api/company/{cid}")
def get_company_detail(cid: int):
    if not (1 <= cid <= len(df)):
        raise HTTPException(404, "Company not found")
    c = json.loads(df.iloc[[cid - 1]].to_json(orient="records"))[0]
    return {"company": c}

# ---------- User History & Watchlist Endpoints ----------
@app.get("/api/history")
def get_history(authorization: str = Header(None)):
    uid = get_current_user(authorization)
    rows = db().execute("SELECT * FROM hist WHERE uid=? ORDER BY id DESC LIMIT 25", (uid,)).fetchall()
    results = []
    for r in rows:
        try:
            results.append({
                "id": r["id"],
                "ts": r["ts"],
                "input": json.loads(r["inp"]),
                "output": json.loads(r["out"])
            })
        except Exception:
            continue
    return results

@app.get("/api/watchlist")
def get_watchlist(authorization: str = Header(None)):
    uid = get_current_user(authorization)
    ids = [r["cid"] for r in db().execute("SELECT cid FROM wl WHERE uid=?", (uid,))]
    if not ids:
        return []
    return json.loads(df[df.Company_ID.isin(ids)][COLS_EXP].to_json(orient="records"))

@app.post("/api/watchlist/{cid}")
def add_watchlist(cid: int, authorization: str = Header(None)):
    uid = get_current_user(authorization)
    with db() as d:
        d.execute("INSERT OR IGNORE INTO wl VALUES(?, ?)", (uid, cid))
    return {"ok": True}

@app.delete("/api/watchlist/{cid}")
def delete_watchlist(cid: int, authorization: str = Header(None)):
    uid = get_current_user(authorization)
    with db() as d:
        d.execute("DELETE FROM wl WHERE uid=? AND cid=?", (uid, cid))
    return {"ok": True}

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "EcoVest AI", "dataset_records": len(df)}

@app.get("/")
def serve_index():
    return FileResponse(
        f"{B}/static/index.html",
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

app.mount("/", StaticFiles(directory=f"{B}/static", html=True), name="static")
