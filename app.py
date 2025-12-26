import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polyline
import pydeck as pdk
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO
from google import genai
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Running Coach", page_icon="🏃‍♂️", layout="wide")

# --- LOAD SECRETS ---
try:
    STRAVA_CLIENT_ID = st.secrets['STRAVA_CLIENT_ID']
    STRAVA_CLIENT_SECRET = st.secrets['STRAVA_CLIENT_SECRET']
    GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']
except FileNotFoundError:
    load_dotenv()
    STRAVA_CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
    STRAVA_CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- 💾 SETTINGS MANAGEMENT ---
SETTINGS_FILE = "user_settings.json"

def load_settings():
    default_settings = {
        "race_date": "2026-02-22",
        "race_goal_time": "3:45:00",
        "race_name": "Marathon"
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return default_settings
    return default_settings

def save_settings(race_date, goal_time, race_name):
    settings = {
        "race_date": str(race_date),
        "race_goal_time": goal_time,
        "race_name": race_name
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

# --- 📱 CSS STYLING (UPDATED FOR COMPACTNESS) ---
st.markdown("""
<style>
    /* 1. COMPACT METRICS (Performance Summary) */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important; /* Reduced from 3rem to fit small screens */
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    
    /* 2. COMPACT TROPHY GRID */
    .trophy-grid { 
        display: grid; 
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); /* Narrower cards */
        gap: 8px; 
        margin-bottom: 25px; 
    }
    
    /* 3. TROPHY CARD STYLING */
    .trophy-card { 
        background-color: #ffffff; 
        border: 1px solid #e2e8f0; 
        border-left: 4px solid #f59e0b; 
        border-radius: 6px; 
        padding: 8px; 
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05); 
    }
    .trophy-title { 
        font-size: 0.65rem; 
        text-transform: uppercase; 
        color: #64748b; 
        font-weight: 700; 
    }
    .trophy-time { 
        font-size: 1.1rem; /* Smaller time font */
        font-weight: 700; 
        color: #0f172a; 
        font-family: monospace; 
        margin: 2px 0; 
    }
    .trophy-date { font-size: 0.6rem; color: #94a3b8; }
    
    /* 4. COUNTDOWN BOX */
    .countdown-box { 
        background-color: #f8fafc; 
        border: 1px solid #e2e8f0; 
        border-radius: 10px; 
        padding: 10px; 
        text-align: center; 
        margin-bottom: 20px; 
    }
    .countdown-header { font-size: 1.6rem !important; }
    
    /* Mobile Fixes */
    @media (max-width: 768px) {
        [data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; min-width: 100% !important; }
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
        [data-testid="stToolbar"] { visibility: hidden; }
    }
</style>
""", unsafe_allow_html=True)

# --- AUTHENTICATION ---
def get_auth_url():
    redirect_uri = "https://gemini-running-coach-n4auhgbfxaprhqpqjhorxe.streamlit.app"
    # redirect_uri = "http://localhost:8501" # Uncomment for local testing
    return f"http://www.strava.com/oauth/authorize?client_id={STRAVA_CLIENT_ID}&response_type=code&redirect_uri={redirect_uri}&approval_prompt=force&scope=activity:read_all"

def exchange_token(code):
    response = requests.post(
        'https://www.strava.com/oauth/token',
        data={'client_id': STRAVA_CLIENT_ID, 'client_secret': STRAVA_CLIENT_SECRET, 'code': code, 'grant_type': 'authorization_code'}
    )
    if response.status_code == 200: return response.json()['access_token']
    return None

# --- HELPER FUNCTIONS ---
def format_pace(pace_decimal):
    if pd.isna(pace_decimal) or pace_decimal == 0: return "0:00"
    minutes = int(pace_decimal)
    seconds = int((pace_decimal - minutes) * 60)
    return f"{minutes}:{seconds:02d}"

def format_duration(minutes):
    if pd.isna(minutes) or minutes == 0: return "0m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0: return f"{hours}h {mins}m"
    return f"{mins}m"

def decode_map(polyline_str):
    if not polyline_str: return pd.DataFrame(columns=['lat', 'lon'])
    coords = polyline.decode(polyline_str)
    return pd.DataFrame(coords, columns=['lat', 'lon'])

# --- IMAGE ENGINE ---
def load_font(size, variant="Modern"): return ImageFont.load_default()
def draw_frosted_glass(img, rect_coords, radius=15):
    box = img.crop(rect_coords).filter(ImageFilter.GaussianBlur(radius))
    white_layer = Image.new("RGBA", box.size, (255, 255, 255, 30))
    return Image.alpha_composite(box.convert("RGBA"), white_layer)

def generate_aesthetic_image(upload_file, run_data, config):
    try:
        img = Image.open(upload_file).convert("RGBA")
        target_width = 1080
        ratio = target_width / img.width
        target_height = int(img.height * ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        if config['style'] == "Minimal Card":
            img.paste(draw_frosted_glass(img, (50, height-300, width-50, height-50)), (50, height-300))
            draw.rectangle([50, height-300, width-50, height-50], outline="white", width=2)
            draw.text((100, height-200), f"{run_data['distance_km']:.2f} km", fill="white", font=load_font(60))
        return img.convert("RGB")
    except Exception: return None

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_strava_data(token):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {'Authorization': f"Bearer {token}"}
    one_year_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365)
    timestamp = int(one_year_ago.timestamp())
    all_activities = []
    page = 1
    while True:
        params = {'per_page': 200, 'page': page, 'after': timestamp}
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if not data or not isinstance(data, list): break
        runs = [act for act in data if act.get('type') == 'Run']
        all_activities.extend(runs)
        page += 1
    if not all_activities: return pd.DataFrame()

    df = pd.DataFrame(all_activities)
    df['start_date_local'] = pd.to_datetime(df['start_date_local'], utc=True)
    df['distance_km'] = df['distance'] / 1000
    df['duration_min'] = df['moving_time'] / 60
    df['pace_min_km'] = df['duration_min'] / df['distance_km']
    df['avg_hr'] = df['average_heartrate'] if 'average_heartrate' in df.columns else None
    df['cadence'] = df['average_cadence'] * 2 if 'average_cadence' in df.columns else 0 
    if df['cadence'].mean() > 240: df['cadence'] = df['cadence'] / 2
    df['elevation'] = df['total_elevation_gain'] if 'total_elevation_gain' in df.columns else 0
    df['map_polyline'] = df['map'].apply(lambda x: x.get('summary_polyline') if isinstance(x, dict) else None) if 'map' in df.columns else None
    df = df.sort_values(by='start_date_local', ascending=False).reset_index(drop=True)
    return df

# --- 🧠 GEMINI FUNCTIONS ---
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_run_analysis(run, df_history, goal_summary):
    client = get_gemini_client()
    run_date = run['start_date_local']
    start_window = run_date - timedelta(days=30)
    month_history = df_history[(df_history['start_date_local'] < run_date) & (df_history['start_date_local'] >= start_window)]
    
    avg_pace_month = month_history['pace_min_km'].mean() if not month_history.empty else 0
    total_dist_month = month_history['distance_km'].sum()
    
    prompt = f"""
    Act as a tough but fair running coach.
    1. ANALYZE THIS RUN: {run_date.date()}, {run['distance_km']:.2f}km, {format_pace(run['pace_min_km'])}/km, HR: {run['avg_hr']}
    2. CONTEXT (Last 30 Days): Avg Pace: {format_pace(avg_pace_month)}/km, Vol: {total_dist_month:.1f} km
    3. GOAL: {goal_summary}
    TASK: Compare to baseline. Feedback on HR/Pace. Roast if needed. 1 Actionable tip.
    """
    try:
        response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        return response.text
    except Exception as e: return f"AI Brain Freeze: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Goals & Settings")
    current_settings = load_settings()
    with st.form("goal_form"):
        st.caption("Update your race target here.")
        target_date_input = st.date_input("Race Date", date.fromisoformat(current_settings["race_date"]))
        target_time_input = st.text_input("Target Time (HH:MM:SS)", current_settings["race_goal_time"])
        target_name_input = st.text_input("Event Name", current_settings.get("race_name", "Marathon"))
        if st.form_submit_button("💾 Save Goal"):
            save_settings(target_date_input, target_time_input, target_name_input)
            st.rerun()

    today = date.today()
    days_left = (target_date_input - today).days
    try:
        h, m, s = map(int, target_time_input.split(':'))
        req_pace = (h*60 + m + s/60) / 42.195
        req_pace_fmt = format_pace(req_pace)
    except: req_pace_fmt = "--:--"
        
    st.markdown(f"""
    <div class="countdown-box">
        <div style="font-size:0.7rem;color:#64748b;font-weight:bold;text-transform:uppercase;">{target_name_input}</div>
        <div class="countdown-header" style="font-weight:800;color:#0284c7;">{days_left} Days</div>
        <div style="margin-top:5px;font-size:0.7rem;color:#64748b;font-weight:bold;">REQ PACE: {req_pace_fmt}/km</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    if st.button("🔄 Sync Data"): st.cache_data.clear(); st.rerun()
    if st.button("🚪 Logout"): st.session_state.clear(); st.rerun()

# --- MAIN APP UI ---
st.title("🏃‍♂️ Gemini Running Coach")
auth_code = st.query_params.get("code")

if "access_token" not in st.session_state:
    if auth_code:
        token = exchange_token(auth_code)
        if token: st.session_state["access_token"] = token; st.rerun()
        else: st.error("Login failed.")
    else:
        st.info("Please log in."); st.link_button("🔗 Connect with Strava", get_auth_url()); st.stop()

token = st.session_state["access_token"]
with st.spinner("Syncing Strava Activities..."):
    df = fetch_strava_data(token)

if df.empty: st.warning("No runs found."); st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🏃 Run Details", "🧠 AI Coach", "📈 Trends"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.markdown("### ⚡ Performance Summary")
    time_opts = {"1 Week": 7, "15 Days": 15, "1 Month": 30, "3 Months": 90, "6 Months": 180}
    sel_time = st.radio("Duration:", options=list(time_opts.keys()), index=2, horizontal=True, label_visibility="collapsed")
    days = time_opts[sel_time]
    
    now = pd.Timestamp.now(tz='UTC')
    r_curr = df[df['start_date_local'] >= (now - pd.Timedelta(days=days))]
    r_prev = df[(df['start_date_local'] >= (now - pd.Timedelta(days=days*2))) & (df['start_date_local'] < (now - pd.Timedelta(days=days)))]
    
    def get_mean(d, c): return d[c].mean() if len(d) > 0 and c in d.columns else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    
    curr_dist = r_curr['distance_km'].sum()
    c1.metric("Distance", f"{curr_dist:.1f} km", f"{curr_dist - r_prev['distance_km'].sum():.1f} km")
    
    curr_pace = get_mean(r_curr, 'pace_min_km')
    c2.metric("Pace", f"{format_pace(curr_pace)}", f"{curr_pace - get_mean(r_prev, 'pace_min_km'):.2f} m/k", delta_color="inverse")
    
    curr_elev = r_curr['elevation'].sum()
    c3.metric("Elev", f"{int(curr_elev)} m", f"{int(curr_elev - r_prev['elevation'].sum())} m")
    
    curr_hr = get_mean(r_curr, 'avg_hr')
    c4.metric("HR", f"{int(curr_hr)} bpm", f"{int(curr_hr - get_mean(r_prev, 'avg_hr'))}", delta_color="inverse")
    
    c5.metric("Runs", len(r_curr), len(r_curr) - len(r_prev))

    st.divider()
    st.markdown("### 🏆 Personal Bests (Estimated)")
    trophies = [{"name": "1 km", "dist": 1.0}, {"name": "1 Mile", "dist": 1.609}, {"name": "5 km", "dist": 5.0}, {"name": "10 km", "dist": 10.0}, {"name": "Half Marathon", "dist": 21.1}, {"name": "Marathon", "dist": 42.2}]
    html_trophy = '<div class="trophy-grid">'
    for t in trophies:
        q_runs = df[df['distance_km'] >= (t['dist'] * 0.98)].copy() 
        if not q_runs.empty:
            best = q_runs.sort_values('pace_min_km').iloc[0]
            tm = best['pace_min_km'] * t['dist']
            time_str = f"{int(tm//60)}:{int(tm%60):02d}:{int((tm*60)%60):02d}" if tm > 60 else f"{int(tm)}:{int((tm-int(tm))*60):02d}"
            html_trophy += f'<div class="trophy-card"><div class="trophy-title">{t["name"]}</div><div class="trophy-time">{time_str}</div><div class="trophy-date">{best["start_date_local"].strftime("%b %y")}</div></div>'
        else:
            html_trophy += f'<div class="trophy-card" style="opacity:0.6;background-color:#f1f5f9;"><div class="trophy-title">{t["name"]}</div><div class="trophy-time" style="color:#94a3b8;">--:--</div></div>'
    st.markdown(html_trophy + '</div>', unsafe_allow_html=True)
    
    # --- NEW: RECENT RUNS TABLE ---
    st.divider()
    st.markdown("### 🏃 Recent Runs (Last 30 Days)")
    
    last_30_runs = df[df['start_date_local'] >= (now - pd.Timedelta(days=30))].copy()
    
    if not last_30_runs.empty:
        table_data = last_30_runs[['start_date_local', 'name', 'distance_km', 'duration_min', 'pace_min_km', 'avg_hr', 'cadence', 'elevation']].copy()
        table_data['Date'] = table_data['start_date_local'].dt.strftime('%b %d')
        table_data['Name'] = table_data['name']
        table_data['Dist (km)'] = table_data['distance_km'].apply(lambda x: f"{x:.2f}")
        table_data['Time'] = table_data['duration_min'].apply(format_duration)
        table_data['Pace'] = table_data['pace_min_km'].apply(format_pace)
        table_data['HR'] = table_data['avg_hr'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "-")
        table_data['Cadence'] = table_data['cadence'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "-")
        table_data['Elev (m)'] = table_data['elevation'].apply(lambda x: f"{int(x)}")
        
        st.dataframe(
            table_data[['Date', 'Name', 'Dist (km)', 'Time', 'Pace', 'HR', 'Cadence', 'Elev (m)']], 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("No runs recorded in the last 30 days.")

# --- TAB 2: RUN DETAILS ---
with tab2:
    st.header("🏃 Run Details")
    df['label'] = df.apply(lambda x: f"{x['start_date_local'].strftime('%b %d')} - {x['name']} ({x['distance_km']:.2f} km)", axis=1)
    sel_run = st.selectbox("Select a Run:", df['label'].tolist())
    run = df[df['label'] == sel_run].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{run['distance_km']:.2f} km")
    c2.metric("Time", format_duration(run['duration_min']))
    c3.metric("Pace", f"{format_pace(run['pace_min_km'])} /km")
    c4.metric("Elev", f"{int(run['elevation'])} m")
    
    st.divider()
    c_map, c_hr = st.columns([1, 1])
    with c_map:
        st.subheader("🗺️ Route")
        if run['map_polyline']:
            md = decode_map(run['map_polyline'])
            if not md.empty:
                st.pydeck_chart(pdk.Deck(layers=[pdk.Layer(type="PathLayer", data=[{"path": md[['lon', 'lat']].values.tolist()}], get_path="path", get_color=[234, 88, 12], width_min_pixels=3)], initial_view_state=pdk.ViewState(latitude=md['lat'].mean(), longitude=md['lon'].mean(), zoom=13), map_style="road"))
            else: st.warning("No Map Data")
        else: st.info("No GPS Data")
    with c_hr:
        st.subheader("❤️ Intensity")
        if pd.notnull(run['avg_hr']):
            fig = go.Figure(go.Indicator(mode="gauge+number", value=run['avg_hr'], title={'text': "Avg Heart Rate"}, gauge={'axis': {'range': [None, 180]}, 'bar': {'color': "#334155"}, 'steps': [{'range': [0, 100], 'color': "#cbd5e1"}, {'range': [100, 130], 'color': "#60a5fa"}, {'range': [130, 150], 'color': "#4ade80"}, {'range': [150, 170], 'color': "#facc15"}, {'range': [170, 180], 'color': "#f87171"}]}))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1e293b"}, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No HR Data")

    st.divider()
    st.subheader("🤖 Coach's Assessment")
    if st.button("🧠 Analyze This Run", type="primary"):
        with st.spinner("Analyzing..."):
            goal_summary = f"{target_name_input} on {target_date_input} (Target: {target_time_input})"
            analysis = generate_run_analysis(run, df, goal_summary)
            st.markdown(analysis)

# --- TAB 3: AI COACH ---
with tab3:
    st.header("🧠 Gemini Coach")
    df['Month'] = df['start_date_local'].dt.to_period('M')
    monthly_stats = df.groupby('Month').agg({'distance_km': 'sum', 'pace_min_km': 'mean', 'id': 'count'}).sort_index(ascending=True)
    yearly_context_str = "YEARLY LOG:\n" + "\n".join([f"- {idx}: {row['distance_km']:.1f}km, {row['id']} runs, {format_pace(row['pace_min_km'])}/km" for idx, row in monthly_stats.iterrows()])
    goal_summary = f"{target_time_input} for {target_name_input} on {target_date_input}"
    
    st.subheader("Quick Analysis")
    col_q1, col_q2, col_q3 = st.columns(3)
    prompt_trigger = None
    if col_q1.button("📈 Trends"): prompt_trigger = "How is my training trending compared to previous months?"
    if col_q2.button("🔮 Outlook"): prompt_trigger = f"Outlook for {target_name_input} on {target_date_input}?"
    if col_q3.button("🛡️ Advice"): prompt_trigger = "Advice based on recent consistency?"

    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": f"I have analyzed your year. Ask me anything!"}]
    if prompt_trigger: st.session_state.messages.append({"role": "user", "content": prompt_trigger})
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    user_input = st.chat_input("Ask your coach...")
    final_prompt = prompt_trigger if prompt_trigger else user_input
    
    if final_prompt:
        if not prompt_trigger: st.session_state.messages.append({"role": "user", "content": final_prompt}); st.chat_message("user").markdown(final_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = get_gemini_client()
                try:
                    response = client.models.generate_content(model="gemini-flash-latest", contents=f"Coach. GOAL: {goal_summary}\nDATA: {yearly_context_str}\nQ: {final_prompt}")
                    reply = response.text
                except Exception as e: reply = f"Error: {e}"
                st.markdown(reply); st.session_state.messages.append({"role": "assistant", "content": reply})

# --- TAB 4: TRENDS ---
with tab4:
    st.header("📈 Trends")
    weekly = df.groupby(df['start_date_local'].dt.to_period('W').apply(lambda r: r.start_time))['distance_km'].sum().reset_index()
    fig = px.bar(weekly, x='start_date_local', y='distance_km', title="Weekly Volume")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b")
    st.plotly_chart(fig, use_container_width=True)
