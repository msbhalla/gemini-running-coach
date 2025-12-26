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
from datetime import date, datetime
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Running Coach", page_icon="🏃‍♂️", layout="wide")

# --- LOAD SECRETS ---
load_dotenv()
STRAVA_CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
STRAVA_CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- 💾 SETTINGS MANAGEMENT ---
SETTINGS_FILE = "user_settings.json"

def load_settings():
    """Loads user goals from a JSON file, or returns defaults."""
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
    """Saves user goals to a JSON file."""
    settings = {
        "race_date": str(race_date),
        "race_goal_time": goal_time,
        "race_name": race_name
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

# --- 🎨 LIGHT MODE CSS ---
st.markdown("""
<style>
    /* Trophy Cabinet Grid Layout */
    .trophy-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }
    
    /* Trophy Card - Light Theme */
    .trophy-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #f59e0b; /* Amber */
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        transition: transform 0.2s;
    }
    .trophy-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .trophy-title { 
        font-size: 0.8rem; 
        text-transform: uppercase; 
        color: #64748b; /* Slate-500 */
        font-weight: 600;
        letter-spacing: 0.5px; 
    }
    .trophy-time { 
        font-size: 1.6rem; 
        font-weight: 700; 
        color: #0f172a; /* Slate-900 */
        font-family: monospace; 
        margin: 5px 0;
    }
    .trophy-date { 
        font-size: 0.75rem; 
        color: #94a3b8; 
    }
    
    /* Countdown Box - Light Theme */
    .countdown-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- AUTHENTICATION ---
def get_auth_url():
    return f"http://www.strava.com/oauth/authorize?client_id={STRAVA_CLIENT_ID}&response_type=code&redirect_uri=http://localhost:8501&approval_prompt=force&scope=activity:read_all"

def exchange_token(code):
    response = requests.post(
        'https://www.strava.com/oauth/token',
        data={
            'client_id': STRAVA_CLIENT_ID,
            'client_secret': STRAVA_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code'
        }
    )
    if response.status_code == 200:
        return response.json()['access_token']
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

# --- 🎨 IMAGE ENGINE ---
def load_font(size, variant="Modern"):
    return ImageFont.load_default()

def draw_frosted_glass(img, rect_coords, radius=15):
    box = img.crop(rect_coords)
    box = box.filter(ImageFilter.GaussianBlur(radius))
    white_layer = Image.new("RGBA", box.size, (255, 255, 255, 30))
    box = Image.alpha_composite(box.convert("RGBA"), white_layer)
    return box

def generate_aesthetic_image(upload_file, run_data, config):
    try:
        img = Image.open(upload_file).convert("RGBA")
        target_width = 1080
        ratio = target_width / img.width
        target_height = int(img.height * ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # Simplified drawing logic
        if config['style'] == "Minimal Card":
            draw_frosted_glass(img, (50, height-300, width-50, height-50))
            draw.rectangle([50, height-300, width-50, height-50], outline="white", width=2)
            draw.text((100, height-200), f"{run_data['distance_km']:.2f} km", fill="white", font=load_font(60))
            
        return img.convert("RGB")
    except Exception:
        return None

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

@st.cache_data(ttl=3600)
def fetch_activity_splits(token, activity_id):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {'Authorization': f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get('splits_metric', [])
    return []

# --- 🧠 GEMINI FUNCTIONS ---
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_training_plan(history_summary, goal_date, goal_desc):
    client = get_gemini_client()
    prompt = f"""
    Act as a professional marathon coach.
    Create a specific 1-week training plan (Monday to Sunday) for a runner.
    
    Current Status:
    - Recent Weekly Volume: ~{history_summary['dist_30d']/4:.1f} km/week
    - Average Pace: {format_pace(history_summary['avg_pace_30d'])} /km
    
    Goal:
    - Race Date: {goal_date}
    - Target: {goal_desc}
    
    Output format: Markdown table with columns: Day, Workout Type, Distance, Target Pace, Notes.
    Add a brief motivational quote at the end.
    """
    try:
        response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        return response.text
    except Exception as e:
        return f"Error generating plan: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Goals & Settings")
    
    current_settings = load_settings()
    
    with st.form("goal_form"):
        st.caption("Update your race target here. This will be saved.")
        target_date_input = st.date_input("Race Date", date.fromisoformat(current_settings["race_date"]))
        target_time_input = st.text_input("Target Time (HH:MM:SS)", current_settings["race_goal_time"])
        target_name_input = st.text_input("Event Name", current_settings.get("race_name", "Marathon"))
        
        submitted = st.form_submit_button("💾 Save Goal")
        if submitted:
            save_settings(target_date_input, target_time_input, target_name_input)
            st.success("Goal Updated!")
            st.rerun()

    target_date = target_date_input
    target_time_str = target_time_input
    
    today = date.today()
    days_left = (target_date - today).days
    
    try:
        h, m, s = map(int, target_time_str.split(':'))
        total_min = h*60 + m + s/60
        req_pace = total_min / 42.195
        req_pace_fmt = format_pace(req_pace)
    except:
        req_pace_fmt = "--:--"
        
    st.markdown(f"""
    <div class="countdown-box">
        <div style="font-size: 0.8rem; color: #64748b; font-weight: bold; text-transform: uppercase;">{target_name_input}</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #0284c7;">{days_left} Days</div>
        <div style="margin-top: 10px; font-size: 0.8rem; color: #64748b; font-weight: bold;">REQUIRED PACE</div>
        <div style="font-size: 1.3rem; font-weight: bold; color: #16a34a;">{req_pace_fmt} /km</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    if st.button("🔄 Sync Data Now"):
        st.cache_data.clear()
        st.rerun()
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# --- MAIN APP UI ---
st.title("🏃‍♂️ Gemini Running Coach")

query_params = st.query_params
auth_code = query_params.get("code")

if "access_token" not in st.session_state:
    if auth_code:
        token = exchange_token(auth_code)
        if token:
            st.session_state["access_token"] = token
            st.rerun()
        else:
            st.error("Login failed.")
    else:
        st.info("Please log in.")
        st.link_button("🔗 Connect with Strava", get_auth_url())
        st.stop()

token = st.session_state["access_token"]
with st.spinner("Syncing Strava Activities..."):
    df = fetch_strava_data(token)

if df.empty:
    st.warning("No runs found.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🏃 Run Details", "🧠 AI Coach", "📈 Trends"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.markdown("### ⚡ Performance Summary")
    
    # 1. Updated Time Options (Added 3 Months, 6 Months)
    time_options = {
        "1 Week": 7, 
        "15 Days": 15, 
        "1 Month": 30, 
        "3 Months": 90, 
        "6 Months": 180
    }
    
    selected_timeframe = st.radio("Select Duration:", options=list(time_options.keys()), index=2, horizontal=True, label_visibility="collapsed")
    days = time_options[selected_timeframe]
    
    now = pd.Timestamp.now(tz='UTC')
    runs_curr = df[df['start_date_local'] >= (now - pd.Timedelta(days=days))]
    runs_prev = df[(df['start_date_local'] >= (now - pd.Timedelta(days=days*2))) & (df['start_date_local'] < (now - pd.Timedelta(days=days)))]
    
    def get_mean(dframe, col): return dframe[col].mean() if len(dframe) > 0 and col in dframe.columns else 0

    # 2. Updated Columns (5 columns to include Elev & HR)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Distance
    curr_dist = runs_curr['distance_km'].sum()
    prev_dist = runs_prev['distance_km'].sum()
    c1.metric("Distance", f"{curr_dist:.1f} km", f"{curr_dist - prev_dist:.1f} km")
    
    # Avg Pace
    curr_pace = get_mean(runs_curr, 'pace_min_km')
    prev_pace = get_mean(runs_prev, 'pace_min_km')
    c2.metric("Avg Pace", f"{format_pace(curr_pace)} /km", f"{curr_pace - prev_pace:.2f} m/k", delta_color="inverse")
    
    # Elevation (New)
    curr_elev = runs_curr['elevation'].sum()
    prev_elev = runs_prev['elevation'].sum()
    c3.metric("Elevation", f"{int(curr_elev)} m", f"{int(curr_elev - prev_elev)} m")
    
    # Heart Rate (New)
    curr_hr = get_mean(runs_curr, 'avg_hr')
    prev_hr = get_mean(runs_prev, 'avg_hr')
    c4.metric("Avg HR", f"{int(curr_hr)} bpm", f"{int(curr_hr - prev_hr)} bpm", delta_color="inverse")
    
    # Runs count
    c5.metric("Runs", len(runs_curr), len(runs_curr) - len(runs_prev))

    st.divider()

    st.markdown("### 🏆 Personal Bests (Estimated)")
    
    trophies = [
        {"name": "1 km", "dist": 1.0},
        {"name": "1 Mile", "dist": 1.609},
        {"name": "5 km", "dist": 5.0},
        {"name": "10 km", "dist": 10.0},
        {"name": "Half Marathon", "dist": 21.1},
        {"name": "Marathon", "dist": 42.2},
    ]
    
    html_trophy = '<div class="trophy-grid">'
    
    for t in trophies:
        qualifying_runs = df[df['distance_km'] >= (t['dist'] * 0.98)].copy() 
        if not qualifying_runs.empty:
            best = qualifying_runs.sort_values('pace_min_km').iloc[0]
            time_mins = best['pace_min_km'] * t['dist']
            
            if time_mins > 60:
                h = int(time_mins // 60)
                m = int(time_mins % 60)
                s = int((time_mins*60) % 60)
                time_str = f"{h}:{m:02d}:{s:02d}"
            else:
                m = int(time_mins)
                s = int((time_mins - m) * 60)
                time_str = f"{m}:{s:02d}"
                
            date_str = best['start_date_local'].strftime("%b '%y")
            
            # NOTE: Indentation removed below to fix rendering
            html_trophy += f"""<div class="trophy-card">
<div class="trophy-title">{t['name']}</div>
<div class="trophy-time">{time_str}</div>
<div class="trophy-date">{date_str} • {format_pace(best['pace_min_km'])}/km</div>
</div>"""
        else:
            html_trophy += f"""<div class="trophy-card" style="opacity: 0.6; background-color: #f1f5f9;">
<div class="trophy-title">{t['name']}</div>
<div class="trophy-time" style="color: #94a3b8;">--:--</div>
<div class="trophy-date">Not yet recorded</div>
</div>"""
    
    html_trophy += "</div>"
    st.markdown(html_trophy, unsafe_allow_html=True)
    
    st.divider()
    
    # 3. Monthly Log
    st.markdown("### 📅 Monthly Log")
    df['Month_Year'] = df['start_date_local'].dt.to_period('M')
    monthly = df.groupby('Month_Year').agg({'distance_km': 'sum', 'id': 'count', 'pace_min_km': 'mean'}).sort_values(by='Month_Year', ascending=False).reset_index()
    monthly['Month'] = monthly['Month_Year'].dt.strftime('%B %Y')
    monthly['Pace'] = monthly['pace_min_km'].apply(format_pace)
    st.dataframe(monthly[['Month', 'distance_km', 'id', 'Pace']].rename(columns={'distance_km': 'Dist (km)', 'id': 'Runs', 'Pace': 'Avg Pace'}), hide_index=True, use_container_width=True)

# --- TAB 2: RUN DETAILS ---
with tab2:
    st.header("🏃 Run Details")
    df['label'] = df.apply(lambda x: f"{x['start_date_local'].strftime('%b %d')} - {x['name']} ({x['distance_km']:.2f} km)", axis=1)
    selected_label = st.selectbox("Select a Run:", df['label'].tolist())
    run = df[df['label'] == selected_label].iloc[0]
    
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
            map_data = decode_map(run['map_polyline'])
            if not map_data.empty:
                layer = pdk.Layer(type="PathLayer", data=[{"path": map_data[['lon', 'lat']].values.tolist()}], get_path="path", get_color=[234, 88, 12], width_min_pixels=3)
                view = pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=13)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="road"))
            else: st.warning("No Map Data")
        else: st.info("No GPS Data")
        
    with c_hr:
        st.subheader("❤️ Intensity")
        if pd.notnull(run['avg_hr']):
            max_hr = 176 
            current_hr = run['avg_hr']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_hr,
                title = {'text': "Avg Heart Rate"},
                gauge = {
                    'axis': {'range': [None, max_hr]},
                    'bar': {'color': "#334155"}, 
                    'steps': [
                        {'range': [0, 0.6*max_hr], 'color': "#cbd5e1"}, 
                        {'range': [0.6*max_hr, 0.7*max_hr], 'color': "#60a5fa"}, 
                        {'range': [0.7*max_hr, 0.8*max_hr], 'color': "#4ade80"}, 
                        {'range': [0.8*max_hr, 0.9*max_hr], 'color': "#facc15"}, 
                        {'range': [0.9*max_hr, max_hr], 'color': "#f87171"} 
                    ],
                }
            ))
            fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "#1e293b"}, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            zone = "Z1 (Recovery)"
            if current_hr > 0.9*max_hr: zone = "Z5 (Max Effort)"
            elif current_hr > 0.8*max_hr: zone = "Z4 (Threshold)"
            elif current_hr > 0.7*max_hr: zone = "Z3 (Aerobic)"
            elif current_hr > 0.6*max_hr: zone = "Z2 (Easy)"
            
            st.info(f"Primary Zone: **{zone}** (Based on Max HR {max_hr})")
        else:
            st.warning("No Heart Rate data available for this run.")

    st.divider()
    
    st.subheader("📸 Studio")
    uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        config = {'style': "Minimal Card", 'selected_stats': [], 'position': "Bottom", 'font_size': 100, 'text_color': "#fff", 'font_style': "Modern", 'show_pb': False}
        if st.button("Generate"):
            img = generate_aesthetic_image(uploaded_file, run, config)
            if img: st.image(img)

# --- TAB 3: AI COACH ---
with tab3:
    st.header("🧠 Gemini Coach")
    
    user_settings = load_settings()
    goal_summary = f"{user_settings['race_goal_time']} for {user_settings['race_name']} on {user_settings['race_date']}"
    
    mode = st.radio("Select Mode:", ["💬 Chat with Coach", "📅 Generate Training Plan"], horizontal=True)
    
    now = pd.Timestamp.now(tz='UTC')
    last_30d = df[df['start_date_local'] >= (now - pd.Timedelta(days=30))]
    history_summary = {
        "dist_30d": last_30d['distance_km'].sum(),
        "avg_pace_30d": last_30d['pace_min_km'].mean() if len(last_30d) > 0 else 0
    }

    if mode == "📅 Generate Training Plan":
        st.info(f"Targeting: **{goal_summary}**")
        if st.button("Generate Plan", type="primary"):
            with st.spinner("Drafting your schedule..."):
                plan = generate_training_plan(history_summary, user_settings['race_date'], f"Sub-{user_settings['race_goal_time']} Marathon")
                st.markdown(plan)
                
    elif mode == "💬 Chat with Coach":
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": f"Hi! I see you're training for {user_settings['race_name']} on {user_settings['race_date']}. How can I help?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your coach..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    client = get_gemini_client()
                    context_prompt = f"""
                    You are a running coach.
                    User Context: 
                    - Goal: {goal_summary}
                    - Recent Monthly Distance: {history_summary['dist_30d']:.1f} km
                    - Recent Avg Pace: {format_pace(history_summary['avg_pace_30d'])}/km
                    User Question: {prompt}
                    """
                    try:
                        response = client.models.generate_content(model="gemini-flash-latest", contents=context_prompt)
                        reply = response.text
                    except Exception as e:
                        reply = f"Sorry, I'm having trouble connecting. Error: {e}"
                        
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

# --- TAB 4: TRENDS ---
with tab4:
    st.header("📈 Trends")
    weekly = df.groupby(df['start_date_local'].dt.to_period('W').apply(lambda r: r.start_time))['distance_km'].sum().reset_index()
    fig = px.bar(weekly, x='start_date_local', y='distance_km', title="Weekly Volume")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b")
    st.plotly_chart(fig, use_container_width=True)