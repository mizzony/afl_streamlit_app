# ------------------ INSTALL (CLI) ------------------
# pip install streamlit pandas numpy xgboost scikit-learn plotly

# ------------------ IMPORTS ------------------
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# ------------------ PAGE CONFIG & WHITE-THEME + TAB BAR STYLING ------------------
st.set_page_config(page_title="AFL Fanclubs", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
/* White theme basics */
body, .stApp {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-family: 'Segoe UI', sans-serif;
    padding: 1rem;
}
h1, h2, h3, h4 {
    color: #003366 !important;
}

/* Buttons & metrics */
.stMetric {
    background-color: #f0f2f6 !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}
.stButton>button {
    background-color: #007acc !important;
    color: #ffffff !important;
    font-weight: bold !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
}
.stButton>button:hover {
    background-color: #005fa3 !important;
}

/* Full-width blue tab bar */
.stTabs > div:first-of-type {
    background-color: #007acc !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
    margin-bottom: 1rem !important;
}
/* Individual tabs */
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #ffffff !important;
    font-weight: bold !important;
    margin: 0 0.5rem !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #005fa3 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #005fa3 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TEAM LOGOS ------------------
team_logos = {
    'Adelaide': 'https://resources.afl.com.au/photo-resources/2020/03/16/1732f5cf-b3c0-432f-8014-5d8e88d3e7ba/Adelaide.png',
    'Brisbane': 'https://resources.afl.com.au/photo-resources/2020/03/16/ab26b9c5-82cf-4b86-bcfc-d0ff6a5e672d/Brisbane.png',
    'Carlton': 'https://resources.afl.com.au/photo-resources/2020/03/16/b8ebdb6f-4ac7-4c38-b5c6-cfd67f5f4895/Carlton.png',
    'Collingwood': 'https://resources.afl.com.au/photo-resources/2020/03/16/4dfbba6e-03f6-4d2e-8160-f073cfcd4902/Collingwood.png',
    'Essendon': 'https://resources.afl.com.au/photo-resources/2020/03/16/c94ec419-82a3-4ec9-bc36-0d1e8bb12526/Essendon.png',
    'Fremantle': 'https://resources.afl.com.au/photo-resources/2020/03/16/4ff55c08-13c4-4d30-9b20-658b9c7b63f6/Fremantle.png',
    'Geelong': 'https://resources.afl.com.au/photo-resources/2020/03/16/13ea794d-d949-4e5f-bcb8-9bd0b4032ce1/Geelong.png',
    'Gold Coast': 'https://resources.afl.com.au/photo-resources/2020/03/16/0d5d295f-c676-4f25-91f3-ef5f74818d0c/GoldCoast.png',
    'GWS Giants': 'https://resources.afl.com.au/photo-resources/2020/03/16/3b7c3f51-857d-437e-8609-1b0adf7ee804/GWS.png',
    'Hawthorn': 'https://resources.afl.com.au/photo-resources/2020/03/16/97ff6f6a-79e3-45ab-9f08-140ed515e3c0/Hawthorn.png',
    'Melbourne': 'https://resources.afl.com.au/photo-resources/2020/03/16/9c85b6b2-5aa1-4c14-8206-f6c934edaca2/Melbourne.png',
    'North Melbourne': 'https://resources.afl.com.au/photo-resources/2020/03/16/23a7be7b-7d6b-45bb-b7f5-3f2ea05ed4d0/NorthMelbourne.png',
    'Port Adelaide': 'https://resources.afl.com.au/photo-resources/2020/03/16/c7652609-bbff-4d08-b06f-35ef2e72c53c/PortAdelaide.png',
    'Richmond': 'https://resources.afl.com.au/photo-resources/2020/03/16/16b5e5dc-5d5e-4df2-8f15-ecf54a28df60/Richmond.png',
    'St Kilda': 'https://resources.afl.com.au/photo-resources/2020/03/16/396361b5-5adf-4e05-9468-3b515a0873ea/StKilda.png',
    'Sydney Swans': 'https://resources.afl.com.au/photo-resources/2020/03/16/51a4da64-6475-4359-8897-3a5db9e13c5c/Sydney.png',
    'West Coast': 'https://resources.afl.com.au/photo-resources/2020/03/16/60015a08-63c4-4635-8b76-5ad9fa01e9af/WestCoast.png',
    'Western Bulldogs': 'https://resources.afl.com.au/photo-resources/2020/03/16/2232b5d9-f9fa-4781-83c7-9ee4c88b64c8/WesternBulldogs.png'
}

# ------------------ DATA LOAD ------------------
@st.cache_data
def load_match_data():
    df = pd.read_csv('https://raw.githubusercontent.com/mizzony/AFL/refs/heads/main/afl_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Attendance'] = df['Attendance'].str.replace(',', '').astype(float)
    df['Rainfall'] = df['Rainfall'].fillna(df['Rainfall'].median())
    df = df[(df['HomeTeamScore'] >= 0) & (df['AwayTeamScore'] >= 0)]
    recent = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=180)]
    home_avg = recent.groupby('HomeTeam')['HomeTeamScore'].mean()
    away_avg = recent.groupby('AwayTeam')['AwayTeamScore'].mean()
    df['HomeTeam_PastAvgPoints'] = df['HomeTeam'].map(home_avg).fillna(0)
    df['AwayTeam_PastAvgPoints'] = df['AwayTeam'].map(away_avg).fillna(0)
    return df, home_avg, away_avg

@st.cache_data
def load_player_data():
    pdf = pd.read_csv('https://raw.githubusercontent.com/mizzony/afl_streamlit_app/refs/heads/main/player_stats_last5.csv')
    pdf.columns = pdf.columns.str.strip()
    return pdf

data, home_team_avg_points, away_team_avg_points = load_match_data()
player_df = load_player_data()

# ------------------ CLEANING ------------------
def clean_label(value, known_list):
    key = re.sub(r"[^A-Za-z0-9]", "", value).lower()
    for item in known_list:
        target = re.sub(r"[^A-Za-z0-9]", "", item).lower()
        # match if either is substring of the other
        if key in target or target in key:
            return item
    return value


# ------------------ MODEL TRAINING ------------------
X = data[['HomeTeam','Year','Rainfall','Venue','HomeTeam_PastAvgPoints','AwayTeam','AwayTeam_PastAvgPoints']]
y = data['Win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
label_encoders = {}
for col in X_train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                          seed=42, n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# ------------------ UPCOMING FIXTURES ------------------
upcoming_raw = pd.DataFrame({
    'Date': pd.to_datetime(['2025-05-08','2025-05-09','2025-05-10','2025-05-10','2025-05-10','2025-05-10']),
    'HomeTeam': ['Fremantle','St Kilda','Melbourne','Essendon','Gold Coast','Port Adelaide'],
    'AwayTeam': ['Collingwood','Carlton','Hawthorn','Sydney Swans','Western Bulldogs','Adelaide'],
    'Venue': ['Perth Stadium','Docklands','MCG','Docklands','Carrara','Adelaide Oval']
})
venue_list = data['Venue'].unique().tolist()
team_list = list(set(data['HomeTeam']).union(data['AwayTeam']))
upcoming = upcoming_raw.copy()
upcoming['DispHome'] = upcoming['HomeTeam']; upcoming['DispAway'] = upcoming['AwayTeam']
upcoming['Venue'] = upcoming['Venue'].apply(lambda v: clean_label(v, venue_list))
upcoming['HomeTeam'] = upcoming['DispHome'].apply(lambda t: clean_label(t, team_list))
upcoming['AwayTeam'] = upcoming['DispAway'].apply(lambda t: clean_label(t, team_list))
if 'bets' not in st.session_state:
    st.session_state.bets = []

# ------------------ ODDS FUNCTION ------------------
def calculate_odds_with_model(model, enc, home_avg, away_avg, dfm, overround=1.05):
    out = []
    for _, r in dfm.iterrows():
        try:
            inp = pd.DataFrame({
                'HomeTeam':[enc['HomeTeam'].transform([r['HomeTeam']])[0]],
                'Year':[r['Date'].year],'Rainfall':[0.0],
                'Venue':[enc['Venue'].transform([r['Venue']])[0]],
                'HomeTeam_PastAvgPoints':[home_avg.get(r['HomeTeam'],0)],
                'AwayTeam':[enc['AwayTeam'].transform([r['AwayTeam']])[0]],
                'AwayTeam_PastAvgPoints':[away_avg.get(r['AwayTeam'],0)]
            })
            p = model.predict_proba(inp)[0][1]
            p = np.clip(p,0.1,0.9)
            q = 1-p
            ah, aq = p*overround, q*overround
            p_adj, q_adj = ah/(ah+aq), aq/(ah+aq)
            ho, ao = round(min(max(1.01,1/p_adj),5),2), round(min(max(1.01,1/q_adj),5),2)
            out.append({
                'Match':f"{r['DispHome']} vs {r['DispAway']}",
                'Start Time':r['Date'].strftime('%a %d %b'),
                'Home Odds':ho,'Away Odds':ao,
                'Home Imp':f"{p_adj*100:.2f}%",'Away Imp':f"{q_adj*100:.2f}%"
            })
        except Exception as e:
            st.warning(f"Error {r['DispHome']} vs {r['DispAway']}: {e}")
    return out

odds_list = calculate_odds_with_model(
    model, label_encoders, home_team_avg_points,
    away_team_avg_points, upcoming
)

# ------------------ TOP-LEVEL TABS ------------------
st.title("🏉 AFL Fanclubs: Match Predictor & Betting Odds Simulator")

st.markdown("""
<div style='background-color:#d32f2f; color:#ffffff; padding:15px; border-radius:8px;
            text-align:center; font-size:1.25rem; font-weight:bold; margin-bottom:20px;'>
⚠️ Chances are you're about to lose. Think. Is this a bet you really want to place? ⚠️
</div>
""", unsafe_allow_html=True)
tab_predictor, tab_betting, tab_stats = st.tabs(["🔮 Predictor","📅 Betting","📊 Player Stats"])

with tab_predictor:
    st.header("🔮 Predictor")
    h = st.selectbox("Home", sorted(data['HomeTeam'].unique()))
    a = st.selectbox("Away", sorted(data['AwayTeam'].unique()), index=1)
    v = st.selectbox("Venue", sorted(data['Venue'].unique()))
    r = st.slider("Rain (mm)",0.0,50.0,0.0)
    y = st.number_input("Year",2020,2025,2024)
    if st.button("Predict"):
        dfp = pd.DataFrame({
            'HomeTeam':[label_encoders['HomeTeam'].transform([h])[0]],
            'Year':[y],'Rainfall':[r],
            'Venue':[label_encoders['Venue'].transform([v])[0]],
            'HomeTeam_PastAvgPoints':[home_team_avg_points.get(h,0)],
            'AwayTeam':[label_encoders['AwayTeam'].transform([a])[0]],
            'AwayTeam_PastAvgPoints':[away_team_avg_points.get(a,0)]
        })
        res = model.predict(dfp)[0]
        st.success("🏡 Home Wins!" if res==1 else "🚶‍♂️ Away Wins!")

with tab_betting:
    st.header("📅 Betting")
    for i,m in enumerate(odds_list):
        home,away = m['Match'].split(' vs ')
        c1,c2 = st.columns(2)
        with c1:
            st.metric(f"{home} Odds",f"@ {m['Home Odds']:.2f}",delta=m['Home Imp'])
            if st.button(f"Bet {home}",key=f"bh{i}"):
                st.session_state.bets.append({'Match':m['Match'],'Team':home,'Odds':m['Home Odds'],'Amt':50,'Ret':round(50*m['Home Odds'],2)})
        with c2:
            st.metric(f"{away} Odds",f"@ {m['Away Odds']:.2f}",delta=m['Away Imp'])
            if st.button(f"Bet {away}",key=f"ba{i}"):
                st.session_state.bets.append({'Match':m['Match'],'Team':away,'Odds':m['Away Odds'],'Amt':50,'Ret':round(50*m['Away Odds'],2)})
    if st.session_state.bets:
        st.subheader("Bet History")
        dfb = pd.DataFrame(st.session_state.bets)
        dfb['Res'] = np.where(dfb['Odds']<2,'Win','Loss')
        dfb['P/L'] = np.where(dfb['Res']=='Win',dfb['Ret']-dfb['Amt'],-dfb['Amt'])
        st.dataframe(dfb)
        st.metric("Net P/L", f"${dfb['P/L'].sum():.2f}")

with tab_stats:
    st.header("📊 Player Stats")
    seasons = sorted(player_df['Season'].unique(),reverse=True)
    ss = st.selectbox("Season",seasons)
    dfc = player_df[player_df['Season']==ss]
    teams = sorted(dfc['Team'].dropna().unique())
    sel = st.multiselect("Teams",teams,default=teams)
    dff = dfc[dfc['Team'].isin(sel)]
    t1,t2,t3,t4,t5,t6 = st.tabs(["Table","Top10","Chart","Avg","Compare","Spot"])
    with t1: st.dataframe(dff)
    with t2:
        st.bar_chart(dff.sort_values('Disposals',ascending=False).head(10).set_index('Player')['Disposals'])
        st.bar_chart(dff.sort_values('Goals',ascending=False).head(10).set_index('Player')['Goals'])
    with t3:
        st.subheader("🎬 Improvement Chart")

        # Let users pick any players to compare
        players = st.multiselect(
            "Select players to compare their Disposals over time",
            player_df['Player'].unique(),
            default=list(player_df['Player'].unique()[:2])
        )

        if not players:
            st.info("Please select at least one player to display the chart.")
        else:
            # Build a time-series across all seasons for the selected players
            df_chart = player_df[player_df['Player'].isin(players)].copy()
            df_chart['Season'] = df_chart['Season'].astype(str)  # ensure categorical ordering

            fig = px.line(
                df_chart,
                x="Season",
                y="Disposals",
                color="Player",
                markers=True,
                title="Player Disposals Over Seasons"
            )
            fig.update_layout(
                xaxis=dict(type='category', categoryorder='category ascending'),
                yaxis_title="Disposals",
                xaxis_title="Season"
            )

            st.plotly_chart(fig, use_container_width=True)

    with t4:
        met = st.selectbox("Metric",['Disposals','Goals','Kicks','Tackles'])
        st.bar_chart(dff.groupby('Team')[met].mean().sort_values(ascending=False))
    with t5:
        p1 = st.selectbox("P1", dff['Player'].unique())
        p2 = st.selectbox("P2", dff['Player'].unique(), index=1)
        if p1 != p2:
            stats = ['Goals', 'Disposals', 'Kicks', 'Marks', 'Tackles']
            # Compute means only on those numeric columns
            m1 = dff[dff['Player'] == p1][stats].mean()
            m2 = dff[dff['Player'] == p2][stats].mean()
            st.bar_chart(pd.DataFrame({p1: m1, p2: m2}))

    with t6:
        sp = st.selectbox("Spot",dff['Player'].unique())
        inf = dff[dff['Player']==sp].iloc[0]
        lg = team_logos.get(inf['Team'])
        cA,cB = st.columns([1,3])
        if lg: cA.image(lg,width=80)
        cB.markdown(f"### {sp} - {inf['Team']}")
        for s in ['Goals','Disposals','Kicks','Marks','Tackles']:
            cB.metric(s,inf[s])
        csv = dff[dff['Player']==sp].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="{sp}.csv">Download CSV</a>',unsafe_allow_html=True)
