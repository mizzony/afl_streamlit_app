import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="AFL Fanclubs", layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #f5f9ff;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        color: #444;
        background-color: #e1ecf4;
        padding: 10px;
        margin: 2px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d4e5f4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976d2;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏉 AFL Fanclubs: Match Predictor & Player Stats Explorer")

# ------------------ MATCH PREDICTOR ------------------
st.header("🔮 AFL Match Winner Predictor")
st.markdown("<small><i>This is the testing version, model hasn't been refined yet.</i></small>", unsafe_allow_html=True)

@st.cache_data
def load_match_data():
    data = pd.read_csv('https://raw.githubusercontent.com/mizzony/AFL/refs/heads/main/afl_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Attendance'] = data['Attendance'].str.replace(',', '').astype(float)
    data['Rainfall'] = data['Rainfall'].fillna(data['Rainfall'].median())
    data = data[(data['HomeTeamScore'] >= 0) & (data['AwayTeamScore'] >= 0)]
    six_months_ago = data['Date'].max() - pd.Timedelta(days=180)
    data_last_6_months = data[data['Date'] >= six_months_ago]
    home_avg = data_last_6_months.groupby('HomeTeam')['HomeTeamScore'].mean()
    away_avg = data_last_6_months.groupby('AwayTeam')['AwayTeamScore'].mean()
    data['HomeTeam_PastAvgPoints'] = data['HomeTeam'].map(home_avg).fillna(0)
    data['AwayTeam_PastAvgPoints'] = data['AwayTeam'].map(away_avg).fillna(0)
    return data, home_avg, away_avg

data, home_team_avg_points, away_team_avg_points = load_match_data()

X = data[['HomeTeam', 'Year', 'Rainfall', 'Venue', 'HomeTeam_PastAvgPoints', 'AwayTeam', 'AwayTeam_PastAvgPoints']]
y = data['Win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoders = {}
for col in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42, n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

st.sidebar.header("⚙️ Match Details")
home_team = st.sidebar.selectbox("Home Team", sorted(data['HomeTeam'].unique()))
away_team = st.sidebar.selectbox("Away Team", sorted(data['AwayTeam'].unique()))
venue = st.sidebar.selectbox("Venue", sorted(data['Venue'].unique()))
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
year = st.sidebar.number_input("Year", min_value=2020, max_value=2025, value=2024)

if st.sidebar.button("Predict Match Outcome"):
    if home_team not in home_team_avg_points.index or away_team not in away_team_avg_points.index:
        st.warning("Invalid team selection.")
    else:
        input_data = pd.DataFrame({
            'HomeTeam': [label_encoders['HomeTeam'].transform([home_team])[0]],
            'Year': [year],
            'Rainfall': [rainfall],
            'Venue': [label_encoders['Venue'].transform([venue])[0]],
            'HomeTeam_PastAvgPoints': [home_team_avg_points.get(home_team, 0)],
            'AwayTeam': [label_encoders['AwayTeam'].transform([away_team])[0]],
            'AwayTeam_PastAvgPoints': [away_team_avg_points.get(away_team, 0)],
        })
        pred = model.predict(input_data)
        result = "🏡 Home Team Wins!" if pred[0] == 1 else "🚶‍♂️ Away Team Wins!"
        st.success(f"### ✅ Prediction: {result}", icon="🎯")
        st.balloons()
# ------------------ PLAYER STATS SECTION ------------------
st.header("📊 Player Stats Explorer (2000–2024)")

@st.cache_data
def load_player_data():
    df = pd.read_csv("https://drive.google.com/uc?export=download&id=14kyzN0g_3RxFAvK3X1mwqsJ9DeZPYCzu")
    df.columns = df.columns.str.strip()
    return df

df = load_player_data()
selected_season = st.selectbox('Select Season', sorted(df['Season'].unique(), reverse=True))
df_season = df[df['Season'] == selected_season]
teams = sorted(df_season['Team'].dropna().unique())
selected_team = st.multiselect('Select Team(s)', teams, teams)
df_selected = df_season[df_season['Team'].isin(selected_team)]

# Tabs for player insights
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Stats Table",
    "🏆 Top Players",
    "🎬 Improvement Chart",
    "📈 Team Averages",
    "🔄 Compare Two Players",
    "⭐ Player Spotlight"
])

with tab1:
    st.subheader(f"Player Stats - {selected_season}")
    st.dataframe(df_selected, use_container_width=True)

with tab2:
    st.subheader("Top 10 by Disposals")
    top10_disp = df_selected.sort_values(by='Disposals', ascending=False).head(10)
    st.bar_chart(top10_disp.set_index('Player')['Disposals'])
    st.subheader("Top 10 by Goals")
    top10_goals = df_selected.sort_values(by='Goals', ascending=False).head(10)
    st.bar_chart(top10_goals.set_index('Player')['Goals'])

with tab3:
    st.subheader("Animated Player Improvement")
    player_choices = st.multiselect("Select players", df['Player'].unique(), default=["Dustin Martin", "Patrick Cripps"])
    df_follow = df[df['Player'].isin(player_choices)]
    fig = px.line(df_follow, x="Season", y="Disposals", color="Player", markers=True, title="Player Disposals Over Time")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("📈 Average Stats Per Team")
    metric = st.selectbox("Select metric to compare:", ['Disposals', 'Goals', 'Kicks', 'Tackles'])
    team_avg = df_selected.groupby('Team')[metric].mean().sort_values(ascending=False)
    fig = px.bar(team_avg, x=team_avg.index, y=metric, title=f"Average {metric} by Team")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("🔄 Compare Two Players")
    players = df_selected['Player'].unique()
    p1 = st.selectbox("Select Player 1:", players)
    p2 = st.selectbox("Select Player 2:", players, index=1 if len(players) > 1 else 0)
    stats = ['Goals', 'Disposals', 'Kicks', 'Marks', 'Tackles']
    if p1 and p2 and p1 != p2:
        df_p1 = df_selected[df_selected['Player'] == p1][stats].mean()
        df_p2 = df_selected[df_selected['Player'] == p2][stats].mean()
        df_compare = pd.DataFrame({'Player 1': df_p1, 'Player 2': df_p2}, index=stats)
        st.bar_chart(df_compare)

with tab6:
    st.subheader("⭐ Player Spotlight")
    spotlight_player = st.selectbox("Select a player to spotlight:", df_selected['Player'].unique())
    player_info = df_selected[df_selected['Player'] == spotlight_player]
    if not player_info.empty:
        team = player_info['Team'].values[0]
        team_logos = {
            'Adelaide': 'https://resources.afl.com.au/photo-resources/2020/03/16/1732f5cf-b3c0-432f-8014-5d8e88d3e7ba/Adelaide.png',
            'Brisbane': 'https://resources.afl.com.au/photo-resources/2020/03/16/ab26b9c5-82cf-4b86-bcfc-d0ff6a5e672d/Brisbane.png',
            'Carlton': 'https://resources.afl.com.au/photo-resources/2020/03/16/b8ebdb6f-4ac7-4c38-b5c6-cfd67f5f4895/Carlton.png',
            'Collingwood': 'https://resources.afl.com.au/photo-resources/2020/03/16/4dfbba6e-03f6-4d2e-8160-f073cfcd4902/Collingwood.png',
            'Essendon': 'https://resources.afl.com.au/photo-resources/2020/03/16/c94ec419-82a3-4ec9-bc36-0d1e8bb12526/Essendon.png',
            'Fremantle': 'https://resources.afl.com.au/photo-resources/2020/03/16/4ff55c08-13c4-4d30-9b20-658b9c7b63f6/Fremantle.png',
            'Geelong': 'https://resources.afl.com.au/photo-resources/2020/03/16/13ea794d-d949-4e5f-bcb8-9bd0b4032ce1/Geelong.png',
            'Gold Coast': 'https://resources.afl.com.au/photo-resources/2020/03/16/0d5d295f-c676-4f25-91f3-ef5f74818d0c/GoldCoast.png',
            'GWS': 'https://resources.afl.com.au/photo-resources/2020/03/16/3b7c3f51-857d-437e-8609-1b0adf7ee804/GWS.png',
            'Hawthorn': 'https://resources.afl.com.au/photo-resources/2020/03/16/97ff6f6a-79e3-45ab-9f08-140ed515e3c0/Hawthorn.png',
            'Melbourne': 'https://resources.afl.com.au/photo-resources/2020/03/16/9c85b6b2-5aa1-4c14-8206-f6c934edaca2/Melbourne.png',
            'North Melbourne': 'https://resources.afl.com.au/photo-resources/2020/03/16/23a7be7b-7d6b-45bb-b7f5-3f2ea05ed4d0/NorthMelbourne.png',
            'Port Adelaide': 'https://resources.afl.com.au/photo-resources/2020/03/16/c7652609-bbff-4d08-b06f-35ef2e72c53c/PortAdelaide.png',
            'Richmond': 'https://resources.afl.com.au/photo-resources/2020/03/16/16b5e5dc-5d5e-4df2-8f15-ecf54a28df60/Richmond.png',
            'St Kilda': 'https://resources.afl.com.au/photo-resources/2020/03/16/396361b5-5adf-4e05-9468-3b515a0873ea/StKilda.png',
            'Sydney': 'https://resources.afl.com.au/photo-resources/2020/03/16/51a4da64-6475-4359-8897-3a5db9e13c5c/Sydney.png',
            'West Coast': 'https://resources.afl.com.au/photo-resources/2020/03/16/60015a08-63c4-4635-8b76-5ad9fa01e9af/WestCoast.png',
            'Western Bulldogs': 'https://resources.afl.com.au/photo-resources/2020/03/16/2232b5d9-f9fa-4781-83c7-9ee4c88b64c8/WesternBulldogs.png'
        }
        logo_url = team_logos.get(team, None)
        cols = st.columns([1, 2])
        if logo_url:
            cols[0].image(logo_url, width=100)
        cols[1].markdown(f"### {spotlight_player} - {team} ({selected_season})")
        cols[1].metric("Goals", player_info['Goals'].values[0])
        cols[1].metric("Disposals", player_info['Disposals'].values[0])
        cols[1].metric("Kicks", player_info['Kicks'].values[0])
        cols[1].metric("Marks", player_info['Marks'].values[0])
        cols[1].metric("Tackles", player_info['Tackles'].values[0])

        # Export player info to downloadable CSV
        export_csv = player_info.to_csv(index=False)
        b64_csv = base64.b64encode(export_csv.encode()).decode()
        download_link = f'<a href="data:file/csv;base64,{b64_csv}" download="{spotlight_player.replace(" ", "_")}_summary.csv">📥 Download {spotlight_player}\'s Summary</a>'
        st.markdown(download_link, unsafe_allow_html=True)

        # Shareable link placeholder
        st.markdown("<small>To share this player, send your friend to this app and search for their name in the Player Spotlight tab.</small>", unsafe_allow_html=True)


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_afl_player_stats.csv">Download CSV</a>'
    return href

st.markdown(filedownload(df_selected), unsafe_allow_html=True)
