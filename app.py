import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Experience Intelligence System",
    page_icon="🧠",
    layout="wide"
)

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 AI-Powered Experience Intelligence System")
st.markdown(
    "### Analytics + AI to predict **Satisfaction, Churn & Engagement**"
)

col1, col2, col3 = st.columns(3)

col1.metric("🚕 Taxi Satisfaction AI", "Live")
col2.metric("🧭 Waze Churn AI", "Live")
col3.metric("📱 TikTok Engagement AI", "Live")

st.divider()

st.header("🚕 Taxi Experience Intelligence")

taxi_model = joblib.load("models/taxi_model.pkl")

distance = st.slider("Trip Distance (km)", 1.0, 50.0, 12.0)
duration = st.slider("Trip Duration (minutes)", 5, 120, 35)
cost_per_km = st.slider("Cost per KM", 5.0, 50.0, 15.0)
peak = st.selectbox("Peak Hour?", ["No", "Yes"])

peak_val = 1 if peak == "Yes" else 0

taxi_input = pd.DataFrame([[
    distance, duration, cost_per_km, peak_val
]], columns=[
    'trip_distance', 'trip_duration_min', 'cost_per_km', 'peak_hour'
])

tip_pred = taxi_model.predict(taxi_input)[0]

st.metric("💡 Predicted Tip Ratio (Satisfaction)", f"{tip_pred:.2f}")

with st.expander("🤖 AI Insight"):
    st.write(
        "Longer trips and higher cost per KM reduce perceived satisfaction."
    )
    st.subheader("📌 Why this satisfaction score?")

reasons = []

if cost_per_km > 18:
    reasons.append("High cost per KM makes the ride feel expensive.")

if duration > 30:
    reasons.append("Long travel duration increases passenger fatigue.")

if peak_val == 1:
    reasons.append("Peak-hour traffic causes delays and frustration.")

if distance > 20:
    reasons.append("Very long distance trips reduce comfort.")

if not reasons:
    reasons.append("Trip conditions look balanced and customer-friendly.")

for r in reasons:
    st.write("•", r)



st.divider()
st.header("🧭 Waze Churn Risk AI")

waze_model = joblib.load("models/waze_model.pkl")
waze_scaler = joblib.load("models/waze_scaler.pkl")

sessions = st.slider("Sessions per Day", 0.0, 10.0, 1.2)
favorite = st.slider("Favorite Navigation Ratio", 0.0, 1.0, 0.4)
inactive = st.slider("Inactive Ratio", 0.0, 1.0, 0.3)
driving_consistency = st.slider("Driving Consistency", 0.0, 1.0, 0.6)
device = st.selectbox("Device", ["Android", "iOS"])

device_android = 1 if device == "Android" else 0

waze_input = pd.DataFrame([[
    sessions, 0.0, favorite, inactive, driving_consistency, device_android
]], columns=[
    'sessions_per_day',
    'km_per_drive',
    'favorite_ratio',
    'inactive_ratio',
    'driving_consistency',
    'device_android'
])

waze_scaled = waze_scaler.transform(waze_input)
churn_prob = waze_model.predict_proba(waze_scaled)[0][1]

st.metric("⚠️ Churn Probability", f"{churn_prob*100:.1f}%")

if churn_prob > 0.6:
    st.error("🚨 High Risk User")
elif churn_prob > 0.3:
    st.warning("⚠️ Medium Risk User")
else:
    st.success("✅ Likely Retained")

with st.expander("🤖 AI Insight"):
    st.write(
        "High inactivity and low favorite usage strongly increase churn risk."
    )
    st.subheader("✅ Suggested Actions")

if tip_pred < 0.15:
    st.write("• Offer dynamic discounts during peak hours")
    st.write("• Optimize routes to reduce duration")
    st.write("• Improve driver communication & comfort")
else:
    st.write("• Current pricing and timing look acceptable")
    st.write("• Maintain service quality")


st.divider()
st.header("📱 TikTok Engagement Intelligence")

tiktok_model = joblib.load("models/tiktok_model.pkl")

views_per_sec = st.slider("Views per Second", 0.0, 500.0, 60.0)
interaction = st.slider("Interaction Intensity", 0.0, 10.0, 1.5)
short = st.checkbox("Short Video (<30s)")
verified = st.checkbox("Verified Creator")
banned = st.checkbox("Author Banned")

tiktok_input = pd.DataFrame([[
    views_per_sec,
    interaction,
    int(short),
    int(verified),
    int(banned)
]], columns=[
    'views_per_second',
    'interaction_intensity',
    'short_video_flag',
    'verified_flag',
    'ban_flag'
])

eng_pred = tiktok_model.predict(tiktok_input)[0]

if eng_pred == 1:
    st.success("🔥 High Engagement Content")
else:
    st.warning("📉 Low Engagement Risk")

with st.expander("🤖 AI Insight"):
    st.write(
        "Short videos with high interaction intensity perform better."
    )
    st.subheader("🔁 What-if Scenario: Optimized Trip")

optimized_input = taxi_input.copy()
optimized_input['cost_per_km'] *= 0.8
optimized_input['trip_duration_min'] *= 0.85

optimized_tip = taxi_model.predict(optimized_input)[0]

st.metric(
    "Optimized Satisfaction Score",
    f"{optimized_tip:.2f}",
    delta=f"{optimized_tip - tip_pred:.2f}"
)
st.markdown("---")
