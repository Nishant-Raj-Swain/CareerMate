import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from auth import (
    init_auth_db, seed_default_admin, authenticate_admin, create_session,
    validate_session, terminate_session, is_rate_limited,
    record_failed_attempt, clear_failed_attempts
)

st.set_page_config(page_title="Secure WhatsApp Admin", layout="wide")

# Initialize database schema and default admin account
init_auth_db()
seed_default_admin()

if "session_token" not in st.session_state:
    st.session_state["session_token"] = None

# Block unauthorized access with Login portal
if not validate_session(st.session_state["session_token"]):
    st.title("🔒 Admin Authentication Portal")
    st.subheader("Sign In")
    
    login_user = st.text_input("Username", key="login_user")
    login_pass = st.text_input("Password", type="password", key="login_pass")
    
    if st.button("Log In"):
        if is_rate_limited():
            st.error("Too many failed login attempts. Account locked for 15 minutes.")
        elif authenticate_admin(login_user, login_pass):
            clear_failed_attempts()
            token = create_session(login_user)
            st.session_state["session_token"] = token
            st.success("Authenticated successfully!")
            st.rerun()
        else:
            record_failed_attempt()
            st.error("Invalid credentials.")
            
    st.stop() # Stops page loading if not logged in

# Protected Analytics Dashboard
st.sidebar.title("🔐 Session Active")
if st.sidebar.button("Log Out"):
    terminate_session(st.session_state["session_token"])
    st.session_state["session_token"] = None
    st.rerun()

st.title("📊 WhatsApp Bot Analytics Dashboard")

DB_NAME = "bot_analytics.db"

def load_data():
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT whatsapp_no, command, timestamp FROM user_logs"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

if df.empty:
    st.warning("No command logs recorded yet.")
    st.stop()

# Dashboard Content
col1, col2, col3 = st.columns(3)
col1.metric("Total Executed Commands", len(df))
col2.metric("Total Unique Users", df['whatsapp_no'].nunique())
col3.metric("Top Command", df['command'].mode()[0] if not df.empty else "N/A")

st.markdown("---")

st.subheader("🌐 Overall Command Usage")
overall_counts = df['command'].value_counts().reset_index()
overall_counts.columns = ['command', 'count']

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=overall_counts, x='count', y='command', palette='Blues_r', ax=ax1)
st.pyplot(fig1)

st.markdown("---")

st.subheader("👤 User-Specific Command Analysis")
selected_user = st.selectbox("Select User Phone Number:", df['whatsapp_no'].unique().tolist())
user_df = df[df['whatsapp_no'] == selected_user]

user_counts = user_df['command'].value_counts().reset_index()
user_counts.columns = ['Command', 'Usage Count']

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(data=user_counts, x='Usage Count', y='Command', palette='Viridis', ax=ax2)
st.pyplot(fig2)

with st.expander("📄 View Full Raw Logs"):
    st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)
