import os
import streamlit as st
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Secure WhatsApp Admin", layout="wide")

FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")

if "session_token" not in st.session_state:
    st.session_state["session_token"] = None

def remote_login(username, password):
    try:
        response = httpx.post(
            f"{FASTAPI_URL}/api/auth/login",
            json={"username": username, "password": password},
            timeout=10.0
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("Too many failed attempts. Account locked for 15 minutes.")
        else:
            st.error("Invalid credentials.")
    except Exception as e:
        st.error(f"Unable to connect to backend server: {e}")
    return None

def remote_logout(token):
    try:
        httpx.post(
            f"{FASTAPI_URL}/api/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0
        )
    except Exception:
        pass

def fetch_analytics_logs(token):
    try:
        response = httpx.get(
            f"{FASTAPI_URL}/api/analytics/logs",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        elif response.status_code == 401:
            st.warning("Session expired. Please log in again.")
            st.session_state["session_token"] = None
            st.rerun()
    except Exception as e:
        st.error(f"Failed to fetch analytics from backend API: {e}")
    return pd.DataFrame()

# --- Authentication Portal ---
if not st.session_state["session_token"]:
    st.title("🔒 Admin Authentication Portal")
    st.subheader("Sign In to Remote Dashboard")
    
    with st.form("login_form"):
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        submit_button = st.form_submit_button("Log In")
        
        if submit_button:
            if login_user and login_pass:
                result = remote_login(login_user, login_pass)
                if result and "token" in result:
                    st.session_state["session_token"] = result["token"]
                    st.success("Authenticated successfully!")
                    st.rerun()
            else:
                st.warning("Please enter both username and password.")
                
    st.stop()

# --- Protected Analytics Dashboard ---
st.sidebar.title("🔐 Session Active")
if st.sidebar.button("Log Out"):
    remote_logout(st.session_state["session_token"])
    st.session_state["session_token"] = None
    st.rerun()

st.title("📊 WhatsApp Bot Analytics Dashboard")

df = fetch_analytics_logs(st.session_state["session_token"])

if df.empty:
    st.warning("No command logs recorded yet or unable to fetch logs from backend.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Total Executed Commands", len(df))
col2.metric("Total Unique Users", df['whatsapp_no'].nunique() if 'whatsapp_no' in df else 0)
col3.metric("Top Command", df['command'].mode()[0] if 'command' in df and not df.empty else "N/A")

st.markdown("---")

st.subheader("🌐 Overall Command Usage")
if 'command' in df:
    overall_counts = df['command'].value_counts().reset_index()
    overall_counts.columns = ['command', 'count']

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=overall_counts, x='count', y='command', palette='Blues_r', ax=ax1)
    st.pyplot(fig1)

st.markdown("---")

st.subheader("👤 User-Specific Command Analysis")
if 'whatsapp_no' in df:
    user_list = df['whatsapp_no'].unique().tolist()
    selected_user = st.selectbox("Select User Phone Number:", user_list)
    
    user_df = df[df['whatsapp_no'] == selected_user]
    user_counts = user_df['command'].value_counts().reset_index()
    user_counts.columns = ['Command', 'Usage Count']

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=user_counts, x='Usage Count', y='Command', palette='Viridis', ax=ax2)
    st.pyplot(fig2)

with st.expander("📄 View Full Raw Logs"):
    st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)
