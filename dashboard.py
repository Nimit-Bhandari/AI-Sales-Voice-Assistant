import streamlit as st
import json
import time
import os

st.set_page_config(page_title="AI Voice Assistant", layout="wide")

st.title("🎧 AI Voice Assistant – Live Dashboard")

st.write("This dashboard shows live input from your speech engine and the AI reasoning output.")

placeholder_user = st.empty()
placeholder_ai = st.empty()

JSON_FILE = "live_output.json"

# UI Styling
st.markdown("""
    <style>
        .box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f2f2f2;
            margin-bottom: 15px;
        }
        .title {
            font-weight: 700;
            font-size: 18px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)


while True:
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r") as f:
                data = json.load(f)

            # ---------------------------
            # LEFT SIDE: USER VOICE TEXT
            # ---------------------------
            user_html = f"""
            <div class='box'>
                <div class='title'>🧑 User Said:</div>
                <div style='font-size: 20px; color:#333;'>{data['text']}</div>
            </div>
            """
            placeholder_user.markdown(user_html, unsafe_allow_html=True)

            # ---------------------------
            # RIGHT SIDE: AI REASONING
            # ---------------------------
            ai = data["ai_suggestions"]

            ai_html = f"""
            <div class='box'>
                <div class='title'>🤖 AI Understanding</div>
                <b>Sentiment:</b> {data['sentiment']}<br>
                <b>Intent:</b> {data['intent']}<br>
                <b>Category:</b> {data['category']}<br>
                <b>Emotion:</b> {data['emotion']}<br>
                <b>Entities:</b> {data['entities']}<br>
            </div>

            <div class='box'>
                <div class='title'>💬 AI Suggestions</div>
                <b>Next Question:</b> {ai['next_question']}<br><br>
                <b>Objection Handling:</b> {ai['objection_response']}<br><br>
                <b>Recommendation:</b> {ai['recommendation']}
            </div>
            """

            placeholder_ai.markdown(ai_html, unsafe_allow_html=True)

        except:
            pass

    time.sleep(0.4)
