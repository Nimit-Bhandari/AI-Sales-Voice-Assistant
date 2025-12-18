import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from vosk import Model, KaldiRecognizer
import json
import re
from textblob import TextBlob

# ------------------------------------------------------
# LOAD VOSK MODEL (offline speech-to-text)
# ------------------------------------------------------
MODEL_PATH = "vosk-model-en-us-0.22-lgraph"
model = Model(MODEL_PATH)

# ------------------------------------------------------
# YOUR DATASET (category + emotion detection)
# ------------------------------------------------------
conversation_patterns = [
    {"category": "electronics_purchase", "emotion": "friendly_curious",
     "keywords": ["budget", "headset", "bluetooth", "order", "battery"]},

    {"category": "internet_plan_upgrade", "emotion": "polite_interested",
     "keywords": ["upgrade", "internet", "plan", "router", "mbps"]},

    {"category": "product_return", "emotion": "frustrated_relived",
     "keywords": ["wrong", "received", "return", "replacement", "pickup"]},

    {"category": "travel_booking", "emotion": "excited_persuasive",
     "keywords": ["trip", "package", "goa", "weekend", "emi"]},

    {"category": "bank_query", "emotion": "calm_reassuring",
     "keywords": ["account", "debit card", "dispatched", "shipped"]},

    {"category": "mobile_recharge", "emotion": "neutral_satisfied",
     "keywords": ["recharge", "299", "data", "gb", "plan"]},

    {"category": "software_subscription", "emotion": "curious_convinced",
     "keywords": ["renew", "discount", "antivirus", "offer"]},

    {"category": "restaurant_reservation", "emotion": "polite_pleasant",
     "keywords": ["book table", "reservation", "8 pm", "indoor", "outdoor"]},

    {"category": "gadget_repair", "emotion": "concerned_relieved",
     "keywords": ["repair", "laptop", "won't turn on", "diagnosis"]},

    {"category": "clothing_sale", "emotion": "cheerful_excited",
     "keywords": ["sale", "discount", "winter jackets", "offer"]},
]

# ------------------------------------------------------
# CATEGORY + EMOTION
# ------------------------------------------------------
def classify_text_custom(text):
    t = text.lower()

    for item in conversation_patterns:
        for kw in item["keywords"]:
            if kw in t:
                return item["category"], item["emotion"]

    return "unknown", "neutral"

# ------------------------------------------------------
# ENTITY EXTRACTION
# ------------------------------------------------------
def extract_entities(text):
    entities = []

    money = re.findall(r"(₹\s?\d+|\d+\s?rupees|\d+\s?rs)", text.lower())
    for m in money:
        entities.append((m, "MONEY"))

    order_ids = re.findall(r"#\d+", text)
    for oid in order_ids:
        entities.append((oid, "ORDER_ID"))

    dates = re.findall(r"\b(\d{1,2}\s?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))\b", text.lower())
    for d in dates:
        entities.append((d[0], "DATE"))

    times = re.findall(r"\b\d{1,2}\s?(am|pm)\b", text.lower())
    for t in times:
        entities.append((t, "TIME"))

    percent = re.findall(r"\b\d+%\b", text)
    for p in percent:
        entities.append((p, "PERCENT"))

    locations = ["goa", "delhi", "mumbai", "bangalore", "india"]
    for loc in locations:
        if loc in text.lower():
            entities.append((loc.capitalize(), "LOCATION"))

    brands = ["hp", "dell", "lenovo", "asus", "boat", "sony"]
    for brand in brands:
        if brand in text.lower():
            entities.append((brand.capitalize(), "BRAND"))

    return entities

# ------------------------------------------------------
# SENTIMENT + INTENT
# ------------------------------------------------------
def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    t = text.lower()
    if "book" in t:
        intent = "booking"
    elif "order" in t:
        intent = "purchase"
    elif "upgrade" in t:
        intent = "upgrade_request"
    elif "return" in t or "wrong" in t:
        intent = "return_request"
    else:
        intent = "general_statement"

    entities = extract_entities(text)
    return sentiment, intent, entities

# ------------------------------------------------------
# RULE-BASED RESPONSE GENERATOR
# ------------------------------------------------------
def generate_AI(intent, sentiment, entities, category):
    suggested = f"May I know more details regarding your {intent}?"
    objection = f"I understand your concern. I'll assist you with your {intent}."
    recommendation = f"Since your query is related to '{category}', you may explore related offers or solutions."

    return suggested, objection, recommendation

# ------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------
st.title("🎤 Offline Voice AI Assistant (VOSK + Streamlit)")
st.write("Speak into the microphone and the system will analyze your intent, sentiment, category, entities and generate responses.")

# ------------------------------------------------------
# AUDIO PROCESSOR CLASS
# ------------------------------------------------------
class AudioProcessor:
    def __init__(self):
        self.recognizer = KaldiRecognizer(model, 16000)
        self.text_output = ""

    def recv(self, frame):
        # Convert frame to raw audio bytes and feed to VOSK
        audio = frame.to_ndarray().tobytes()

        if self.recognizer.AcceptWaveform(audio):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")
            if text.strip():
                # store latest recognized text for UI to read
                self.text_output = text

        return frame  # returning frame is fine; audio won't be sent back when sendback_audio=False

# ------------------------------------------------------
# STREAMING MICROPHONE INPUT
# - IMPORTANT: sendback_audio=False prevents the server from returning audio to the browser.
# ------------------------------------------------------
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={
        "audio": True,
        "video": False,
        # hint the browser to use echo cancellation (may help)
        "googEchoCancellation": True
    },
    sendback_audio=False,   # <------ VERY IMPORTANT: disable server -> browser audio (prevents robotic echo)
)

# ------------------------------------------------------
# DISPLAY RESULTS LIVE
# ------------------------------------------------------
if webrtc_ctx and webrtc_ctx.audio_processor:
    text = webrtc_ctx.audio_processor.text_output

    if text:
        st.subheader("🗣️ You said:")
        st.write(text)

        sentiment, intent, entities = analyze_text(text)
        category, emotion = classify_text_custom(text)

        sq, obj, rec = generate_AI(intent, sentiment, entities, category)

        st.subheader("📌 Analysis")
        st.json({
            "sentiment": sentiment,
            "intent": intent,
            "entities": entities,
            "category": category,
            "emotion": emotion
        })

        st.subheader("🤖 AI Response")
        st.write(f"**Suggested Question:** → {sq}")
        st.write(f"**Objection Handling:** → {obj}")
        st.write(f"**Recommendation:** → {rec}")
