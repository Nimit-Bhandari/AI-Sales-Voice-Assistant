import sounddevice as sd
import queue
import json
import sys
from vosk import Model, KaldiRecognizer
from textblob import TextBlob
import re

# =========================================================
# ============== DATASET FOR CATEGORY DETECTION ===========
# =========================================================

conversation_patterns = [
    {
        "category": "electronics_purchase",
        "emotion": "friendly_curious",
        "keywords": ["budget", "headset", "bluetooth", "order", "battery"]
    },
    {
        "category": "internet_plan_upgrade",
        "emotion": "polite_interested",
        "keywords": ["upgrade", "internet", "plan", "router", "mbps"]
    },
    {
        "category": "product_return",
        "emotion": "frustrated_relieved",
        "keywords": ["wrong", "received", "return", "replacement", "pickup"]
    },
    {
        "category": "travel_booking",
        "emotion": "excited_persuasive",
        "keywords": ["trip", "package", "goa", "weekend", "emi"]
    },
    {
        "category": "bank_query",
        "emotion": "calm_reassuring",
        "keywords": ["account", "debit card", "dispatched", "shipped"]
    },
    {
        "category": "mobile_recharge",
        "emotion": "neutral_satisfied",
        "keywords": ["recharge", "299", "data", "gb", "plan"]
    },
    {
        "category": "software_subscription",
        "emotion": "curious_convinced",
        "keywords": ["renew", "discount", "antivirus", "offer"]
    },
    {
        "category": "restaurant_reservation",
        "emotion": "polite_pleasant",
        "keywords": ["book table", "reservation", "8 pm", "indoor", "outdoor"]
    },
    {
        "category": "gadget_repair",
        "emotion": "concerned_relieved",
        "keywords": ["repair", "laptop", "won't turn on", "diagnosis"]
    },
    {
        "category": "clothing_sale",
        "emotion": "cheerful_excited",
        "keywords": ["sale", "discount", "winter jackets", "offer"]
    }
]


# =========================================================
# ====== CATEGORY + EMOTION CLASSIFICATION FUNCTION =======
# =========================================================

def classify_text_custom(text):
    text_lower = text.lower()
    detected_category = "unknown"
    detected_emotion = "neutral"

    for item in conversation_patterns:
        for kw in item["keywords"]:
            if kw in text_lower:
                detected_category = item["category"]
                detected_emotion = item["emotion"]
                break

    return {
        "category": detected_category,
        "emotion": detected_emotion
    }


# =========================================================
# ================== ENTITY EXTRACTION =====================
# =========================================================

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


# =========================================================
# ============ SENTIMENT + INTENT DETECTION ===============
# =========================================================

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
    elif "recharge" in t:
        intent = "mobile_recharge"
    else:
        intent = "general_statement"

    return {
        "sentiment": sentiment,
        "intent": intent,
        "entities": extract_entities(text)
    }


# =========================================================
# ============== FREE TEMPLATE-BASED REASONING ============
# =========================================================

def template_reasoning(sentiment, intent, entities, category, emotion):
    
    next_question = ""
    objection_response = ""
    recommendation = ""

    # SENTIMENT RULES
    if sentiment == "negative":
        objection_response = "I understand your concern, and I’ll resolve it quickly."
        next_question = "Could you tell me what went wrong exactly?"
    elif sentiment == "positive":
        next_question = "Great! Would you like more options based on this?"
    else:
        next_question = "Can you share a bit more details?"

    # INTENT RULES
    if intent == "purchase":
        next_question = "Do you have any preferred brand or budget?"
        recommendation = "I can help compare the best options."

    elif intent == "upgrade_request":
        next_question = "What is your current plan/device so I can suggest a better upgrade?"
        recommendation = "Higher tiers usually give smoother performance."

    elif intent == "return_request":
        next_question = "Was the issue related to defect, wrong item, or something else?"
        objection_response = "No worries, I’ll help with the return process."
        recommendation = "Replacement is also available."

    elif intent == "booking":
        next_question = "Which date and time do you prefer?"
        recommendation = "I can check the best available slots."

    elif intent == "mobile_recharge":
        next_question = "How much data do you need daily?"
        recommendation = "The ₹299 plan is usually enough for moderate usage."

    # CATEGORY RULES
    if category == "internet_plan_upgrade":
        recommendation = "Higher Mbps plans give smoother browsing."

    if category == "travel_booking":
        recommendation = "Weekend Goa packages are trending right now."

    if category == "gadget_repair":
        next_question = "Has this issue happened before?"
        recommendation = "A quick diagnosis will confirm the problem."

    # ENTITY RULE
    if entities:
        ent_text = ", ".join([e[0] for e in entities])
        next_question = f"Thanks for the details about {ent_text}. Could you share one more thing?"

    return {
        "next_question": next_question,
        "objection_response": objection_response,
        "recommendation": recommendation
    }


# =========================================================
# ===================== VOSK SETUP =========================
# =========================================================

MODEL_PATH = "vosk-model-en-us-0.22-lgraph"
SAMPLE_RATE = 16000

try:
    model = Model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

recognizer = KaldiRecognizer(model, SAMPLE_RATE)
audio_q = queue.Queue()

def callback(indata, frames, time, status):
    audio_q.put(bytes(indata))


# =========================================================
# ===================== MAIN LOOP ==========================
# =========================================================

with sd.RawInputStream(
    samplerate=SAMPLE_RATE,
    blocksize=8000,
    dtype="int16",
    channels=1,
    callback=callback
):
    print("\n🎤 Listening... Speak clearly. Press Ctrl+C to stop.\n")

    try:
        while True:
            data = audio_q.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")

                if text.strip():

                    base = analyze_text(text)
                    custom = classify_text_custom(text)

                    reasoning = template_reasoning(
                        sentiment=base["sentiment"],
                        intent=base["intent"],
                        entities=base["entities"],
                        category=custom["category"],
                        emotion=custom["emotion"]
                    )

                    final_output = {
                        "text": text,
                        "sentiment": base["sentiment"],
                        "intent": base["intent"],
                        "entities": base["entities"],
                        "category": custom["category"],
                        "emotion": custom["emotion"],
                        "ai_suggestions": reasoning
                    }

                    # SAVE TO JSON FOR DASHBOARD
                    with open("live_output.json", "w") as f:
                        json.dump(final_output, f, indent=4)

                    print(json.dumps(final_output, indent=4))

    except KeyboardInterrupt:
        print("\nStopped.")
