from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("saba_behavior_model.pkl")

@app.route("/")
def home():
    return "SABA AI Model is Running"

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json()

    input_data = pd.DataFrame([{
        "session_minute": data["session_minute"],
        "task_difficulty": data["task_difficulty"],
        "sleep_quality": data["sleep_quality"],
        "hunger_level": data["hunger_level"],
        "sensory_load": data["sensory_load"],
        "preferred_items_available": data["preferred_items_available"],
        "refusal_count": data["refusal_count"],
        "leaving_seat_count": data["leaving_seat_count"]
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(max(probabilities) * 100, 1)

    reasons = []

    if data["refusal_count"] >= 3:
        reasons.append("ارتفاع عدد مرات الرفض")

    if data["leaving_seat_count"] >= 2:
        reasons.append("زيادة ترك الكرسي أو الحركة")

    if data["task_difficulty"] >= 3:
        reasons.append("صعوبة المهمة مرتفعة")

    if data["hunger_level"] >= 3:
        reasons.append("مستوى الجوع مرتفع")

    if data["sensory_load"] >= 3:
        reasons.append("الضغط الحسي مرتفع")

    if data["sleep_quality"] >= 3:
        reasons.append("جودة النوم ضعيفة")

    if data["preferred_items_available"] >= 3:
        reasons.append("المعززات ضعيفة أو غير كافية")

    reason_text = " + ".join(reasons) if reasons else "المؤشرات الحالية مستقرة"

    if prediction == "give_break":
        recommendation = "يوصى بإعطاء بريك قصير ثم العودة للمهمة بشكل تدريجي."
    elif prediction == "change_activity":
        recommendation = "يوصى بتعديل النشاط أو تقليل صعوبته واستخدام معزز مناسب."
    else:
        recommendation = "يمكن الاستمرار في الجلسة مع مراقبة المؤشرات السلوكية."

    return jsonify({
        "decision": prediction,
        "confidence": confidence,
        "reason": reason_text,
        "recommendation": recommendation
    })

if __name__ == "__main__":
    app.run(debug=True)