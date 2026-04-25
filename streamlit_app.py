import streamlit as st
import pandas as pd
import joblib

model = joblib.load("saba_behavior_model.pkl")

st.set_page_config(
    page_title="SABA",
    layout="centered"
)

# 🔥 إزالة شعار Streamlit
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====== تصميم ======
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
    direction: rtl;
}

.main-card {
    background: linear-gradient(135deg, #0f172a, #1e40af);
    padding: 28px;
    border-radius: 28px;
    color: white;
    margin-bottom: 22px;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 26px;
    margin-bottom: 18px;
}

.result-box {
    background: white;
    padding: 24px;
    border-radius: 26px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ====== الهيدر ======
st.markdown("""
<div class="main-card">
    <h1>SABA</h1>
    <h3>Smart ABA Assistant</h3>
    <p>نظام ذكي يساعد الأخصائي على تحليل بيانات الجلسة واتخاذ القرار المناسب</p>
</div>
""", unsafe_allow_html=True)

# ====== الإدخال ======
st.markdown('<div class="card">', unsafe_allow_html=True)

session_minute = st.number_input("مدة العمل (دقائق)", 0, 120, 10)
refusal_count = st.number_input("عدد مرات الرفض", 0, 20, 0)
leaving_seat_count = st.number_input("عدد مرات ترك الكرسي", 0, 20, 0)

task_difficulty_label = st.selectbox("صعوبة المهمة", ["منخفضة", "متوسطة", "مرتفعة"])
hunger_label = st.selectbox("مستوى الجوع", ["منخفض", "متوسط", "مرتفع"])
sleep_label = st.selectbox("جودة النوم", ["جيدة", "متوسطة", "ضعيفة"])
sensory_label = st.selectbox("الضغط الحسي", ["منخفض", "متوسط", "مرتفع"])
reinforcer_label = st.selectbox("توفر المعززات", ["متوفر", "متوسط", "ضعيف"])

task_difficulty = {"منخفضة": 1, "متوسطة": 2, "مرتفعة": 3}[task_difficulty_label]
hunger_level = {"منخفض": 1, "متوسط": 2, "مرتفع": 3}[hunger_label]
sleep_quality = {"جيدة": 1, "متوسطة": 2, "ضعيفة": 3}[sleep_label]
sensory_load = {"منخفض": 1, "متوسط": 2, "مرتفع": 3}[sensory_label]
preferred_items_available = {"متوفر": 1, "متوسط": 2, "ضعيف": 3}[reinforcer_label]

analyze = st.button("تحليل الحالة", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ====== التحليل ======
if analyze:

    input_data = pd.DataFrame([{
        "session_minute": session_minute,
        "task_difficulty": task_difficulty,
        "sleep_quality": sleep_quality,
        "hunger_level": hunger_level,
        "sensory_load": sensory_load,
        "preferred_items_available": preferred_items_available,
        "refusal_count": refusal_count,
        "leaving_seat_count": leaving_seat_count
    }])

    prediction = model.predict(input_data)[0]
    confidence = round(max(model.predict_proba(input_data)[0]) * 100, 1)

    reasons = []
    if refusal_count >= 3: reasons.append("ارتفاع الرفض")
    if leaving_seat_count >= 2: reasons.append("زيادة الحركة")
    if sensory_load >= 3: reasons.append("ضغط حسي مرتفع")
    if hunger_level >= 3: reasons.append("جوع مرتفع")
    if sleep_quality >= 3: reasons.append("نوم ضعيف")

    reason_text = " + ".join(reasons) if reasons else "المؤشرات مستقرة"

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if prediction == "give_break":
        st.error("إعطاء استراحة")
        recommendation = "اعطاء بريك ثم العودة تدريجياً"
    elif prediction == "change_activity":
        st.warning("تعديل النشاط")
        recommendation = "تغيير النشاط أو تقليل صعوبته"
    else:
        st.success("استمرار الجلسة")
        recommendation = "يمكن الاستمرار مع المراقبة"

    st.info(f"السبب: {reason_text}")
    st.success(f"التوصية: {recommendation}")
    st.metric("ثقة المودل", f"{confidence}%")
    st.progress(confidence / 100)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="result-box">
        <h3>القرار المقترح</h3>
        <p>أدخل البيانات ثم اضغط تحليل الحالة</p>
    </div>
    """, unsafe_allow_html=True)
