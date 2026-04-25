import streamlit as st
import pandas as pd
import joblib

model = joblib.load("saba_behavior_model.pkl")

st.set_page_config(
    page_title="SABA",
    layout="centered"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stApp {
    background-color: #f4f7fb;
    direction: rtl;
}
.block-container {
    max-width: 520px;
    padding-top: 2rem;
}
.main-card {
    background: linear-gradient(135deg, #0f172a, #1e40af);
    padding: 28px;
    border-radius: 28px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 18px 40px rgba(30,64,175,0.25);
}
.result-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    margin-top: 18px;
    box-shadow: 0 10px 25px rgba(15,23,42,0.08);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-card">
    <h1 style="margin-bottom:5px;">SABA</h1>
    <h3 style="margin-top:0;">Smart ABA Assistant</h3>
    <p>نظام ذكي يساعد الأخصائي على تحليل بيانات الجلسة واقتراح القرار المناسب.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("بيانات الجلسة")

session_minute = st.number_input("مدة العمل بالدقائق", min_value=0, max_value=120, value=10, step=1)
refusal_count = st.number_input("عدد مرات الرفض", min_value=0, max_value=20, value=0, step=1)
leaving_seat_count = st.number_input("عدد مرات ترك الكرسي", min_value=0, max_value=20, value=0, step=1)

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

if st.button("تحليل الحالة", use_container_width=True):
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
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(max(probabilities) * 100, 1)

    reasons = []

    if refusal_count >= 3:
        reasons.append("ارتفاع عدد مرات الرفض")
    if leaving_seat_count >= 2:
        reasons.append("زيادة ترك الكرسي أو الحركة")
    if task_difficulty >= 3:
        reasons.append("صعوبة المهمة مرتفعة")
    if hunger_level >= 3:
        reasons.append("مستوى الجوع مرتفع")
    if sensory_load >= 3:
        reasons.append("الضغط الحسي مرتفع")
    if sleep_quality >= 3:
        reasons.append("جودة النوم ضعيفة")
    if preferred_items_available >= 3:
        reasons.append("المعززات ضعيفة أو غير كافية")

    reason_text = " + ".join(reasons) if reasons else "المؤشرات الحالية مستقرة"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if prediction == "give_break":
        st.error("إعطاء استراحة")
        recommendation = "يوصى بإعطاء بريك قصير ثم العودة للمهمة بشكل تدريجي."
    elif prediction == "change_activity":
        st.warning("تعديل النشاط")
        recommendation = "يوصى بتعديل النشاط أو تقليل صعوبته واستخدام معزز مناسب."
    else:
        st.success("استمرار الجلسة")
        recommendation = "يمكن الاستمرار في الجلسة مع مراقبة المؤشرات السلوكية."

    st.info("سبب القرار: " + reason_text)
    st.success("التوصية: " + recommendation)
    st.metric("ثقة المودل", str(confidence) + "%")
    st.progress(confidence / 100)

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="result-card">
        <h3>القرار المقترح</h3>
        <p>أدخل بيانات الجلسة ثم اضغط تحليل الحالة.</p>
    </div>
    """, unsafe_allow_html=True)
