import streamlit as st
import pandas as pd
import joblib

# تحميل المودل
model = joblib.load("saba_behavior_model.pkl")

# إعداد الصفحة
st.set_page_config(
    page_title="SABA AI",
    page_icon="🧠",
    layout="centered"
)

# ====== Header ======
st.markdown("""
    <h1 style='text-align:center;'>SABA</h1>
    <p style='text-align:center; color:gray;'>
    Smart ABA Assistant — نظام دعم قرار ذكي للأخصائيين
    </p>
    <hr>
""", unsafe_allow_html=True)

# ====== Input Section ======
st.subheader("📊 بيانات الجلسة")

col1, col2 = st.columns(2)

with col1:
    session_minute = st.number_input("مدة العمل (دقائق)", 0, 60, 15)
    refusal_count = st.number_input("عدد مرات الرفض", 0, 10, 0)
    leaving_seat_count = st.number_input("عدد مرات ترك الكرسي", 0, 10, 0)
    task_difficulty = st.selectbox("صعوبة المهمة", [1,2,3], format_func=lambda x: ["منخفضة","متوسطة","مرتفعة"][x-1])

with col2:
    hunger_level = st.selectbox("مستوى الجوع", [1,2,3], format_func=lambda x: ["منخفض","متوسط","مرتفع"][x-1])
    sleep_quality = st.selectbox("جودة النوم", [1,2,3], format_func=lambda x: ["جيدة","متوسطة","ضعيفة"][x-1])
    sensory_load = st.selectbox("الضغط الحسي", [1,2,3], format_func=lambda x: ["منخفض","متوسط","مرتفع"][x-1])
    preferred_items_available = st.selectbox("توفر المعززات", [1,2,3], format_func=lambda x: ["متوفر","متوسط","ضعيف"][x-1])

st.markdown("---")

# ====== Analyze Button ======
if st.button("🚀 تحليل الحالة", use_container_width=True):

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

    # ===== القرار =====
    if prediction == "give_break":
        st.error("🔴 إعطاء استراحة")
        recommendation = "يفضل إعطاء بريك فوري ثم العودة تدريجياً."
    elif prediction == "change_activity":
        st.warning("🟠 تعديل النشاط")
        recommendation = "يفضل تغيير النشاط أو تقليل صعوبته."
    else:
        st.success("🟢 استمرار الجلسة")
        recommendation = "يمكن الاستمرار مع مراقبة السلوك."

    # ===== السبب =====
    reasons = []

    if refusal_count >= 3:
        reasons.append("ارتفاع الرفض")
    if leaving_seat_count >= 2:
        reasons.append("زيادة الحركة")
    if sensory_load >= 3:
        reasons.append("ضغط حسي مرتفع")
    if hunger_level >= 3:
        reasons.append("جوع مرتفع")
    if sleep_quality >= 3:
        reasons.append("نوم ضعيف")

    reason_text = " + ".join(reasons) if reasons else "المؤشرات مستقرة"

    # ===== عرض النتائج =====
    st.markdown("### 🧠 التحليل")
    st.info(f"**السبب:** {reason_text}")

    st.markdown("### 💡 التوصية")
    st.success(recommendation)

    st.markdown("### 📈 ثقة المودل")
    st.progress(confidence / 100)
    st.write(f"{confidence}%")
