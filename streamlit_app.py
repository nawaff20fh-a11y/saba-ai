import streamlit as st
import pandas as pd
import joblib

# تحميل المودل
model = joblib.load("saba_behavior_model.pkl")

st.set_page_config(page_title="SABA", layout="centered")

st.title("SABA")
st.write("Smart ABA Assistant")

st.subheader("بيانات الجلسة")

session_minute = st.number_input("مدة العمل بالدقائق", min_value=0, max_value=120, value=10, step=1)
refusal_count = st.number_input("عدد مرات الرفض", min_value=0, max_value=20, value=0, step=1)
leaving_seat_count = st.number_input("عدد مرات ترك الكرسي", min_value=0, max_value=20, value=0, step=1)

task_difficulty = st.selectbox(
    "صعوبة المهمة",
    [1, 2, 3],
    format_func=lambda x: ["منخفضة", "متوسطة", "مرتفعة"][x - 1]
)

hunger_level = st.selectbox(
    "مستوى الجوع",
    [1, 2, 3],
    format_func=lambda x: ["منخفض", "متوسط", "مرتفع"][x - 1]
)

sleep_quality = st.selectbox(
    "جودة النوم",
    [1, 2, 3],
    format_func=lambda x: ["جيدة", "متوسطة", "ضعيفة"][x - 1]
)

sensory_load = st.selectbox(
    "الضغط الحسي",
    [1, 2, 3],
    format_func=lambda x: ["منخفض", "متوسط", "مرتفع"][x - 1]
)

preferred_items_available = st.selectbox(
    "توفر المعززات",
    [1, 2, 3],
    format_func=lambda x: ["متوفر", "متوسط", "ضعيف"][x - 1]
)

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
    confidence = round(max(model.predict_proba(input_data)[0]) * 100, 1)

    if prediction == "give_break":
        st.error("إعطاء استراحة")
        recommendation = "يوصى بإعطاء بريك قصير ثم العودة تدريجياً."
    elif prediction == "change_activity":
        st.warning("تعديل النشاط")
        recommendation = "يوصى بتغيير النشاط أو تقليل صعوبته."
    else:
        st.success("استمرار الجلسة")
        recommendation = "يمكن الاستمرار مع المراقبة."

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
    if preferred_items_available >= 3:
        reasons.append("المعززات ضعيفة")

    reason_text = " + ".join(reasons) if reasons else "المؤشرات مستقرة"

    st.info("السبب: " + reason_text)
    st.success("التوصية: " + recommendation)
    st.metric("ثقة المودل", f"{confidence}%")
    st.progress(confidence / 100)
