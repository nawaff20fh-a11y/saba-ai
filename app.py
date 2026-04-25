import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="SABA AI", layout="centered")

st.title("🧠 SABA – Smart ABA Assistant")

st.write("أدخل بيانات الجلسة وسيعطيك النظام القرار المناسب 👇")

# Inputs
age = st.number_input("عمر الطفل", min_value=2, max_value=18, value=5)
task_difficulty = st.slider("صعوبة المهمة", 1, 5, 3)
sleep_quality = st.slider("جودة النوم", 1, 5, 3)
hunger_level = st.slider("مستوى الجوع", 1, 5, 2)
sensory_load = st.slider("الضغط الحسي", 1, 5, 3)
refusal_count = st.number_input("عدد مرات الرفض", 0, 20, 2)
leaving_seat = st.number_input("ترك الكرسي", 0, 20, 1)
distraction = st.slider("التشتت", 1, 5, 3)

# Load model
model = joblib.load("saba_behavior_model.pkl")

# Predict
if st.button("🔍 تحليل القرار"):
    input_data = np.array([[age, task_difficulty, sleep_quality, hunger_level,
                            sensory_load, refusal_count, leaving_seat, distraction]])

    prediction = model.predict(input_data)[0]

    if prediction == "give_break":
        st.error("🛑 الطفل يحتاج بريك الآن")
    elif prediction == "change_activity":
        st.warning("🔄 يفضل تغيير النشاط")
    else:
        st.success("✅ كمل الجلسة")
