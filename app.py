import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Sahifa sozlamalari va CSS uslubi
st.set_page_config(page_title="Dori Vositasini Bashorat qilish", layout="centered")
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(120deg, #84fab0, #8fd3f4);
            padding: 2rem;
            border-radius: 10px;
        }
        input, select {
            border: 2px solid #8fd3f4;
            border-radius: 8px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5em 1em;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
        .stMarkdown {
            color: #08415C;
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sahifa sarlavhasi
st.title("💊 Dori Vositasini Bashorat qilish")

# Ma'lumot kiritish qismiga yo'riqnoma
with st.expander("📋 Kirish uchun yo'riqnoma"):
    st.write("Quyidagi ma'lumotlarni kiriting va `Dorini Bashorat qilish` tugmasini bosing. "
             "Kiritilgan ma'lumotlarga asosan mos dori vositasi tavsiya qilinadi.")

# Kirish qutilari
age = st.slider("Yoshni kiriting:", 0, 120, 25)
sex = st.radio("Jinsni tanlang:", ["Erkak", "Ayol"], horizontal=True)
bp = st.selectbox("Qon bosimi:", ["Past", "Normal", "Yuqori"])
cholesterol = st.selectbox("Xolesterin darajasi:", ["Normal", "Yuqori"])
na_to_k = st.number_input("Na_to_K nisbatini kiriting:", min_value=0.0, max_value=50.0, value=12.0, step=0.1)

# Foydalanuvchi kiritgan ma'lumotlarni tekshirish
st.markdown("### Kiritilgan Ma'lumotlar:")
st.write(f"- **Yosh:** {age}")
st.write(f"- **Jins:** {sex}")
st.write(f"- **Qon bosimi:** {bp}")
st.write(f"- **Xolesterin darajasi:** {cholesterol}")
st.write(f"- **Na_to_K nisbati:** {na_to_k}")
st.markdown("---")

# Bashorat tugmasi
if st.button("🔍 Dorini Bashorat qilish"):
    # Ma'lumotlarni raqamli formatga o'tkazish
    sex_val = 1 if sex == "Erkak" else 0
    bp_mapping = {"Past": 1, "Normal": 2, "Yuqori": 0}
    cholesterol_mapping = {"Normal": 1, "Yuqori": 0}

    bp_val = bp_mapping[bp]
    cholesterol_val = cholesterol_mapping[cholesterol]
    
    # Modelga kiruvchi ma'lumotlar
    model_input = [[age, sex_val, bp_val, cholesterol_val, na_to_k]]
    
    # Modelni yuklash va ehtimolliklar bilan bashorat qilish
    model = joblib.load('decision_tree_model (1).pkl')
    pred_proba = model.predict_proba(model_input)[0]
    
    # Natijani ko'rsatish
    dori_guruhlari = ["Drug A", "Drug B", "Drug C", "Drug X", "Drug Y"]
    recommended_drug = dori_guruhlari[np.argmax(pred_proba)]
    
    st.success(f"💊 Ushbu bemorga mos dori: **{recommended_drug}**")
    
    # Har bir guruh uchun foizlarni ko'rsatish uchun diagramma
    fig, ax = plt.subplots()
    ax.bar(dori_guruhlari, pred_proba * 100, color=['#4CAF50', '#FF5733', '#33B5E5', '#FFC107', '#9C27B0'])
    ax.set_xlabel("Dori Guruhlari")
    ax.set_ylabel("Ehtimollik (%)")
    ax.set_title("Dori Guruhlariga Moslik Diagrammasi")
    
    st.pyplot(fig)