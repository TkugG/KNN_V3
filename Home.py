import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

st.markdown(
    """
    <style>
    .stApp {
        background-color: #29292b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

html_7 = """
   stype="body: white;"
   """

st.header("โปรเจคการจำแนกคุณภาพไวน์ขาว")
html_7 = """
   <div style="background: linear-gradient(135deg, #2C3E50, #4CA1AF); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.15); margin-bottom: 20px;">
    <h2 style="color: white; margin: 0;">🥂 โปรเจคการจำแนกคุณภาพไวน์ขาว</h2>
</div>
   """
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/winequality-white.csv", sep=';')
st.write(dt.head(10))

dt1 = dt['alcohol'].sum()
dt2 = dt['residualsugar'].sum()
dt3 = dt['pH'].sum()
dt4 = dt['volatileacidity'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
   st.write("ไม่แสดงข้อมูล")

html_8 = """
   <div style="background: linear-gradient(90deg, #129df3, #121df3); padding: 12px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
    <h5 style="color: white; margin: 0; font-weight: bold;">🤓☝️ สถิติข้อมูลไวน์ขาว</h5>
</div>
   """
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

alcohol = st.slider("ปริมาณแอลกอฮอล์ (alcohol)", 8.0, 14.2, 10.5)
sugar = st.slider("น้ำตาลคงเหลือ (residualsugar)", 0.6, 65.0, 5.0)
ph = st.number_input("ค่าความเป็นกรด-ด่าง (pH)", 2.7, 3.8, 3.2)
acidity = st.number_input("ค่ากรดระเหย (volatileacidity)", 0.08, 1.1, 0.26)

if st.button("ทำนายผล"): 
   X = dt[['alcohol', 'residualsugar', 'pH', 'volatileacidity']]
   y = (dt['quality'] >= 7).astype(int)

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[alcohol, sugar, ph, acidity]])
   st.write(Knn_model.predict(x_input)) 
   out=Knn_model.predict(x_input)

   if out[0] == 1:
      st.markdown("""
        <div style="background-color: #D4EFDF; border-left: 6px solid #27AE60; padding: 20px; border-radius: 10px; margin-top: 15px;">
            <h4 style="color: #145A32; margin: 0;">🏆 ผลการทำนาย: ไวน์คุณภาพดี (พรีเมียม)</h4>
            <p style="color: #1E8449; margin: 5px 0 0 0;">สัดส่วนสารเคมีอยู่ในเกณฑ์ไวน์คุณภาพสูง (คะแนนประเมิน ≥ 7)</p>
        </div>
        """, unsafe_allow_html=True)
      st.image("./img/winew.png")
   else:
      st.markdown("""
        <div style="background-color: #EBF5FB; border-left: 6px solid #2980B9; padding: 20px; border-radius: 10px; margin-top: 15px;">
            <h4 style="color: #1B4F72; margin: 0;">🍷 ผลการทำนาย: ไวน์คุณภาพทั่วไป</h4>
            <p style="color: #21618C; margin: 5px 0 0 0;">สัดส่วนสารเคมีอยู่ในเกณฑ์มาตรฐานทั่วไป (คะแนนประเมิน < 7)</p>
        </div>
        """, unsafe_allow_html=True)
      st.image("./img/wine.png")
else:
    st.write("ไม่ทำนาย")