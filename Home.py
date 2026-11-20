import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

st.header("โปรเจคการจำแนกคุณภาพไวน์ขาว")
html_7 = """
   <div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
   <center><h5>สถิติข้อมูลไวน์ขาว</h5></center>
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
      #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
   st.write("ไม่แสดงข้อมูล")

   html_8 = """
   <div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
   <center><h5>ทำนายข้อมูล</h5></center>
   </div>
   """
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

alcohol = st.slider("ปริมาณแอลกอฮอล์ (alcohol)", 8.0, 14.2, 10.5)
sugar = st.slider("น้ำตาลคงเหลือ (residualsugar)", 0.6, 65.0, 5.0)
ph = st.number_input("ค่าความเป็นกรด-ด่าง (pH)", 2.7, 3.8, 3.2)
acidity = st.number_input("ค่ากรดระเหย (volatileacidity)", 0.08, 1.1, 0.26)

if st.button("ทำนายผล"):
      #st.write("ทำนาย")
      #dt = pd.read_csv("./data/iris.csv") 

   X = dt('alcohol','residualsugar','pH','volatileacidity' axis=1)
   y = (dt['quality'] >= 7).atypes(int)

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = np.array([[alcohol, residualsugar, pH, volatileacidity]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

if out[0] == '1':
   st.success("ไวน์คุณภาพดี")
else:
   st.success("ไวน์ทั่วไป")