#нужные библиотеки
import streamlit as st
from fastai.vision.all import *
import plotly.express as px
from PIL import *
import pathlib
from decimal import Decimal

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath



st.title("Hush kelibsiz!!! 🙌")
st.header('Ushbu dastur :blue[sun'iy neyron tarmoqlari yordamida] :red[ko'krak qafasi rentgenogramma asosida] pnevmoniya kasalligini aniqlaydi⚕️')

#загружаем изображении

file_upload = st.file_uploader('Rentgenogramma tasvirini yuklash', type= ['png', 'jpeg', 'gif', 'svg', 'jpg'])

if file_upload:
    st.image(file_upload)

    img = PILImage.create(file_upload)



result = st.button('Rentgenogramma tasvirini tahlil qilish ')
if result:
    model = load_learner('Covid_19_model_new.pkl')
    pred, pred_id, probs = model.predict(img)
    if probs [pred_id] * 100 > 65:
        st.success(f'Natija: {pred}')
        st.success(f'Ehtimolligi: {probs [pred_id]*100/1}%')
    else:
        st.info('Rentgenogramma tasvirida muammo bor!!! Iltimos boshqa tasvir bilan urunib ko'ring')


     #plotly
    fig = px.bar(x=probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)
