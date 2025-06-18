import streamlit as st
import pandas as pd
import numpy as np

from model.utils import read_model

st.set_page_config(
    page_title="Apartment App",
)

model_path = 'model/model.pkl'

# Загрузка модели
model = read_model(model_path)

# Заголовок
st.markdown("## Прогноз стоимости недвижимости")
st.markdown("Введите данные объекта ниже для получения оценки:")

# Форма ввода данных
st.header("Введите характеристики объекта:")

# Получаем данные
total_square = st.number_input("Площадь (кв.м)", min_value=1.0, max_value=3000.0, value=50.0, step=1.0)
rooms = st.number_input("Количество комнат", min_value=1, max_value=15, value=2, step=1)
floor = st.number_input("Этаж", min_value=1, max_value=70, value=3, step=1)
lat = st.slider("Широта", min_value=54.0, max_value=56.5, value=55.75, step=0.01)
lon = st.slider("Долгота", min_value=36.0, max_value=39.0, value=37.62, step=0.01)
city = st.selectbox("Город", options=['Москва', 'Балашиха', 'Люберцы', 'Красногорск', 'Химки', 'Королёв',
       'Мытищи', 'Пушкино', 'Котельники', 'Одинцово', 'Щёлково',
       'Дзержинский', 'Реутов', 'Ивантеевка', 'Московский', 'Лобня',
       'Долгопрудный', 'Щербинка', 'Подольск', 'Видное', 'Лыткарино'])

# Кнопка для запуска
if st.button("Прогнозировать цену"):
    # Создаём датафрейм из одного объекта
    input_data = pd.DataFrame([{
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "city": city
    }])

    # Предсказание
    prediction = model.predict(input_data)[0]
    formatted_price = f"{prediction:,.0f}".replace(",", " ")
    st.success(f"Прогнозируемая стоимость: {formatted_price} руб.")



