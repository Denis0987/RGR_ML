import streamlit as st
from PIL import Image
import os

# Установка настроек страницы
st.set_page_config(page_title="Дашборд анализа данных", layout="wide")

# Заголовок страницы
st.title(" Дашборд анализа данных и моделирования")
st.markdown("""
На этой странице представлены основные результаты анализа данных и оценки моделей машинного обучения для задачи прогнозирования цены втомобиля.
""")

image_dir = "figures"
# Функция для отображения изображения с подписью
def display_image_with_caption(image_path, caption):
    image = Image.open(os.path.join(image_dir, image_path))
    st.image(image, caption=caption, use_container_width=True)
    
st.subheader("Aнализ данных:")
# Создание сетки для изображений
# corr_matrix_all.png
display_image_with_caption(
    "corr_matrix_all.png",
    "Тепловая карта корреляций (для всех признаков)\n"
)
st.write("Показывает взаимосвязь между признаками.")
st.write("Цвета отражают силу корреляции: от -1 до 1.")
st.write("Выявляет наиболее значимые признаки.")

# corr_matrix.png
display_image_with_caption(
    "corr_matrix.png",
    "Тепловая карта корреляций (для признаков влияющих на цену)\n"
)
st.write("Показывает взаимосвязь между важными признаками.")
st.write("Цвета отражают силу корреляции: от -1 до 1.")
    
# Создание сетки для изображений
col1, col2 = st.columns(2)

with col1:
# Размещение графиков 
    st.subheader("Aнализ влияние важных признаков на цену:")
     # scatter_year_produced_price_usd.png
    display_image_with_caption(
        "scatter_year_produced_price_usd.png",
        "Зависимость price_usd от year_produced\n"
    )
    st.write("Помогает понять,что с увеличением года производства цена автомобиля значительно возрастает.")
    
    # boxplot_car_age.png
    display_image_with_caption(
        "boxplot_car_age.png",
        "Анализ важного признака- car_age.\n"
    )
    st.write("Большинство автомобилей имеет возраст около 20 лет, с некоторыми выбросами старых автомобилей (более 30 лет)")
    
    # pairplot.png
    display_image_with_caption(
        "pairplot.png",
        "Взаимосвязь важных признаков влияющих на целевую переменную.\n"
    )
    st.write("Молодые автомобили имеют более высокие цены")
    st.write("Избыток автомобилей с возрастом около 20 лет, что также подтверждается распределением на диагональных гистограммах")
    st.write("Цены на старые автомобили значимо падают с возрастом, что ожидаемо")
    
with col2:
    st.subheader("Oценки моделей машинного обучения (на предобработанных данных):")
    # polynomial_metrics_table.png
    display_image_with_caption(
        "polynomial_metrics_table.png",
        "Таблица метрик для Polynomial. \n"
    )

    # boosting_metrics_table.png
    display_image_with_caption(
        "boosting_metrics_table.png",
        "Таблица метрик для Gradient Boosting Regressor. \n"
    )

    # catboost_metrics_table.png
    display_image_with_caption(
        "catboost_metrics_table.png",
        "Таблица метрик для CatBoost Regressor. \n"
    )
    
    # bbagging_metrics_table.png
    display_image_with_caption(
        "bagging_metrics_table.png",
        "Таблица метрик для Bagging Regressor. \n"
    )

    # stacking_metrics_table.png
    display_image_with_caption(
        "stacking_metrics_table.png",
        "Таблица метрик для Stacking Regressor. \n"
    )

    # mlp_metrics_table.png
    display_image_with_caption(
        "mlp_metrics_table.png",
        "Таблица метрик для MLPRegressor. \n"
    )
    st.write("**R² (R-squared)**: Мера объяснённой дисперсии.")
    st.write("**RMSE (Root Mean Squared Error)**: Чем меньше, тем лучше точность модели.")
    st.write("**MAE (Mean Absolute Error)**: Средняя абсолютная ошибка.")

