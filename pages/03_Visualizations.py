import streamlit as st
from PIL import Image
import os

# Установка настроек страницы
st.set_page_config(page_title="Дашборд анализа данных", layout="wide")

# Заголовок страницы
st.title(" Дашборд анализа данных и моделирования")
st.markdown("""
На этой странице представлены основные результаты анализа данных и оценки моделей машинного обучения для задачи прогнозирования цены дома.
""")

image_dir = "figures"
# Функция для отображения изображения с подписью
def display_image_with_caption(image_path, caption):
    image = Image.open(os.path.join(image_dir, image_path))
    st.image(image, caption=caption, use_container_width=True)

# Создание сетки для изображений
col1, col2 = st.columns(2)

st.subheader("Aнализ данных:")
    # corr_matrix.png
display_image_with_caption(
    "corr_matrix.png",
    "Тепловая карта корреляций\n"
)
st.write("Показывает взаимосвязь между признаками.")
st.write("Цвета отражают силу корреляции: от -1 до 1.")
st.write("Выявляет наиболее значимые признаки.")

with col1:
# Размещение графиков 
    st.subheader("Aнализ влияние важных признаков на цену:")
     # scatter_area_target.png
    display_image_with_caption(
        "scatter_area_target.png",
        "Зависимость price от area\n"
    )
    st.write("Помогает понять,что площадь — один из ключевых факторов для оценки цены")
    
    # boxplot_Bedrooms_price.png
    display_image_with_caption(
        "boxplot_Bedrooms_price.png",
        "Зависимость price от Bedrooms.\n"
    )
    st.write("Позволяет наглядно показать, что число спален — важный фактор для цены, но не единственный (так как разброс большой):")
    
    # pairplot.png
    display_image_with_caption(
        "pairplot.png",
        "Взаимосвязь важных признаков на целевую переменную.\n"
    )
    st.write("Видна положительная зависимость (чем больше площадь, тем выше цена), но разброс достаточно велик")
    st.write("По вертикальным полосам видно, что большее количество спален связано с ростом цены,но внутри каждой группы спален цены сильно варьируются.")
    st.write("Наличие ступенчатости: у квартир с большим числом спален — обычно больше площадь,но внутри каждого значения bedrooms разброс по area тоже заметный.")
    
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

    # xgb_metrics_table.png
    display_image_with_caption(
        "xgb_metrics_table.png",
        "Таблица метрик для XGBRegressor. \n"
    )
    
    # bbagging_metrics_table.png
    display_image_with_caption(
        "bagging_metrics_table.png",
        "Таблица метрик для Bagging Regressor. \n"
    )

    # stacking_metrics_table.png
    display_image_with_caption(
        "stacking_metrics_table.png",
        "Таблица метрик для StackingRegressor. \n"
    )

    # mlp_metrics_table.png
    display_image_with_caption(
        "mlp_metrics_table.png",
        "Таблица метрик для MLPRegressor. \n"
    )
    st.write("**R² (R-squared)**: Мера объяснённой дисперсии.")
    st.write("**RMSE (Root Mean Squared Error)**: Чем меньше, тем лучше точность модели.")
    st.write("**MAE (Mean Absolute Error)**: Средняя абсолютная ошибка.")
