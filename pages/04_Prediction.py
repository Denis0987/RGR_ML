import streamlit as st
import pandas as pd
import pickle
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Установка заголовка страницы
st.set_page_config(page_title="Предсказание", layout="centered")
st.title("Прогнозирование стоимости недвижимости")

st.markdown("Загрузите CSV-файл или введите данные вручную для получения прогноза.")

# Путь к папке с моделями
models_dir = "models"

# Проверка наличия папки models
if not os.path.exists(models_dir):
    st.error(f"Папка с моделями не найдена: {models_dir}")
else:
    # Получаем список всех .pkl и .json файлов
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') or f.endswith('.json')]

    if not model_files:
        st.warning("В папке models нет моделей (.pkl или .json файлов)")
    else:
        # Выбор модели пользователем
        selected_model = st.selectbox("Выберите модель", model_files)

        # Определяем тип файла
        model_path = os.path.join(models_dir, selected_model)
        if selected_model.endswith('.pkl'):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            st.success(f"Модель '{selected_model}' (.pkl) загружена")
        elif selected_model.endswith('.json'):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            st.success(f"Модель '{selected_model}' (.json, XGBoost) загружена")
        else:
            st.error("Неподдерживаемый формат модели!")
            model = None

        if model is not None:
            # Получаем список признаков модели
            model_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.get_booster().get_fscore().keys()

            # Разделение на две колонки: загрузка файла и ручной ввод
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Загрузите CSV-файл")
                uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
                if uploaded_file:
                    try:
                        df_uploaded = pd.read_csv(uploaded_file)
                        st.success("Файл успешно загружен!")
                        st.write("Предпросмотр данных:")
                        st.dataframe(df_uploaded.head())

                        # Обработка пропущенных значений
                        df_uploaded = df_uploaded.fillna(0)  # Заполняем пропуски нулями

                        # Преобразуем категориальные признаки в one-hot (если необходимо)
                        df_uploaded = pd.get_dummies(df_uploaded, drop_first=True)

                        # Приводим данные в такой же порядок, как и в модели
                        missing_columns = set(model_columns) - set(df_uploaded.columns)
                        for col in missing_columns:
                            df_uploaded[col] = 0  # Добавляем недостающие признаки с нулевым значением

                        # Приводим порядок признаков к нужному
                        df_uploaded = df_uploaded[model_columns]  # Применяем правильный порядок

                        # Предсказание по файлу
                        X = df_uploaded
                        predictions = model.predict(X)
                        df_uploaded['predicted_trip_duration'] = predictions
                        st.download_button(
                            label="Скачать с предсказаниями",
                            data=df_uploaded.to_csv(index=False),
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

                        # Показываем первые 10 строк с предсказаниями
                        st.subheader("Предсказания для загруженных данных:")
                        st.write(df_uploaded.head(10))  # Показываем 10 строк с предсказаниями

                    except Exception as e:
                        st.error(f"Ошибка при обработке файла: {e}")

            with col2:
                st.subheader("Ввод параметров объекта недвижимости вручную")

                area = st.number_input("Площадь (кв.м):", min_value=10.0, max_value=10000.0, value=50.0)
                price = st.number_input("Цена (в рупиях):", min_value=0.0, max_value=10_000_000_000.0, value=5000000.0)
                latitude = st.number_input("Ширина (latitude):", min_value=-90.0, max_value=90.0, value=19.0760)
                longitude = st.number_input("Долгота (longitude):", min_value=-180.0, max_value=180.0, value=72.8777)
                bedrooms = st.number_input("Количество спален:", min_value=1, max_value=10, value=2)
                bathrooms = st.number_input("Количество ванных комнат:", min_value=1, max_value=10, value=1)
                balcony = st.number_input("Количество балконов:", min_value=0, max_value=10, value=1)
                parking = st.number_input("Наличие парковки (1 — есть, 0 — нет):", min_value=0, max_value=1, value=1)
                lift = st.number_input("Наличие лифта (1 — есть, 0 — нет):", min_value=0, max_value=1, value=1)

                # Статус недвижимости
                status = st.selectbox("Статус недвижимости", [
                    "Ready to Move", "Under Construction", "unknown"
                ])
                status_ready = 1 if status == "Ready to Move" else 0
                status_under = 1 if status == "Under Construction" else 0
                status_unknown = 1 if status == "unknown" else 0

                # Тип здания
                type_building = st.selectbox("Тип здания", [
                    "Flat", "Individual House"
                ])
                type_flat = 1 if type_building == "Flat" else 0
                type_house = 1 if type_building == "Individual House" else 0

                # Меблировка
                furnished = st.selectbox("Меблировка", [
                    "Furnished", "Semi-Furnished", "Unfurnished", "unknown"
                ])
                furn_furnished = 1 if furnished == "Furnished" else 0
                furn_semi = 1 if furnished == "Semi-Furnished" else 0
                furn_unfurnished = 1 if furnished == "Unfurnished" else 0
                furn_unknown = 1 if furnished == "unknown" else 0

                # Новый или старый дом
                neworold = st.selectbox("Состояние дома", ["New", "Old"])
                new_home = 1 if neworold == "New" else 0
                old_home = 1 if neworold == "Old" else 0

                # Добавление недостающих признаков
                input_data = pd.DataFrame({
                    'price': [price],
                    'area': [area],
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'balcony': [balcony],
                    'parking': [parking],
                    'lift': [lift],
                    'Status_Ready to Move': [status_ready],
                    'Status_Under Construction': [status_under],
                    'Status_unknown': [status_unknown],
                    'type_of_building_Flat': [type_flat],
                    'type_of_building_Individual House': [type_house],
                    'Furnished_status_Furnished': [furn_furnished],
                    'Furnished_status_Semi-Furnished': [furn_semi],
                    'Furnished_status_Unfurnished': [furn_unfurnished],
                    'Furnished_status_unknown': [furn_unknown],
                    'neworold_New': [new_home],
                    'neworold_Old': [old_home],
                })

                # Добавляем недостающие признаки с нулевыми значениями
                missing_columns = set(model_columns) - set(input_data.columns)
                for col in missing_columns:
                    input_data[col] = 0  # Добавляем недостающие признаки с нулевым значением

                # Приводим признаки в правильный порядок
                input_data = input_data[model_columns]  # Применяем правильный порядок

                # Получаем предсказание
                if st.button("Получить прогноз стоимости"):
                    prediction = model.predict(input_data)[0]
                    
                    # Форматируем число с правильным разделением тысяч
                    formatted_prediction = "{:,.0f}".format(prediction).replace(",", " ")

                    st.success(f"Оценочная стоимость: **{formatted_prediction} рупий**")
