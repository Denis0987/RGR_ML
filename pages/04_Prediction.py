import streamlit as st
import pandas as pd
import pickle
import os
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

# Установка заголовка страницы
st.set_page_config(page_title="Предсказание", layout="centered")
st.title("Прогнозирование стоимости автомобиля")

st.markdown("Загрузите CSV-файл или введите данные вручную для получения прогноза.")

# Путь к папке с моделями
models_dir = "models"

# Проверка наличия папки models
if not os.path.exists(models_dir):
    st.error(f"Папка с моделями не найдена: {models_dir}")
else:
    # Получаем список всех .pkl и .json файлов
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') or f.endswith('.json') or f.endswith('.cbm')]

    if not model_files:
        st.warning("В папке models нет моделей (.pkl, .json или .cbm файлов)")
    else:
        # Выбор модели пользователем
        selected_model = st.selectbox("Выберите модель", model_files)

        # Определяем тип файла
        model_path = os.path.join(models_dir, selected_model)
        if selected_model.endswith('.pkl'):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            st.success(f"Модель '{selected_model}' (.pkl) загружена")
        elif selected_model.endswith('.json') or selected_model.endswith('.cbm'):
            model = CatBoostRegressor()
            model.load_model(model_path)  # Загружаем модель CatBoost
            st.success(f"Модель '{selected_model}' (.json, .cbm, CatBoost) загружена")
        else:
            st.error("Неподдерживаемый формат модели!")
            model = None

        if model is not None:
            # Для CatBoost мы извлекаем признаки из модели
            if isinstance(model, CatBoostRegressor):
                model_columns = model.feature_names_
            else:
                # Здесь указываем все признаки, которые использовались при обучении модели
                model_columns = [
                    'odometer_value', 'year_produced', 'engine_has_gas', 'engine_capacity', 'has_warranty', 
                    'is_exchangeable', 'number_of_photos', 'up_counter', 'feature_0', 'feature_1', 'feature_2', 
                    'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 
                    'duration_listed', 'car_age', 'region_ad_count', 'manufacturer_name_encoded', 'model_name_encoded', 
                    'transmission_encoded', 'color_encoded', 'engine_fuel_encoded', 'engine_type_encoded', 'body_type_encoded', 
                    'state_encoded', 'drivetrain_encoded', 'location_region_encoded'
                ]

            # Разделение на две колонки: загрузка файла и ручной ввод
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Загрузите CSV-файл")
                uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
                if uploaded_file:
                    try:
                        # Загрузим CSV-файл
                        df = pd.read_csv(uploaded_file)
                        st.success("Файл успешно загружен!")
                        st.write("Предпросмотр данных:")
                        st.dataframe(df.head())

                        # Удаляем целевой признак price_usd из данных для предсказания
                        df = df.drop('price_usd', axis=1, errors='ignore')

                        # Обработка пропущенных значений
                        df = df.fillna(0)  # Заполняем пропуски нулями

                        # Преобразуем категориальные признаки в one-hot (если необходимо)
                        df = pd.get_dummies(df, drop_first=True)

                        # Обработка всех недостающих признаков, если они отсутствуют в новых данных
                        missing_columns = set(model_columns) - set(df.columns)
                        for col in missing_columns:
                            df[col] = 0  # Добавляем недостающие признаки с нулевыми значениями

                        # Приводим порядок признаков к нужному
                        df = df[model_columns]  # Применяем правильный порядок признаков

                        # Предсказание по файлу
                        X = df
                        predictions = model.predict(X)
                        df['predicted_price'] = predictions
                        st.download_button(
                            label="Скачать с предсказаниями",
                            data=df.to_csv(index=False),
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

                        # Показываем первые 10 строк с предсказаниями
                        st.subheader("Предсказания для загруженных данных:")
                        st.write(df.head(10))  # Показываем 10 строк с предсказаниями

                    except Exception as e:
                        st.error(f"Ошибка при обработке файла: {e}")

            with col2:
                st.subheader("Ввод параметров автомобиля вручную")

                manufacturer_name = st.text_input("Название производителя автомобиля")
                model_name = st.text_input("Модель автомобиля")
                transmission = st.selectbox("Тип трансмиссии", ["Автоматическая", "Механическая"])
                color = st.text_input("Цвет автомобиля")
                odometer_value = st.number_input("Пробег автомобиля (км или милях):", min_value=0.0, value=0.0)
                year_produced = st.number_input("Год выпуска автомобиля:", min_value=1900, max_value=2022, value=2020)
                engine_fuel = st.selectbox("Тип топлива", ["Бензин", "Дизель", "Электричество"])
                engine_has_gas = st.selectbox("Наличие газового оборудования", ["True", "False"])
                engine_type = st.selectbox("Тип двигателя", ["Бензиновый", "Дизельный", "Гибридный"])
                engine_capacity = st.number_input("Объем двигателя (л):", min_value=0.0, value=2.0)
                body_type = st.selectbox("Тип кузова", ["Седан", "Хэтчбек", "Внедорожник"])
                has_warranty = st.selectbox("Наличие гарантии", ["True", "False"])
                state = st.selectbox("Состояние автомобиля", ["Новый", "Б/У"])
                drivetrain = st.selectbox("Тип привода", ["Передний", "Задний", "Полный"])
                is_exchangeable = st.selectbox("Возможность обмена", ["True", "False"])
                location_region = st.text_input("Регион, где находится автомобиль")
                number_of_photos = st.number_input("Количество фотографий автомобиля:", min_value=1, max_value=100, value=5)
                up_counter = st.number_input("Количество поднятий объявления:", min_value=1, max_value=100, value=3)

                # Преобразование бинарных признаков
                engine_has_gas = 1 if engine_has_gas == "True" else 0
                has_warranty = 1 if has_warranty == "True" else 0
                is_exchangeable = 1 if is_exchangeable == "True" else 0

                # Добавление всех введенных данных в DataFrame
                input_data = pd.DataFrame({
                    'manufacturer_name': [manufacturer_name],
                    'model_name': [model_name],
                    'transmission': [transmission],
                    'color': [color],
                    'odometer_value': [odometer_value],
                    'year_produced': [year_produced],
                    'engine_fuel': [engine_fuel],
                    'engine_has_gas': [engine_has_gas],
                    'engine_type': [engine_type],
                    'engine_capacity': [engine_capacity],
                    'body_type': [body_type],
                    'has_warranty': [has_warranty],
                    'state': [state],
                    'drivetrain': [drivetrain],
                    'is_exchangeable': [is_exchangeable],
                    'location_region': [location_region],
                    'number_of_photos': [number_of_photos],
                    'up_counter': [up_counter],
                })

                # Преобразуем категориальные признаки в one-hot
                input_data = pd.get_dummies(input_data, drop_first=True)

                # Добавляем недостающие признаки с нулевыми значениями
                missing_columns = set(model_columns) - set(input_data.columns)
                for col in missing_columns:
                    input_data[col] = 0  # Добавляем недостающие признаки с нулевыми значениями

                # Приводим порядок признаков к нужному
                input_data = input_data[model_columns]  # Применяем правильный порядок

                # Получаем предсказание
                if st.button("Получить прогноз стоимости"):
                    prediction = model.predict(input_data)[0]
                    
                    # Форматируем число с правильным разделением тысяч
                    formatted_prediction = "{:,.0f}".format(prediction).replace(",", " ")

                    st.success(f"Оценочная стоимость: **{formatted_prediction} долларов**")
