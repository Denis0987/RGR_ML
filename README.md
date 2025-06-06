# РГР: Прогнозирование стоимости автомобилей с помощью ML

## О проекте
Этот репозиторий содержит все исходные файлы для расчётно-графической работы (РГР) по дисциплине «Машинное обучение и большие данные» (МО, ОмГТУ, 2025). Тема работы:  
**«Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных»**.

В рамках проекта выполнены следующие ключевые этапы:
1. **Сбор и предобработка данных (EDA).**  
   - Ознакомление с набором данных о ценах на недвижимость (df_cars).  
   - Очистка, заполнение пропусков, преобразование категориальных признаков.  
   - Построение базовых визуализаций (гистограммы, ящиковые диаграммы, тепловые карты корреляций, pairplot).

2. **Реализованные модели**  
   - **Полиномиальная регрессия** (Scikit-learn).  
   - **Gradient Boosting Regressor** (Scikit-learn).  
   - **CatBoost Regressor** ( CatBoost ).  
   - **Bagging Regressor** (Scikit-learn).  
   - **Stacking Regressor** (Scikit-learn).  
   - **MLPRegressor (нейронная сеть)** (Scikit-learn).

   Для каждой модели вычислены базовые метрики качества:
   - Коэффициент детерминации (R²).  
   - Средняя абсолютная ошибка (MAE).  
   - Корень из среднеквадратичной ошибки (RMSE).  


3. **Сериализация моделей.**  
   - Модели Scikit-learn сохранены с помощью `pickle` как `.pkl`.  
   - Модель CatBoost сохранена как `.json` (внутренний формат CatBoost).  
   - Все файлы моделей лежат в папке `models/` и доступны для загрузки при инференсе.

4. **Веб-интерфейс (Streamlit).**  
   Веб-приложение состоит из четырёх страниц (многостраничное приложение Streamlit). Оно даёт возможность:
   - **Страница 1 (General):** Информация о разработчике (ФИО, группа, тема РГР).  
   - **Страница 2 (Dataset):** Описание набора данных (предметная область, список признаков, этапы EDA).  
   - **Страница 3 (DashBoard):** Визуализации зависимостей в данных (минимум 4 разных графика с Matplotlib/Seaborn).  
   - **Страница 4 (Prediction):** Интерфейс для инференса:
     - Загрузка CSV-файла с новыми объектами недвижимости ➔ массовое предсказание.  
     - Ручной ввод признаков объекта ➔ единичное предсказание.  
     - Вывод результата в понятном формате (например, цена в доллары США с разделителем тысяч).  
     - Отображение первых 10 строк загруженных данных с колонкой `predicted_price`. 
   - на **Streamlit Cloud** с развернутым веб-приложением: https://lk375mrkvrjqpaebqcljep.streamlit.app/

---

## Структура репозитория
```text
RGR_ML/  
├─ README.md                       ← Документация проекта (текущий файл)  
├─ requirements.txt                ← Список Python-зависимостей для pip install   
├─ df_cars.csv                     ← Оригинальный CSV-набор 
├─ photo_2025-06-07_00-18-02.jpg   ← Фотография студента
│    
│  
├─ models/                         ← Сохранённые модели (pickle, CatBoost .json)  
│   ├─ poly_model.pkl               ← Полиномиальная регрессия  
│   ├─ boosting_model.pkl           ← Gradient Boosting (Sklearn)  
│   ├─ catboost_model.json          ← CatBoost Regressor  
│   ├─ bagging_model.pkl            ← Bagging Regressor  
│   ├─ stacking_model.pkl           ← Stacking Regressor  
│   └─ mlp_model.pkl                ← MLPRegressor   
│  
├─ figures/                        ← Графики и таблицы метрик  
│   ├─ corr_matrix_all.png                      ← Тепловая карта корреляций для всех признаков 
│   ├─ corr_matrix.png                          ← Тепловая карта корреляций только для важных для предсказания признаков
│   ├─ scatter_year_produced_price_usd.png      ← Рассеяние year_produced vs price_usd  
│   ├─ boxplot_car_age.png                      ← Boxplot: car_age  
│   ├─ pairplot.png                             ← Pairplot ключевых признаков       
│   ├─ polynomial_metrics_table.png             ← Таблица метрик (Poly)          
│   ├─ boosting_metrics_table.png               ← Таблица метрик (GBR)          
│   ├─ catboost_metrics_table.png               ← Таблица метрик (CatBoost)          
│   ├─ bagging_metrics_table.png                ← Таблица метрик (Bagging)          
│   ├─ stacking_metrics_table.png               ← Таблица метрик (Stacking)                
│   └─ mlp_metrics_table.png                    ← Таблица метрик (MLP)  
│  
├─ pages/                          ← Страницы многостраничного Streamlit-приложения  
│   ├─ 01_AboutDeveloper.py         ← Страница с информацией о разработчике  
│   ├─ 02_DatasetInfo.py            ← Страница с описанием набора данных и EDA  
│   ├─ 03_Visualizations.py         ← Страница с визуализациями (Matplotlib, Seaborn)  
│   └─ 04_Prediction.py             ← Страница с интерфейсом для предсказания  
│  
└─app.py                          ← Точка входа для Streamlit: объединяет страницы через sidebar  
                        
    
