# EggsDetector

EggsDetector — это приложение на Python, разработанное для автоматической сегментации и подсчета яиц на изображениях. Программа также отображает карту локальных максимумов яркости с изолиниями, что может использоваться для анализа текстуры или интенсивности изображений.

## 📌 Основные возможности
- **Загрузка изображений:** Поддержка форматов `.jpg`, `.jpeg`, `.png`, `.bmp`.
- **Предобработка изображения:**
  - Преобразование в LAB цветовое пространство.
  - Использование CLAHE (контрастная адаптивная гистограмма).
  - Размытие с использованием медианного фильтра.
  - Морфологическое расширение и закрытие.
- **Сегментация яиц:**
  - Автоматический подсчет белых и красных яиц.
  - Построение объединенной маски (белые и красные области).
- **Карта яркости:**
  - Построение карты с локальными максимумами яркости.
  - Визуализация изолиний яркости на изображении.
  - Яркие области выделяются жирными красными точками.
- **Сохранение результатов:**
  - Предобработанное изображение.
  - Объединённая маска яиц.
  - Карта яркости с изолиниями и максимумами.

---

## 📋 Использование
1. **Загрузите изображение** через кнопку "Загрузить изображение".
2. Нажмите "Предобработка" для обработки изображения.
3. Нажмите "Подсчитать яйца" для сегментации белых и красных яиц.
4. Проверьте результаты в центральном окне (объединенная маска).
5. Карту яркости с изолиниями смотрите в правом окне.
6. Для сохранения результатов введите имя файла и нажмите "Скачать результат".

---

## 📝 Используемые библиотеки
- `numpy`
- `opencv-python`
- `opencv-python-headless`
- `scipy`
- `Pillow`
- `matplotlib`
- `tkinter` (встроенная в стандартную библиотеку Python)

---

## 🔧 Как улучшить
- Добавить больше фильтров для улучшения точности сегментации.
- Улучшить визуализацию карты яркости.
- Улучшить интерфейс (например, добавить выбор различных цветовых карт).

---

## 📜 Лицензия
Этот проект распространяется под лицензией MIT.

