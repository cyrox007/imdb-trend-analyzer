from app import Kinovanga

# Создаём экземпляр
kino = Kinovanga()

# Предсказание без обучения (на предобученной модели)
# Если модель сохранена: kino = Kinovanga(model_path='model.joblib')

rating = kino.predict_rating(
    title="The Batman: Part II",
    director="Мэтт Ривз",
    year=2026,
    runtime=150,
    description="Бэтмен против Смерти. Темный рыцарь в кризисе."
)

print(f"Прогнозируемый рейтинг: {rating}")