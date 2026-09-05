from datetime import datetime, date
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Інструмент для пошуку авіаквитків


class FlightSearchInput(BaseModel):
    """Параметри для пошуку авіаквитків."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    origin: str = Field(
        description='Місто відправлення (наприклад, "Toronto")')
    destination: str = Field(
        description='Місто призначення (наприклад, "Kyiv")')
    numberOfTravelers: int = Field(description='Кількість пасажирів',
                                   gt=0,
                                   le=10)
    dateOfDeparture: date = Field(
        description='Дата вильоту у форматі YYYY-MM-DD (наприклад, "2026-10-01")',
        gt=date.today()
    )
    updated: date = Field(
        default_factory=date.today,
        description='Дата останнього оновлення пошуку'
    )

    @field_validator('origin', 'destination')
    @classmethod
    def city_length(cls, v: str) -> str:
        if not v or len(v.strip()) < 3:
            raise ValueError('Назва міста повинна містити мінімум 3 символи')
        return v.strip().capitalize()

    @model_validator(mode='after')
    def check_origin_destination(self) -> 'FlightSearchInput':
        if self.origin == self.destination:
            raise ValueError(
                'Місто відправлення та призначення не можуть збігатися')
        return self


@tool
def search_flights(flight: FlightSearchInput) -> str:
    """Шукає доступні авіаквитки між двома містами на вказану дату.

    Використовуйте цей інструмент, коли користувач запитує про перельоти,
    квитки на літак або рейси з одного міста в інше.
    """
    mock_db = {
        ("Toronto", "Vancouver"):
            f"Рейс AC101: {flight.origin} -> {flight.destination}, Дата: {flight.dateOfDeparture}, Кількість пасажирів: {flight.numberOfTravelers}, Оновлено: {flight.updated}, Ціна: $350 CAD, Тривалість: 5 год 10 хв",
        ("Kyiv", "Warsaw"):
            f"Рейс LO753: {flight.origin} -> {flight.destination}, Дата: {flight.dateOfDeparture}, Кількість пасажирів: {flight.numberOfTravelers}, Оновлено: {flight.updated}, Ціна: $80 USD, Тривалість: 1 год 30 хв"
    }
    return mock_db.get(
        (flight.origin, flight.destination),
        f"Рейсів для маршруту {flight.origin} -> {flight.destination} на {flight.dateOfDeparture} не знайдено. Спробуйте інші дати.")


# Інструмент для пошуку готелів

class HotelSearchInput(BaseModel):
    """Параметри для пошуку готелів."""
    city: str = Field(description='Місто, де потрібно знайти готель')
    nights: int = Field(
        default=1, description='Кількість ночей для бронювання (від 1 до 30)', gt=0, le=30)
    guests: int = Field(
        default=1, description='Кількість гостей (від 1 до 10)', gt=0, le=10)
    updated: date = Field(
        default_factory=date.today,
        description='Дата останнього оновлення пошуку'
    )


@tool
def search_hotels(hotel: HotelSearchInput) -> str:
    """Шукає готелі в обраному місті.

    Використовуйте цей інструмент, коли користувачу потрібне житло,
    готелі або апартаменти для ночівлі.
    """
    base_price = 120 if hotel.city.capitalize() == "Vancouver" else 90
    total = base_price * hotel.nights * hotel.guests
    return f"Готель 'Центральний' у {hotel.city.capitalize()}: {hotel.nights} ночей для {hotel.guests} гостей. Оновлено: {hotel.updated}. Загальна вартість: ${total}."


# Інструмент для пошуку цікавих місць (Attractions)

class AttractionSearchInput(BaseModel):
    """Параметри для пошуку цікавих місць."""
    city: str = Field(description='Місто, де шукати локації')
    category: str = Field(
        default="all",
        description='Категорія: "nature" (природа), "museum" (музеї), "food" (їжа), "all" (все)'
    )
    updated: date = Field(
        default_factory=date.today,
        description='Дата останнього оновлення пошуку'
    )

    @field_validator('category')
    @classmethod
    def valid_category(cls, v: str) -> str:
        allowed = ["nature", "museum", "food", "all"]
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f'Категорія має бути однією з: {", ".join(allowed)}')
        return v_lower


@tool
def search_attractions(attraction: AttractionSearchInput) -> str:
    """Повертає список цікавих місць для відвідування в місті.

    Використовуйте цей інструмент, коли користувач запитує що подивитися,
    куди піти або які є туристичні атракції.
    """
    db = {
        "Vancouver": {
            "nature": "Стенлі-Парк (Stanley Park), Гора Граус (Grouse Mountain)",
            "museum": "Музей антропології (Museum of Anthropology)",
            "food": "Ринок Гренвілл-Айленд (Granville Island Public Market)"
        },
        "Toronto": {
            "nature": "Торонтські острови (Toronto Islands)",
            "museum": "Королівський музей Онтаріо (ROM)",
            "food": "Ринок Святого Лаврентія (St. Lawrence Market)"
        }
    }

    city_caps = attraction.city.capitalize()
    if city_caps not in db:
        return f"Топ-локації у {city_caps}: Головна площа, Старе місто, Центральний парк. Оновлено: {attraction.updated}."

    if attraction.category == "all":
        items = [f"{k}: {v}" for k, v in db[city_caps].items()]
        return f"Локації у {city_caps} (оновлено: {attraction.updated}):\n" + "\n".join(items)

    return f"{attraction.category.capitalize()} у {city_caps} (оновлено: {attraction.updated}): {db[city_caps].get(attraction.category, 'Немає даних')}"


@tool
def book_flight(flight_number: str, passengers: int, price: str) -> str:
    """Забронювати авіаквитки (ризикована дія — потребує підтвердження).

    Використовуйте цей інструмент лише тоді, коли користувач прямо
    попросив забронювати, купити або оплатити квитки.
    """
    print(
        f"Бронювання рейсу {flight_number} для {passengers} пасажирів. Списано: {price}")
    return f"УСПІХ: Заброньовано рейс {flight_number} для {passengers} пасажирів. Списано: {price}"

# тести


if __name__ == "__main__":
    print("--- Тестування пошуку авіаквитків ---")
    try:
        FlightSearchInput(origin="Toronto",
                          destination="Toronto",
                          numberOfTravelers=2,
                          dateOfDeparture="2026-10-01")
    except Exception as e:
        print(f"Очікувана помилка валідації (однакові міста): {e}")

    try:
        FlightSearchInput(origin="Toronto",
                          destination="Vancouver",
                          numberOfTravelers=2,
                          dateOfDeparture="01-10-2026")
    except Exception as e:
        print(f"Очікувана помилка валідації (формат дати): {e}\n")

    print(search_flights.invoke({
        "flight": {
            "origin": "Toronto",
            "destination": "Vancouver",
            "numberOfTravelers": 2,
            "dateOfDeparture": "2026-10-01"
        }
    }))

    print("\n--- Тестування пошуку готелів ---")
    try:
        HotelSearchInput(city="Vancouver", nights=40)
    except Exception as e:
        print(f"Очікувана помилка валідації (ночі > 30): {e}\n")

    print(search_hotels.invoke({
        "hotel": {
            "city": "Vancouver",
            "nights": 14,
            "guests": 2
        }
    }))

    print("\n--- Тестування пошуку атракцій ---")
    try:
        AttractionSearchInput(city="Vancouver", category="space")
    except Exception as e:
        print(f"Очікувана помилка валідації (невірна категорія): {e}\n")

    print(search_attractions.invoke({
        "attraction": {
            "city": "Vancouver",
            "category": "nature"
        }
    }))
