import chromadb
from chromadb.config import Settings
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="google/gemini-3.7-flash",
                 base_url="https://openrouter.ai/api/v1", temperature=0.1)

# ── Створення бази знань (ChromaDB) ──────────────────────────────
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
knowledge_base = chroma_client.create_collection('travel_knowledge')

# Довідкові документи
# documents = [
#     "Ванкувер — ідеальне місто для велопрогулянок. Маршрут Seawall у Стенлі-Парку має довжину близько 28 км і пропонує чудові краєвиди. Найкращий час для активного відпочинку на природі — з травня по вересень.",
#     "Мальпаїс у Коста-Риці відомий своїми пляжами для серфінгу, геологічними пам'ятками та тропічними лісами. Для оренди автомобіля та подорожей країною туристам з Канади віза не потрібна на термін до 90 днів, але необхідний зворотний квиток.",
#     "Подорож на авто з Торонто до Ванкувера займає близько 40-50 годин чистого часу за кермом. Рекомендується розбити маршрут на 10-14 днів, щоб мати змогу працювати віддалено та відвідувати національні парки.",
#     "Громадяни Канади можуть подорожувати до країн Шенгенської зони без візи на термін до 90 днів. Проте планується введення електронного дозволу ETIAS, який потрібно буде оформлювати перед вильотом.",
#     "Ванкуверська художня галерея часто проводить виставки класичного європейського мистецтва. Картини Вінсента ван Гога (наприклад, натюрморти з квітами) періодично з'являються тут у рамках тимчасових експозицій.",
#     "Для поціновувачів класичної музики працює Vancouver Opera, яка регулярно ставить твори Моцарта. Квитки на такі вистави, як 'Чарівна флейта', зазвичай розкуповують за кілька місяців, тому їх варто бронювати заздалегідь.",
#     "Канадська кулінарна сцена дуже різноманітна. На західному узбережжі популярні страви з морепродуктів (салати з крабом), а у великих містах можна знайти ресторани з авторськими соусами, італійською бурратою та іншими делікатесами.",
#     "Для подорожей у тропічні країни рекомендується завжди мати при собі електроліти для гідратації через високу вологість та спеку. Також корисно оформити медичне страхування для активного відпочинку."
# ]

documents = [
    "Vancouver is an ideal city for cycling. The Seawall route in Stanley Park is about 28 km long and offers wonderful views. The best time for outdoor activities is from May to September.",
    "Malpaís in Costa Rica is known for its surfing beaches, geological landmarks, and tropical forests. For renting a car and traveling around the country, tourists from Canada do not need a visa for up to 90 days, but a return ticket is required.",
    "A road trip from Toronto to Vancouver takes about 40-50 hours of pure driving time. It is recommended to break the route into 10-14 days to be able to work remotely and visit national parks.",
    "Canadian citizens can travel to Schengen Area countries without a visa for up to 90 days. However, the introduction of the ETIAS electronic authorization is planned, which will need to be obtained before departure.",
    "The Vancouver Art Gallery frequently hosts exhibitions of classical European art. Vincent van Gogh's paintings (for example, floral still lifes) periodically appear here as part of temporary exhibitions.",
    "For classical music lovers, there is the Vancouver Opera, which regularly stages works by Mozart. Tickets for performances like 'The Magic Flute' are usually sold out several months in advance, so they should be booked early.",
    "The Canadian culinary scene is very diverse. Seafood dishes (crab salads) are popular on the West Coast, while in major cities you can find restaurants with signature sauces, Italian burrata, and other delicacies.",
    "For traveling to tropical countries, it is recommended to always carry electrolytes for hydration due to the high humidity and heat. It is also useful to get medical insurance for active recreation."
]

doc_ids = [f'doc_{i}' for i in range(len(documents))]
knowledge_base.add(documents=documents, ids=doc_ids)

# ── RAG Tool ─────────────────────────────────────────────────────


@tool
def search_knowledge(query: str) -> str:
    """Пошук інформації у туристичній базі знань англійською мовою.

    Використовуйте цей інструмент, коли користувач запитує довідкову 
    інформацію: візові правила, поради щодо 
    мистецтва (виставки, опера), їжі або погоди.
    Ваш запит (query) має бути максимально короткимб англійською мовою та
    обов'язково включати назви країн, міст або ключові 
    терміни (наприклад: "візові правила Коста-Рика Канада"
    """
    results = knowledge_base.query(query_texts=[query], n_results=3)

    if not results['documents'] or not results['documents'][0]:
        return "В базі знань немає інформації за цим запитом."

    # Об'єднуємо знайдені фрагменти у сирий контекст
    raw_context = '\n---\n'.join(results['documents'][0])

    # постобробка
    synthesis_prompt = (
        f"Ти — аналітик бази знань. Твоє завдання — відповісти на запит, "
        f"використовуючи ТІЛЬКИ наданий контекст.\n\n"
        f"запит користувача: {query}\n\n"
        f"знайдений контекст:\n{raw_context}\n\n"
        f"правила:\n"
        f"1. Обери найбільш відповідні фрагменти з контексту що відповідають запиту\n"
        f"2. Ігноруй будь-яку інформацію з контексту, яка не стосується запиту (наприклад, інформацію про інші міста чи країни).\n"
    )
    print(
        f"Синтезую відповідь на основі знайденого контексту {query}... {synthesis_prompt}")

    # Використовуємо базову LLM для генерації відповіді що базується на знайденому контексті
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    print(f"Запит: {query}")
    print(f"Синтезована відповідь: {response.content}")

    return response.content
