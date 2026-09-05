import operator
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from tools import search_flights, search_hotels, search_attractions, book_flight
from knowledge import search_knowledge
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import time
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

load_dotenv()
# ── Pydantic-моделі для структурованого виводу ──────────────────


class Plan(BaseModel):
    """План виконання задачі."""
    goal: str = Field(description='Головна ціль задачі')
    steps: list[str] = Field(
        description='Список логічних кроків для досягнення цілі. Кожен крок має бути конкретним.')


class ReplanDecision(BaseModel):
    """Рішення replanner: продовжити, перепланувати або завершити."""
    action: Literal['continue', 'replan', 'finish'] = Field(
        description='continue=наступний крок, replan=змінити план, finish=завершити'
    )
    updated_steps: list[str] | None = Field(
        default=None,
        description='Оновлені кроки (обовʼязково, якщо action=replan)',
    )
    reasoning: str = Field(description='Пояснення вашого рішення')

# ── State графа ─────────────────────────────────────────────────


class PlanExecuteState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: list[str]
    current_step: int
    results: list[str]
    completed: bool


# ── Ініціалізація LLM та інструментів ───────────────────────────
llm = ChatOpenAI(model="google/gemini-3.7-flash",
                 base_url="https://openrouter.ai/api/v1", temperature=0.1)
planner_llm = llm.with_structured_output(Plan)
replanner_llm = llm.with_structured_output(ReplanDecision)

tools = [search_flights, search_hotels,
         search_attractions, search_knowledge, book_flight]

RISKY_TOOLS = {'book_flight'}

tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

conn = sqlite3.connect('agent_state.db', check_same_thread=False)
saver = SqliteSaver(conn)

# ── Вузли ───────────────────────────────────────────────────────


def planner_node(state: PlanExecuteState) -> dict:
    """Генерує початковий план."""
    user_msg = state['messages'][0].content if state['messages'] else ''
    tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    prompt = (
        f'Ти туристичний планувальник. Створи план для запиту: {user_msg}\n'
        f'Розбий на 2-4 конкретних кроків (наприклад: 1. Знайти квитки. 2. Знайти готель. 3. Знайти локації).'
        f'Кожен крок плану повинен вирішуватися ВИКЛЮЧНО наявними інструментами без запитування додаткової інформації.'
        f"доступні інструменти:\n{tool_desc}\n\n"
        f"правила планування:\n"
        f"1. кожен крок плану повинен описувати виклик одного з доступних інструментів.\n"
        f"2. Якщо запит має умови (наприклад, 'якщо квитків немає...'), ігноруй їх під час створення початкового плану. Сплануй лише ідеальний перший сценарій. Переплануванням займеться інший вузол."
    )
    plan = planner_llm.invoke(prompt)
    return {
        'plan': plan.steps,
        'current_step': 0,
        'results': [],
        'completed': False,
        'messages': [AIMessage(content=f'Створено план:\n- ' + '\n- '.join(plan.steps))]
    }


def executor_node(state: PlanExecuteState) -> dict:
    """Виконує один поточний крок плану."""
    step_idx = state.get('current_step', 0)
    plan = state.get('plan', [])

    if step_idx >= len(plan):
        return {'completed': True}

    current_step = plan[step_idx]

    prompt = (
        f'Твоє завдання — виконати поточний крок: "{current_step}"\n'
        f'Результати попередніх кроків (використовуй ці дані, якщо потрібно): {state.get("results", [])}'
    )

    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    result = response.content

    # Якщо LLM викликав tool — виконати
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            if tc['name'] in RISKY_TOOLS:
                print(
                    f"\nАгент намагається виконати ризикову дію: {tc['name']}")

                # ── INTERRUPT: зупинити для підтвердження ────────
                approval = interrupt({
                    'action': tc['name'],
                    'args': tc['args'],
                    'message': (
                        f'Підтвердіть ризикову дію:\n'
                        f'Tool: {tc["name"]}\n'
                        f'Параметри: {tc["args"]}'
                    )
                })

                if isinstance(approval, dict) and approval.get('approved'):
                    tool_fn = tools_by_name.get(tc['name'])
                    tool_result = tool_fn.invoke(tc['args'])
                    result = f'Підтверджено. Дані від {tc["name"]}: {tool_result}'
                else:
                    reason = approval.get("reason", "Відхилено оператором") if isinstance(
                        approval, dict) else "Відхилено"
                    result = f'Скасовано: {reason}'

            else:
                tool_fn = tools_by_name.get(tc['name'])
                if tool_fn:
                    try:
                        tool_result = tool_fn.invoke(tc['args'])
                        result = f'Дані від {tc["name"]}: {tool_result}'
                    except Exception as e:
                        result = f'Помилка виконання {tc["name"]}: {e}'

    return {
        'current_step': step_idx + 1,
        'results': state.get('results', []) + [f'Крок {step_idx+1} ({current_step}) -> {result}'],
        'messages': [AIMessage(content=f'Виконано крок {step_idx+1}: {result}')],
    }


def replanner_node(state: PlanExecuteState) -> dict:
    """Оцінює прогрес та вирішує наступні дії."""
    plan = state.get('plan', [])
    step_idx = state.get('current_step', 0)
    results = state.get('results', [])

    user_msg = state['messages'][0].content if state['messages'] else ''

    remaining = plan[step_idx:] if step_idx < len(plan) else []
    prompt = (
        f'оригінальний запит користувача: {user_msg}\n'
        f'Оціни прогрес виконання:\n'
        f'Початковий план: {plan}\n'
        f'Виконано кроків: {step_idx}/{len(plan)}\n'
        f'Отримані результати: {results}\n'
        f'Залишилось виконати: {remaining}\n\n'
        f'Якщо ціль досягнута або продовжувати немає сенсу — обери finish.\n'
        f'Якщо план актуальний — обери continue.\n'
        f'Якщо дані змінилися (наприклад, немає квитків на ці дати) — обери replan і напиши нові кроки.'
    )
    decision = replanner_llm.invoke(prompt)

    if decision.action == 'finish':
        return {'completed': True, 'messages': [AIMessage(content=f'Завершено. Причина: {decision.reasoning}')]}
    elif decision.action == 'replan' and decision.updated_steps:
        return {'plan': decision.updated_steps, 'current_step': 0, 'messages': [AIMessage(content=f'Перепланування: {decision.reasoning}')]}

    return {}  # continue


def should_end(state: PlanExecuteState) -> Literal['executor', '__end__']:
    if state.get('completed'):
        return '__end__'
    return 'executor'


# Збірка графа
graph = StateGraph(PlanExecuteState)
graph.add_node('planner', planner_node)
graph.add_node('executor', executor_node)
graph.add_node('replanner', replanner_node)

graph.add_edge(START, 'planner')
graph.add_edge('planner', 'executor')
graph.add_edge('executor', 'replanner')
graph.add_conditional_edges('replanner', should_end)

# Компілюємо граф з checkpointer
app_with_memory = graph.compile(checkpointer=saver)

# ── Запуск ──────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "name": "ТЕСТ 1: Проста задача (1 дія)",
            "query": "Знайди готель у Києві на 3 ночі для 2 осіб."
        },
        {
            "name": "ТЕСТ 2: Складний послідовний план (Без перепланування)",
            "query": (
                "Сплануй мою поїздку з Торонто до Ванкувера на 10 жовтня 2026 року для 2 осіб. "
                "Спершу перевір авіаквитки, потім підбери готель на 14 ночей,"
                " і наостанок — знайди природні локації та надай інформацію про місто."
            )
        },
        {
            "name": "ТЕСТ 3: Динамічне перепланування (Replanning)",
            "query": (
                "Сплануй мою поїздку з Торонто до Монреаля на 10 жовтня. "
                "Спершу перевір авіаквитки. "
                "УВАГА: Якщо авіаквитків до Монреаля немає, повністю зміни план: "
                "замість Монреаля знайди квитки з Торонто до Ванкувера та підбери готель у Ванкувері."
            )
        },
        {
            "name": "ТЕСТ 4: Тільки довідкова інформація (RAG)",
            "query": "Які візові правила для поїздки до Коста-Рики з Канади? І що варто брати з собою?"
        }
    ]

    for tc in test_cases:
        print(f"\n{'='*70}")
        print(f"{tc['name']}")
        print(f"ЗАПИТ: {tc['query']}")
        print(f"{'='*70}\n")

        initial_state = {
            'messages': [HumanMessage(content=tc['query'])],
            'plan': [],
            'current_step': 0,
            'results': [],
            'completed': False
        }

        config = {'configurable': {
            'thread_id': f'session-tc-{test_cases.index(tc)}'}}

        for event in app_with_memory.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in event.items():

                if isinstance(node_state, dict):

                    if 'messages' in node_state and node_state['messages']:
                        raw_content = node_state['messages'][-1].content
                        last_msg = raw_content if raw_content else "Дію оцінено (без текстових коментарів)"
                        print(f"[{node_name.upper()}] {last_msg}\n")

                    if node_name == 'replanner' and 'plan' in node_state and node_state.get('current_step') == 0:
                        print(f" План було змінено! Нові кроки:")
                        for i, step in enumerate(node_state['plan'], 1):
                            print(f"      {i}. {step}")
                        print("\n")

        print(f"--- {tc['name']} завершено ---\n")
        time.sleep(1)  # Пауза між тестами, щоб не перевантажити API

    # ТЕСТ 5: Переривання та відновлення стану
    print("\n" + "="*70)
    print("ТЕСТ 5: Переривання та відновлення стану")
    print("="*70 + "\n")

    config_persist = {'configurable': {'thread_id': 'session-interrupted-005'}}
    query_persist = "Сплануй поїздку з Києва до Варшави на 25 жовтня 2026. Знайди квитки, а потім готель."

    initial_state_persist = {
        'messages': [HumanMessage(content=query_persist)],
        'plan': [],
        'current_step': 0,
        'results': [],
        'completed': False
    }

    event_count = 0

    for event in app_with_memory.stream(initial_state_persist, config=config_persist, stream_mode="updates"):
        for node_name, node_state in event.items():
            if isinstance(node_state, dict) and 'messages' in node_state and node_state['messages']:
                last_msg = node_state['messages'][-1].content
                print(
                    f"[{node_name.upper()}] {last_msg if last_msg else 'Дія виконана'}\n")

        event_count += 1

        # симуляція збою, вихід з циклу
        if event_count >= 2:
            print("виконання перервано!\n")
            break

    print("Перезапуск програми... Відновлення стану з SqliteSaver...\n")

    restored_state = app_with_memory.get_state(config_persist)
    print(
        f"Відновлений поточний крок: {restored_state.values.get('current_step')}")
    print(
        f"Зібрані результати до збою: {restored_state.values.get('results')}\n")

    print("Продовження виконання з місця зупинки...\n")

    # передаємо None замість initial_state
    for event in app_with_memory.stream(None, config=config_persist, stream_mode="updates"):
        for node_name, node_state in event.items():
            if isinstance(node_state, dict) and 'messages' in node_state and node_state['messages']:
                last_msg = node_state['messages'][-1].content
                print(
                    f"[{node_name.upper()}] {last_msg if last_msg else 'Дія виконана'}\n")

    print("--- ТЕСТ 5 завершено ---\n")

    # ======================================================================
    # ТЕСТ 6: hitl
    # ======================================================================
    print("\n" + "="*70)
    print("ТЕСТ: Human-in-the-Loop (Бронювання квитків)")
    print("="*70 + "\n")

    config_hitl = {'configurable': {'thread_id': 'hitl-booking-session-001'}}
    query_hitl = (
        "Сплануй поїздку з Торонто до Ванкувера для 2 людей на 10 жовтня 2026. "
        "Спершу перевір квитки, а тоді забронюй їх."
    )

    initial_state_hitl = {
        'messages': [HumanMessage(content=query_hitl)],
        'plan': [],
        'current_step': 0,
        'results': [],
        'completed': False
    }

    for event in app_with_memory.stream(initial_state_hitl, config=config_hitl, stream_mode="updates"):
        for node_name, node_state in event.items():
            if isinstance(node_state, dict) and 'messages' in node_state and node_state['messages']:
                last_msg = node_state['messages'][-1].content
                print(
                    f"[{node_name.upper()}] {last_msg if last_msg else 'Дія виконана'}\n")

    state = app_with_memory.get_state(config_hitl)

    if state.next:
        print("граф зупинено. очікування дії людини (hitl)")

        interrupt_data = state.tasks[0].interrupts[0].value
        print(f"Повідомлення від агента: {interrupt_data['message']}")

        time.sleep(1)

        # user_decision = Command(resume={'approved': True})

        user_reject = Command(resume={
            'approved': False,
            'reason': 'Я передумав, забронюй квиток лише для 1 пасажира.'
        })
        print(
            "людина відхилила ризикову дію. Продовження виконання з новими параметрами...\n")

        for event in app_with_memory.stream(user_reject, config=config_hitl, stream_mode="updates"):
            for node_name, node_state in event.items():
                if isinstance(node_state, dict) and 'messages' in node_state and node_state['messages']:
                    last_msg = node_state['messages'][-1].content
                    print(
                        f"[{node_name.upper()}] {last_msg if last_msg else 'Дія виконана'}\n")

                if node_name == 'replanner' and 'plan' in node_state and node_state.get('current_step') == 0:
                    for i, step in enumerate(node_state['plan'], 1):
                        print(f"      {i}. {step}")
                    print("\n")

    state = app_with_memory.get_state(config_hitl)

    if state.next:
        print("граф зупинено. очікування дії людини (hitl)")
        interrupt_data = state.tasks[0].interrupts[0].value
        print(f"Повідомлення від агента: {interrupt_data['message']}")

        time.sleep(1)

        user_approve = Command(resume={'approved': True})
        print("людина підтвердила ризикову дію. Продовження виконання...\n")

        for event in app_with_memory.stream(user_approve, config=config_hitl, stream_mode="updates"):
            for node_name, node_state in event.items():
                if isinstance(node_state, dict) and 'messages' in node_state and node_state['messages']:
                    last_msg = node_state['messages'][-1].content
                    print(
                        f"[{node_name.upper()}] {last_msg if last_msg else 'Дія виконана'}\n")

    print("\n--- ТЕСТ 6 завершено ---")
