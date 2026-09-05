import operator
import sqlite3
import time
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from tools_legacy import search_flights, search_hotels, search_attractions, book_flight
from knowledge import search_knowledge
from trajectory_logger import TrajectoryLogger

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="google/gemini-3.7-flash",
                 base_url="https://openrouter.ai/api/v1", temperature=0.1)
logger = TrajectoryLogger()


class MASState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    plan: list[str]
    current_step: int
    results: list[str]
    step_count: int
    start_time: float
    completed: bool


def log_agent_step(agent_name: str, step_num: int, node: str, input_data: str, output_data: str):
    logger.log_step(
        step_num=step_num,
        node=f"[{agent_name}] {node}",
        input_data=input_data,
        output_data=output_data
    )


class RouteDecision(BaseModel):
    action: Literal['planner', 'booking', 'researcher', 'general'] = Field(
        description='Цільовий агент або "general" для нерозпізнаних запитів'
    )
    reasoning: str = Field(description='Пояснення вибору')


supervisor_llm = llm.with_structured_output(RouteDecision)

SUPERVISOR_SYSTEM = """Ти — супервізор туристичного агентства. Направляй запит до наступних агентів:
- planner: комплексне планування подорожі, пошук квитків, готелів, локацій
- booking: безпосереднє бронювання, оплата, підтвердження квитків
- researcher: довідкові питання — візові правила, faq, туристичні політики, багаж
- general: вітання, нерозпізнані запити"""


def supervisor_node(state: MASState) -> dict:
    user_msg = state['messages'][-1].content
    decision = supervisor_llm.invoke([
        ('system', SUPERVISOR_SYSTEM),
        ('user', user_msg),
    ])

    log_agent_step('supervisor', state.get('step_count', 0), 'route',
                   user_msg, f'→ {decision.action}: {decision.reasoning}')

    return {
        'current_agent': decision.action,
        'step_count': state.get('step_count', 0) + 1,
    }

# --- Агент 1: Planner (Plan-and-Execute з ДЗ2) ---


class Plan(BaseModel):
    steps: list[str] = Field(description="Кроки для виконання")


def planner_agent(state: MASState) -> dict:
    """Використовує search_flights, search_hotels, search_attractions"""
    user_msg = state['messages'][0].content
    planner = llm.with_structured_output(Plan)

    plan = planner.invoke(
        f"Створи план 1-3 кроки для планування подорожі: {user_msg}. Використовуй тільки доступні інструменти.")

    tool_executor = llm.bind_tools(
        [search_flights, search_hotels, search_attractions])
    response = tool_executor.invoke([HumanMessage(content=plan.steps[0])])

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tc = response.tool_calls[0]
        tool_map = {
            "search_flights": search_flights,
            "search_hotels": search_hotels,
            "search_attractions": search_attractions
        }
        tool_fn = tool_map.get(tc['name'])
        result = tool_fn.invoke(
            tc['args']) if tool_fn else "Інструмент не знайдено"
    else:
        result = response.content

    final_msg = AIMessage(content=f"План виконано. Результат: {result}")

    log_agent_step('planner', state.get('step_count', 0),
                   'plan_and_execute', str(plan.steps), str(result))

    return {'messages': [final_msg], 'completed': True, 'step_count': state.get('step_count', 0) + 1}


# --- Агент 2: Booking (ReAct з max_steps + timeout з ДЗ1) ---
MAX_STEPS = 3
TIMEOUT_SECONDS = 10


def booking_agent(state: MASState) -> dict:
    """Використовує book_flight з лімітами"""
    step = state.get('step_count', 0)
    start_time = state.get('start_time', time.time())

    if time.time() - start_time > TIMEOUT_SECONDS or step >= MAX_STEPS:
        msg = AIMessage(content="Досягнуто ліміт часу або кроків.")
        log_agent_step('booking', step, 'halt',
                       'timeout/max_steps', msg.content)
        return {'messages': [msg], 'completed': True}

    agent_llm = llm.bind_tools([book_flight])
    response = agent_llm.invoke(state['messages'])

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tc = response.tool_calls[0]
        tool_res = book_flight.invoke(tc['args'])
        final_res = f"Статус бронювання: {tool_res}"
    else:
        final_res = response.content

    msg = AIMessage(content=final_res)
    log_agent_step('booking', step, 'react_step',
                   state['messages'][-1].content, final_res)

    return {'messages': [msg], 'step_count': step + 1, 'completed': True}

# --- Агент 3: Researcher (Agentic RAG з ДЗ2) ---


def researcher_agent(state: MASState) -> dict:
    """Використовує ChromaDB через search_knowledge"""
    agent_llm = llm.bind_tools([search_knowledge])
    response = agent_llm.invoke([
        SystemMessage(
            content="Ти дослідник туристичної бази знань. Завжди використовуй tool search_knowledge для відповідей.")
    ] + state['messages'])

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tc = response.tool_calls[0]
        rag_result = search_knowledge.invoke(tc['args'])
        synthesis = llm.invoke(
            f"Сформуй фінальну відповідь клієнту на основі цього: {rag_result}")
        final_msg = synthesis.content
    else:
        final_msg = response.content

    log_agent_step('researcher', state.get('step_count', 0),
                   'rag_search', state['messages'][-1].content, final_msg)

    return {'messages': [AIMessage(content=final_msg)], 'completed': True, 'step_count': state.get('step_count', 0) + 1}


def general_agent(state: MASState) -> dict:
    return {'messages': [AIMessage(content="Я загальний асистент. Уточніть, чим можу допомогти щодо вашої подорожі?")], 'completed': True}


def route(state: MASState) -> Literal['planner', 'booking', 'researcher', 'general', '__end__']:
    if state.get('completed'):
        return '__end__'
    return state.get('current_agent', 'general')


g = StateGraph(MASState)
g.add_node('supervisor', supervisor_node)
g.add_node('planner', planner_agent)
g.add_node('booking', booking_agent)
g.add_node('researcher', researcher_agent)
g.add_node('general', general_agent)

g.add_edge(START, 'supervisor')
g.add_conditional_edges('supervisor', route)
g.add_edge('planner', END)
g.add_edge('booking', END)
g.add_edge('researcher', END)
g.add_edge('general', END)

conn = sqlite3.connect('agent_state.db', check_same_thread=False)
saver = SqliteSaver(conn)
app = g.compile(checkpointer=saver)

if __name__ == "__main__":
    queries = [
        ('1', 'Сплануй поїздку з Торонто до Ванкувера на 10 жовтня 2026, знайди готель на 14 ночей.'),
        ('2', 'Будь ласка, забронюй рейс AC101 для 2 пасажирів, ціна $350.'),
        ('3', 'Які візові правила для поїздки до Коста-Рики з Канади?')
    ]

    for tid, query in queries:
        print(f"\n{'='*70}\nЗАПИТ {tid}: {query}\n{'-'*70}")
        config = {'configurable': {'thread_id': f'demo-mas-{tid}'}}

        initial_state = {
            'messages': [HumanMessage(content=query)],
            'current_agent': '',
            'plan': [], 'current_step': 0, 'results': [],
            'step_count': 0, 'start_time': time.time(),
            'completed': False
        }

        result = app.invoke(initial_state, config=config)

        print(
            f"[{result['current_agent'].upper()}] Відповідь: {result['messages'][-1].content}")

    logger.save('trajectory_mas.json')
