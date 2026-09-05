import asyncio
import os
import sqlite3
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import interrupt, Command

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="google/gemini-3.7-flash",
                 base_url="https://openrouter.ai/api/v1", temperature=0.1)


class HITLState(TypedDict):
    messages: Annotated[list, operator.add]


RISKY_TOOLS = {'update_ticket_status'}


async def safe_agent_node(state: HITLState) -> dict:
    global mcp_tools

    agent_llm = llm.bind_tools(mcp_tools)
    response = await agent_llm.ainvoke([
        SystemMessage(
            content="Ти агент підтримки. Використовуй інструменти MCP.")
    ] + state['messages'])

    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        return {'messages': [response]}

    # Обробляємо виклик інструменту
    tc = response.tool_calls[0]
    tool_name = tc['name']
    tool_args = tc['args']
    tool_fn = next((t for t in mcp_tools if t.name == tool_name), None)

    if not tool_fn:
        return {'messages': [AIMessage(content="Інструмент не знайдено.")]}

    # ── HITL: ПЕРЕХОПЛЕННЯ РИЗИКОВИХ ДІЙ ──
    if tool_name in RISKY_TOOLS:
        print(
            f"\n[УВАГА] Агент намагається виконати ризикову дію: {tool_name}")

        # Граф зупиняється тут і чекає на Command(resume=...)
        decision = interrupt({
            'message': f'Потрібне підтвердження для {tool_name}',
            'tool': tool_name,
            'args': tool_args
        })

        action = decision.get('action')
        if action == 'reject':
            reason = decision.get('reason', 'Дію відхилено адміністратором.')
            return {'messages': [AIMessage(content=f"Відхилено: {reason}")]}

        elif action == 'edit':
            print(
                f"[*] Адміністратор змінив аргументи з {tool_args} на {decision.get('args')}")
            tool_args.update(decision.get('args', {}))

        elif action == 'approve':
            print("[*] Адміністратор підтвердив дію.")

    tool_res = await tool_fn.ainvoke(tool_args)
    return {'messages': [AIMessage(content=f"Результат виконання {tool_name}: {tool_res}")]}


async def main():
    global mcp_tools
    print("Підключення до MCP сервера...")
    client = MultiServerMCPClient({
        'support': {
            'command': 'python',
            'args': [os.path.abspath('mcp_server.py')],
            'transport': 'stdio',
        },
    })
    mcp_tools = await client.get_tools()

    # Збірка графа
    g = StateGraph(HITLState)
    g.add_node('agent', safe_agent_node)
    g.add_edge(START, 'agent')
    g.add_edge('agent', END)

    async with AsyncSqliteSaver.from_conn_string("hitl_state.db") as saver:
        app = g.compile(checkpointer=saver)

        query = "Зміни статус тікету TKT-001 на 'closed', бо питання вирішено."
        initial_state = {'messages': [HumanMessage(content=query)]}

        # ── СЦЕНАРІЙ 1: ЗАТВЕРДЖЕННЯ (APPROVE) ──
        print("\n" + "="*50 + "\nСЦЕНАРІЙ 1: APPROVE\n" + "="*50)
        config1 = {'configurable': {'thread_id': 'hitl-approve'}}
        # Зупиниться на interrupt
        await app.ainvoke(initial_state, config=config1)

        result1 = await app.ainvoke(Command(resume={'action': 'approve'}), config=config1)
        print(f"Фінальна відповідь: {result1['messages'][-1].content}")

        # ── СЦЕНАРІЙ 2: ВІДХИЛЕННЯ (REJECT) ──
        print("\n" + "="*50 + "\nСЦЕНАРІЙ 2: REJECT\n" + "="*50)
        config2 = {'configurable': {'thread_id': 'hitl-reject'}}
        # Зупиниться на interrupt
        await app.ainvoke(initial_state, config=config2)

        result2 = await app.ainvoke(Command(resume={
            'action': 'reject', 'reason': 'Заборонено закривати тікети без ревю.'
        }), config=config2)
        print(f"Фінальна відповідь: {result2['messages'][-1].content}")

        # ── СЦЕНАРІЙ 3: РЕДАГУВАННЯ (EDIT) ──
        print("\n" + "="*50 + "\nСЦЕНАРІЙ 3: EDIT\n" + "="*50)
        config3 = {'configurable': {'thread_id': 'hitl-edit'}}
        # Зупиниться на interrupt
        await app.ainvoke(initial_state, config=config3)

        result3 = await app.ainvoke(Command(resume={
            'action': 'edit', 'args': {'new_status': 'in_progress', 'reason': 'edited by admin'}
        }), config=config3)
        print(f"Фінальна відповідь: {result3['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
