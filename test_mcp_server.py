"""Unit tests для MCP Server з використанням pytest та підпроцесу stdio.
Запуск: pytest -v test_mcp_server.py
"""
import json
import sys
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_mcp_server_full_integration():
    """Тестування всіх інструментів, ресурсів та промптів через транспортний рівень."""
    # Піднімаємо справжній сервер через stdio, як у production
    params = StdioServerParameters(
        command=sys.executable, args=["mcp_server.py"])

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # Helper для зручного виклику інструментів і парсингу JSON з відповіді
        async def call_tool(name: str, args: dict) -> dict:
            result = await session.call_tool(name, args)
            # MCP ClientSession загортає відповідь у список об'єктів
            return json.loads(result.content[0].text)

        # ── 1. list_tools ──
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert {'get_ticket', 'update_ticket_status', 'search_tickets',
                'get_customer', 'get_ticket_summary'}.issubset(names)

        # ── 2. get_ticket (існує) ──
        data = await call_tool('get_ticket', {'ticket_id': 'TKT-001'})
        assert data['id'] == 'TKT-001'
        assert 'subject' in data

        # ── 3. get_ticket (не знайдено) ──
        data = await call_tool('get_ticket', {'ticket_id': 'TKT-999'})
        assert 'error' in data

        # ── 4. update_ticket_status (валідний статус) ──
        data = await call_tool('update_ticket_status', {
            'ticket_id': 'TKT-001', 'new_status': 'resolved', 'reason': 'test'
        })
        assert data.get('updated') == 'TKT-001'
        assert data.get('new_status') == 'resolved'

        # ── 5. search_tickets ──
        data = await call_tool('search_tickets', {'customer_id': 'C-100'})
        assert 'TKT-001' in data

        # ── 6. get_customer ──
        data = await call_tool('get_customer', {'customer_id': 'C-101'})
        assert data['name'] == 'Марія Коваленко'

        # ── 7. get_ticket_summary ──
        data = await call_tool('get_ticket_summary', {})
        assert 'total' in data
        assert 'active' in data

        # ── 8. list_resources ──
        resources = await session.list_resources()
        assert any('faq' in str(r.uri) for r in resources.resources)

        # ── 9. list_prompts ──
        prompts = await session.list_prompts()
        assert 'support_reply' in [p.name for p in prompts.prompts]
