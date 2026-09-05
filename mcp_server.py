"""MCP Server для Travel Support. Запуск: python mcp_server.py"""
import json
import re
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name='travel_support_server',
    instructions='Сервер з ресурсами підтримки клієнтів: тікети, клієнти, FAQ та політики.',
)

# Mock data
TICKETS = {
    'TKT-001': {'customer_id': 'C-100', 'subject': 'Помилка бронювання готелю у Ванкувері',
                'status': 'open', 'priority': 'high', 'category': 'booking'},
    'TKT-002': {'customer_id': 'C-101', 'subject': 'Квитки Торонто-Ванкувер не надійшли на пошту',
                'status': 'in_progress', 'priority': 'medium', 'category': 'tech'},
}
CUSTOMERS = {
    'C-100': {'name': 'Олег Петренко', 'tier': 'gold', 'email': 'oleh@example.com'},
    'C-101': {'name': 'Марія Коваленко', 'tier': 'silver', 'email': 'maria@example.com'},
}

# ── Tools ──


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Отримати деталі тікета за ID."""
    if not re.match(r'^TKT-\d{3}$', ticket_id):
        return json.dumps({'error': 'Недійсний формат ticket_id. Очікується TKT-XXX.'}, ensure_ascii=False)

    t = TICKETS.get(ticket_id)
    if not t:
        return json.dumps({'error': f'Тікет {ticket_id} не знайдено'}, ensure_ascii=False)
    return json.dumps({'id': ticket_id, **t}, ensure_ascii=False)


@mcp.tool()
def update_ticket_status(ticket_id: str, new_status: str, reason: str = '') -> str:
    """Оновити статус тікета."""
    valid = {'open', 'in_progress', 'resolved', 'closed'}
    if ticket_id not in TICKETS:
        return json.dumps({'error': f'Тікет {ticket_id} не знайдено'}, ensure_ascii=False)
    if new_status not in valid:
        return json.dumps({'error': f'Недійсний статус. Дозволені: {sorted(valid)}'}, ensure_ascii=False)

    old = TICKETS[ticket_id]['status']
    TICKETS[ticket_id]['status'] = new_status
    return json.dumps({
        'updated': ticket_id, 'old_status': old, 'new_status': new_status,
        'reason': reason, 'timestamp': datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False)


@mcp.tool()
def search_tickets(customer_id: str) -> str:
    """Знайти всі тікети конкретного клієнта."""
    if not re.match(r'^C-\d{3}$', customer_id):
        return json.dumps({'error': 'Недійсний формат customer_id. Очікується C-XXX.'}, ensure_ascii=False)

    results = {k: v for k, v in TICKETS.items(
    ) if v['customer_id'] == customer_id}
    return json.dumps(results if results else {'message': 'Тікетів не знайдено'}, ensure_ascii=False)


@mcp.tool()
def get_customer(customer_id: str) -> str:
    """Отримати дані клієнта за ID."""
    c = CUSTOMERS.get(customer_id)
    if not c:
        return json.dumps({'error': f'Клієнта {customer_id} не знайдено'}, ensure_ascii=False)
    return json.dumps({'id': customer_id, **c}, ensure_ascii=False)


@mcp.tool()
def get_ticket_summary() -> str:
    """Отримати статистику щодо відкритих тікетів (read-only tool)."""
    open_tickets = sum(1 for t in TICKETS.values()
                       if t['status'] in ('open', 'in_progress'))
    return json.dumps({'total': len(TICKETS), 'active': open_tickets}, ensure_ascii=False)

# ── Resources ──


@mcp.resource('faq://general')
def faq_resource() -> str:
    """Загальний FAQ для туристичного саппорту (read-only)."""
    faq = [
        {'q': 'Як скасувати авіаквиток?',
            'a': 'Зверніться до підтримки мінімум за 24 години до вильоту.'},
        {'q': 'Час відповіді підтримки?',
            'a': 'Gold: до 1 год; Silver: до 4 год; Standard: до 24 год.'}
    ]
    return json.dumps(faq, ensure_ascii=False)


@mcp.resource('policy://refunds')
def refund_policy() -> str:
    """Політика повернення коштів (read-only)."""
    return "Повернення здійснюється протягом 3-5 робочих днів на картку, з якої була оплата."

# ── Prompts ──


@mcp.prompt()
def support_reply(customer_name: str, issue_summary: str, tone: str = 'professional') -> str:
    """Шаблон ввічливої відповіді клієнту."""
    tones = {
        'professional': 'Сформулюй формальну та чітку відповідь',
        'empathetic': 'Сформулюй теплу відповідь з розумінням проблеми',
        'concise': 'Сформулюй дуже коротку відповідь',
    }
    return (f'{tones.get(tone, tones["professional"])} клієнту {customer_name}. '
            f'Проблема: {issue_summary}. Вкажи, що питання вже вирішується, та запропонуй наступний крок.')


if __name__ == '__main__':
    mcp.run(transport='stdio')
