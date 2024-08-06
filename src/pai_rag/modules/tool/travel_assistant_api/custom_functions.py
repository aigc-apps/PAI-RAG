from pai_rag.modules.tool.travel_assistant_api.ticket_info import (
    flight_tickets,
    train_tickets,
)


def search_flight_ticket(
    arr_city: str, arr_airport: str, dep_city: str, dep_airport: str, flight_date: str
) -> str:
    try:
        key = f"{dep_city}-{dep_airport}-{arr_city}-{arr_airport}-{flight_date}"
        return str(flight_tickets[key].dict())
    except Exception:
        return f"没有找到相关的航班信息: 出发地: {dep_city}-{dep_airport} 到达地: {arr_city}-{arr_airport}  时间: {flight_date}"


def search_train_ticket(start_city: str, end_city: str, train_date: str) -> str:
    try:
        key = f"{start_city}-{end_city}-{train_date}"
        return str(train_tickets[key].dict())
    except Exception:
        return f"没有找到相关的火车信息: 出发地: {start_city} 到达地: {end_city}  时间: {train_date}"
