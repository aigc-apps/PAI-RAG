from typing import Optional
from llama_index.core.bridge.pydantic import BaseModel

# we will store booking under random IDs
bookings = {}


# we will represent and track the state of a booking as a Pydantic model
class Booking(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None


def get_booking_state_func(room_id: str) -> str:
    try:
        return str(bookings[room_id].dict())
    except Exception:
        return f"没有找到会议室ID:{room_id}"


def update_booking_func(room_id: str, property: str, value: str) -> str:
    booking = bookings[room_id]
    setattr(booking, property, value)
    return f"预订ID {room_id} 更新属性信息 {property} = {value}"


def create_booking_func(room_id: str) -> str:
    bookings[room_id] = Booking()
    # return "Booking created, but not yet confirmed. Please provide your name, email, phone, date, and time."
    return f"会议室预订ID:{room_id}已创建，但尚未确认。请提供您的姓名(name)、电子邮件(email)、电话(phone)、日期(date)和时间(time)。"


def confirm_booking_func(room_id: str) -> str:
    # """Confirm a booking for a given booking ID."""
    """确认给定会议室预订ID的预订。"""
    booking = bookings[room_id]

    if booking.name is None:
        raise ValueError("请提供您的名字:name。")

    if booking.email is None:
        raise ValueError("请提供您的邮箱:email。")

    if booking.phone is None:
        raise ValueError("请提供您的电话号码:phone。")

    if booking.date is None:
        raise ValueError("请提供您需要预定的日期:date。")

    if booking.time is None:
        raise ValueError("请提供您需要预定的具体时间:time。")

    # return f"Booking ID {user_id} confirmed!"
    return f"会议室预订ID {room_id} 信息已全部确认，预定成功！"
