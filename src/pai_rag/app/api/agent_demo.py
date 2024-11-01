from datetime import datetime
from fastapi import APIRouter
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

demo_router = APIRouter()


# Mock 数据
flights_data = [
    {
        "flight_number": "CA123",
        "from": "北京",
        "to": "上海",
        "departure_time": "08:00",
        "arrival_time": "10:00",
        "price": 1200,
    },
    {
        "flight_number": "MU456",
        "from": "北京",
        "to": "上海",
        "departure_time": "14:00",
        "arrival_time": "16:00",
        "price": 1300,
    },
    {
        "flight_number": "HU789",
        "from": "北京",
        "to": "上海",
        "departure_time": "18:00",
        "arrival_time": "20:00",
        "price": 1100,
    },
    {
        "flight_number": "CA234",
        "from": "北京",
        "to": "上海",
        "departure_time": "06:00",
        "arrival_time": "08:00",
        "price": 1250,
    },
    {
        "flight_number": "MU567",
        "from": "北京",
        "to": "上海",
        "departure_time": "21:00",
        "arrival_time": "23:00",
        "price": 1350,
    },
]

highspeed_trains_data = [
    {
        "train_number": "G1234",
        "from": "北京",
        "to": "上海",
        "departure_time": "09:00",
        "arrival_time": "11:30",
        "price": 800,
    },
    {
        "train_number": "G5678",
        "from": "北京",
        "to": "上海",
        "departure_time": "15:00",
        "arrival_time": "17:30",
        "price": 850,
    },
    {
        "train_number": "G9101",
        "from": "北京",
        "to": "上海",
        "departure_time": "18:30",
        "arrival_time": "21:00",
        "price": 780,
    },
    {
        "train_number": "G1123",
        "from": "北京",
        "to": "上海",
        "departure_time": "07:00",
        "arrival_time": "09:30",
        "price": 820,
    },
    {
        "train_number": "G4578",
        "from": "北京",
        "to": "上海",
        "departure_time": "22:00",
        "arrival_time": "00:30",
        "price": 870,
    },
    {
        "train_number": "G85",
        "from": "上海",
        "to": "北京",
        "departure_time": "09:00",
        "arrival_time": "14:30",
        "price": 900,
    },
    {
        "train_number": "G87",
        "from": "上海",
        "to": "北京",
        "departure_time": "11:00",
        "arrival_time": "17:30",
        "price": 1001,
    },
    {
        "train_number": "G88",
        "from": "上海",
        "to": "北京",
        "departure_time": "13:10",
        "arrival_time": "17:40",
        "price": 767,
    },
    {
        "train_number": "G110",
        "from": "上海",
        "to": "北京",
        "departure_time": "18:00",
        "arrival_time": "23:30",
        "price": 598,
    },
]

hotels_data = [
    {
        "hotel_name": "大地花园酒店",
        "city": "北京",
        "price_per_night": 300,
    },
    {
        "hotel_name": "凯悦酒店",
        "city": "北京",
        "price_per_night": 1000,
    },
    {
        "hotel_name": "秋果酒店金融街店",
        "city": "北京",
        "price_per_night": 500,
    },
    {
        "hotel_name": "万豪酒店",
        "city": "上海",
        "price_per_night": 600,
    },
    {
        "hotel_name": "希尔顿酒店",
        "city": "上海",
        "price_per_night": 850,
    },
    {
        "hotel_name": "洲际酒店",
        "city": "上海",
        "price_per_night": 700,
    },
    {
        "hotel_name": "皇冠假日酒店",
        "city": "上海",
        "price_per_night": 750,
    },
    {
        "hotel_name": "如家酒店",
        "city": "上海",
        "price_per_night": 300,
    },
]


@demo_router.get("/flights")
async def get_flights(date: str, to_city: str, from_city: str):
    try:
        _ = datetime.strptime(date, "%Y-%m-%d")
    except Exception as _:
        return {
            "error": f"Invalid date format '{date}'. Please provide a date in YYYY-MM-DD format."
        }

    raw_fights = [
        flight
        for flight in flights_data
        if flight["from"] == from_city and flight["to"] == to_city
    ]

    for flight in raw_fights:
        flight["date"] = date

    return raw_fights


@demo_router.get("/trains")
async def get_trains(date: str, to_city: str, from_city: str):
    try:
        _ = datetime.strptime(date, "%Y-%m-%d")
    except Exception as _:
        return {
            "error": f"Invalid date format '{date}'. Please provide a date in YYYY-MM-DD format."
        }

    raw_trains = [
        train
        for train in highspeed_trains_data
        if train["from"] == from_city and train["to"] == to_city
    ]

    for train in raw_trains:
        train["date"] = date

    return raw_trains


class HotelInput(BaseModel):
    checkin_date: str
    checkout_date: str
    city: str


@demo_router.post("/hotels")
async def get_hotels(input: HotelInput):
    try:
        _ = datetime.strptime(input.checkin_date, "%Y-%m-%d")
        _ = datetime.strptime(input.checkout_date, "%Y-%m-%d")
    except Exception as _:
        return {
            "error": f"Invalid date format '{input}'. Please provide a date in YYYY-MM-DD format."
        }

    hotels = [hotel for hotel in hotels_data if hotel["city"] == input.city]

    for hotel in hotels:
        hotel["checkin_date"] = input.checkin_date
        hotel["checkout_date"] = input.checkout_date

    return hotels
