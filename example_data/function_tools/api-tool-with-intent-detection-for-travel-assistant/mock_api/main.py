from fastapi import FastAPI
from typing import List

app = FastAPI()

# Mock 数据
flights_data = [
    {
        "flight_number": "CA123",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-05",
        "departure_time": "2024-08-05T08:00:00",
        "arrival_time": "2024-08-05T10:00:00",
        "price": 1200,
    },
    {
        "flight_number": "MU456",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-06",
        "departure_time": "2024-08-06T14:00:00",
        "arrival_time": "2024-08-06T16:00:00",
        "price": 1300,
    },
    {
        "flight_number": "HU789",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-06",
        "departure_time": "2024-08-06T18:00:00",
        "arrival_time": "2024-08-06T20:00:00",
        "price": 1100,
    },
    {
        "flight_number": "CA234",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-07",
        "departure_time": "2024-08-07T06:00:00",
        "arrival_time": "2024-08-07T08:00:00",
        "price": 1250,
    },
    {
        "flight_number": "MU567",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-07",
        "departure_time": "2024-08-07T21:00:00",
        "arrival_time": "2024-08-07T23:00:00",
        "price": 1350,
    },
]

highspeed_trains_data = [
    {
        "train_number": "G1234",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-05",
        "departure_time": "2024-08-05T09:00:00",
        "arrival_time": "2024-08-05T11:30:00",
        "price": 800,
    },
    {
        "train_number": "G5678",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-05",
        "departure_time": "2024-08-05T15:00:00",
        "arrival_time": "2024-08-05T17:30:00",
        "price": 850,
    },
    {
        "train_number": "G9101",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-06",
        "departure_time": "2024-08-06T18:30:00",
        "arrival_time": "2024-08-06T21:00:00",
        "price": 780,
    },
    {
        "train_number": "G1123",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-06",
        "departure_time": "2024-08-06T07:00:00",
        "arrival_time": "2024-08-06T09:30:00",
        "price": 820,
    },
    {
        "train_number": "G4578",
        "from": "北京",
        "to": "上海",
        "date": "2024-08-07",
        "departure_time": "2024-08-07T22:00:00",
        "arrival_time": "2024-08-07T00:30:00",
        "price": 870,
    },
]

hotels_data = [
    {
        "hotel_name": "万豪酒店",
        "city": "上海",
        "check_in": "2024-08-05",
        "check_out": "2024-08-07",
        "price_per_night": 600,
    },
    {
        "hotel_name": "希尔顿酒店",
        "city": "上海",
        "check_in": "2024-08-05",
        "check_out": "2024-08-06",
        "price_per_night": 850,
    },
    {
        "hotel_name": "洲际酒店",
        "city": "上海",
        "check_in": "2024-08-06",
        "check_out": "2024-08-07",
        "price_per_night": 700,
    },
    {
        "hotel_name": "皇冠假日酒店",
        "city": "上海",
        "check_in": "2024-08-07",
        "check_out": "2024-08-08",
        "price_per_night": 750,
    },
    {
        "hotel_name": "如家酒店",
        "city": "上海",
        "check_in": "2024-08-07",
        "check_out": "2024-08-09",
        "price_per_night": 300,
    },
]


@app.get("/demo/api/flights", response_model=List[dict])
def get_flights(from_city: str, to_city: str, date: str):
    # 过滤条件可以根据需求添加
    return [
        flight
        for flight in flights_data
        if flight["from"] == from_city
        and flight["to"] == to_city
        and flight["date"] == date
    ]


@app.get("/demo/api/trains", response_model=List[dict])
def get_highspeed_trains(from_city: str, to_city: str, date: str):
    return [
        train
        for train in highspeed_trains_data
        if train["from"] == from_city
        and train["to"] == to_city
        and train["date"] == date
    ]


@app.get("/demo/api/hotels", response_model=List[dict])
def get_hotels(city: str, date: str):
    return [
        hotel
        for hotel in hotels_data
        if hotel["city"] == city and hotel["check_in"] == date
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8070)
