import json
from pymongo import MongoClient
from datetime import datetime

# connect to mongo db 
url = "mongodb+srv://vudinhtruongan:demo@test.6o5s3eu.mongodb.net/test"
client = MongoClient(url)

db = client["user_management"] 
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%d-%m-%Y")
# db[formatted_datetime].drop()
# check_in = db[formatted_datetime]
check_in = db["timesheets"]

pp_list = f"./processing_data/data/{formatted_datetime}.json"



with open(pp_list) as f:
    a = json.load(f)

pp_in = {}
pp_out = {}


for value in set(a.values()):  # duyệt qua tất cả các giá trị duy nhất trong dictionary
    first_key = None
    last_key = None
    count = 0
    for key in a:
        if a[key] == value:
            count += 1
            if count == 1:
                first_key = key
            last_key = key
    if first_key is not None and last_key is not None:
        time_first_key = datetime.strptime(first_key, "%Y-%m-%d %H:%M:%S.%f")
        pp_in[value] = time_first_key
        time_last_key = datetime.strptime(last_key, "%Y-%m-%d %H:%M:%S.%f")
        pp_out[value] = time_last_key
      

combined_dict = {}
for key in pp_in.keys():
    combined_dict[key] = {'check_in': pp_in[key], 'check_out': pp_out[key]}


for key, value in combined_dict.items():
    time_working = (value["check_out"] - value["check_in"]).total_seconds()
    last_checkin = datetime.today().replace(hour=9,minute=15,second=0,microsecond=0)
    if value["check_in"]>last_checkin:
        late = True
    else:
        late = False
    data = {"id": key, "check_in": value["check_in"], "check_out": value["check_out"]}
    check_in.insert_one(data)
