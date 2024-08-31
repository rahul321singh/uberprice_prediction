import datetime

import pytz
# fetch date and time in order to predict price wrt to week days and current time(hours)
def date_time():

    current_time = datetime.datetime.now(pytz.timezone('America/New_York'))

    print("The attributes of Boston are :")

    print("Year :", current_time.year)

    print("Month : ", current_time.month)

    print("Day : ", current_time.day)

    print("Hour : ", current_time.hour)

    print("Minute : ", current_time.minute)


date_time()
