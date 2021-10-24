import pandas as pd
import numpy as np
import os 
import matplotlib
import matplotlib.pyplot as plt
import time
import random
from faker import Faker

fake = Faker()


start_time = time.time()

# for x in range(1,2):P
#     plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
#     plt.xlabel('Months')
#     plt.ylabel('Books Read')
#     plt.savefig(str(x) + 'books_read.png')

# print("--- %s seconds ---" % (time.time() - start_time))

#fake.add_provider(credit_card)

# function to create a dataframe with fake values for our workers
def make_chart_data(num):
    
    dimensions = ['Name', 'Date','CreditCard','Company','WorkerStatus','Position','Country']
    measures = ['Salary', 'Quantity','Sales','Discount']

    status_list = ['Active','Inactive','Suspended']
    team_list = [fake.job() for x in range(4)]

    fake_data = [{'Name':fake.name(),
                  'Date':fake.date_between(start_date='-5y', end_date='+1y'), 
                  'CreditCard':fake.credit_card_provider(), 
                  'Salary':random.randrange(1, 5000, 2), 
                  'Quantity':random.randrange(1, 200, 2),
                  'Sales':random.randrange(50000, 1000000, 2),
                  'Discount':random.random(),
                  'Company':fake.company(),
                  'WorkerStatus':np.random.choice(status_list, p=[0.50, 0.40, 0.10]), # assign items from list with different probabilities
                  'Position':np.random.choice(team_list),
                  'Country':fake.country()} for x in range(num)
                  ]

    return dimensions, measures, fake_data


dimensions, measures, fake_data = make_chart_data(num=15)



def generate_chart(dimensions, measures, fake_data):
    
    fake_data = pd.DataFrame(fake_data)    
  
    x_axis_value = random.choice(dimensions)
    y_axis_value = random.choice(measures)
    
    x_values = fake_data[x_axis_value].unique()
    y_values = fake_data[x_axis_value].value_counts().tolist()

    plt.bar(x_values, y_values)
    plt.show()

    
fig = generate_chart(dimensions, measures, fake_data)

