import pandas as pd
import numpy as np
import os 
import matplotlib
import matplotlib.pyplot as plt
import time
from faker import Faker

fake = Faker()


start_time = time.time()

# for x in range(1,2):
#     plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
#     plt.xlabel('Months')
#     plt.ylabel('Books Read')
#     plt.savefig(str(x) + 'books_read.png')

# print("--- %s seconds ---" % (time.time() - start_time))


# function to create a dataframe with fake values for our workers
def make_chart_data(num):
    
    # lists to randomly assign to workers
    status_list = ['Full Time', 'Part Time', 'Per Diem']
    team_list = [fake.color_name() for x in range(4)]
    

    fake_workers = [{'Worker ID':x+1000,
                  'Worker Name':fake.name(), 
                  'Hire Date':fake.date_between(start_date='-30y', end_date='today'),
                  'Worker Status':np.random.choice(status_list, p=[0.50, 0.30, 0.20]), # assign items from list with different probabilities
                  'Team':np.random.choice(team_list)} for x in range(num)]
        
    return fake_workers

worker_df = pd.DataFrame(make_chart_data(num=15))

print('asd1')