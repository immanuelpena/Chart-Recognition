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
    colorsArray = ['#FF6633', '#FFB399', '#FF33FF', '#FFFF99', '#00B3E6', 
		  '#E6B333', '#3366E6', '#999966', '#99FF99', '#B34D4D',
		  '#80B300', '#809900', '#E6B3B3', '#6680B3', '#66991A', 
		  '#FF99E6', '#CCFF1A', '#FF1A66', '#E6331A', '#33FFCC',
		  '#66994D', '#B366CC', '#4D8000', '#B33300', '#CC80CC', 
		  '#66664D', '#991AFF', '#E666FF', '#4DB3FF', '#1AB399',
		  '#E666B3', '#33991A', '#CC9999', '#B3B31A', '#00E680', 
		  '#4D8066', '#809980', '#E6FF80', '#1AFF33', '#999933',
		  '#FF3380', '#CCCC00', '#66E64D', '#4D80CC', '#9900B3', 
		  '#E64D66', '#4DB380', '#FF4D4D', '#99E6E6', '#6666FF']   
    stylesArray = ['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright',
                    'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 
                    'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks',
                     'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']
    legendPosition = ['best','upper right','upper left', 'lower left', 'lower right','right','center left','center right','lower center','upper center','center']
  
    title_words_list = [x_axis_value,y_axis_value]

    df_plot = fake_data.groupby([x_axis_value])[y_axis_value].sum().reset_index(name=y_axis_value)
    
    df_plot['color'] = np.random.choice(colorsArray,size=(len(df_plot),1))
   
    plt.style.use(np.random.choice(stylesArray)) 

    df_plot.plot(kind='bar',
                 x=x_axis_value,
                 y=y_axis_value,
                 color= df_plot['color'],
                 title = fake.sentence(),
                 width= random.random()
                )
             
    plt.legend(loc = np.random.choice(legendPosition))
    plt.savefig(str(1) + 'books_read.png')
    #plt.bar(x_values, y_values)
   # plt.show()

    
fig = generate_chart(dimensions, measures, fake_data)

