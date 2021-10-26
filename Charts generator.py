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
    country_list = ['US','Mexico','Germany','Chile','Spain']
    company_list = ['Facebook','Google','Microsoft','Amazon','Yahoo']
    team_list = [fake.job() for x in range(4)]

    fake_data = [{'Name':fake.name(),
                  'Date':fake.date_between(start_date='-5y', end_date='+1y').year, 
                  'CreditCard':fake.credit_card_provider(), 
                  'Salary':random.randrange(1, 5000, 2), 
                  'Quantity':random.randrange(1, 200, 2),
                  'Sales':random.randrange(50000, 1000000, 2),
                  'Discount':random.random(),
                  'Company':np.random.choice(company_list),
                  'WorkerStatus':np.random.choice(status_list, p=[0.50, 0.40, 0.10]), # assign items from list with different probabilities
                  'Position':np.random.choice(team_list),
                  'Country':np.random.choice(country_list)} for x in range(num)
                  ]

    return dimensions, measures, fake_data


dimensions, measures, fake_data = make_chart_data(num=1000)



def generate_bar_chart(dimensions, measures, fake_data, num):
    

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
    
    chart_orientation = ['bar','barh']
    chart_axis = ['Company','Country','WorkerStatus']
    fake_data = pd.DataFrame(fake_data) 
    for x in range(num):

        fake_df = fake_data   
    
        x_axis_value = random.choice(chart_axis)
        y_axis_value = random.choice(measures)
        df_plot = fake_df.groupby([x_axis_value])[y_axis_value].sum().reset_index(name=y_axis_value)


        df_plot = df_plot.sample(frac=round(random.uniform(0.4, 1), 10))
    
        plt.style.use(np.random.choice(stylesArray)) 

        df_plot.plot(kind= np.random.choice(chart_orientation, p=[0.80, 0.20]),
                    x=x_axis_value,
                    y=y_axis_value,
                    title = fake.sentence(),
                    width= round(random.uniform(0.4, 1.0), 10)
                    )
                
        plt.legend(loc = np.random.choice(legendPosition))
        plt.savefig(str(x) + 'bar.png', dpi=40)
        plt.clf() 
        #plt.bar(x_values, y_values)
    # plt.show()

    
fig = generate_bar_chart(dimensions, measures, fake_data, num = 1000)



def generate_line_chart(dimensions, measures, fake_data, num):
    

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

    legendPosition = ['best','upper right','upper left', 'lower left', 'lower right','right','center left','center right','lower center','upper center','center']
    stylesArray = ['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright',
                    'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 
                    'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks',
                     'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']
     
    fake_data = pd.DataFrame(fake_data) 

    for x in range(num):

        fake_df = fake_data
    
        x_axis_value = 'Date'
        x_axis_value2 = 'Country'
        y_axis_value = random.choice(measures)
        y_axis_value2 = random.choice(measures)
        #df_plot = fake_data.groupby([x_axis_value,x_axis_value2])[y_axis_value,y_axis_value2].sum()
        fake_df = fake_df.sample(frac=round(random.uniform(0.1, 0.2), 10))

        df_plot = pd.pivot_table(
        fake_df,
        index=['Date','Country'],
        aggfunc={y_axis_value: np.sum, y_axis_value2: np.sum}
        ).rename(columns={y_axis_value: 'Count1'}).reset_index()
        
        #df_plot['color'] = np.random.choice(colorsArray,size=(len(df_plot),1))
       
        plt.style.use(np.random.choice(stylesArray)) 
        fig, ax = plt.subplots(figsize=(8,6))
        for label, df in df_plot.groupby('Country'):
           df.Count1.plot(kind="line", ax=ax, label=label, linewidth = round(random.uniform(0.4, 1.0), 10))
        plt.title(fake.sentence())
        plt.legend(loc = np.random.choice(legendPosition))        
     
        #ax.legend(loc = np.random.choice(legendPosition))
        plt.savefig(str(x) + 'line.png', dpi=40)
        plt.clf() 
        #plt.bar(x_values, y_values)
    # plt.show()

    
fig = generate_line_chart(dimensions, measures, fake_data, num = 1000)



def generate_pie_chart(dimensions, measures, fake_data, num):
    

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
    
    chart_axis = ['Company','Country','WorkerStatus']
    fake_data = pd.DataFrame(fake_data) 
    for x in range(num):

        fake_df = fake_data 
    
        x_axis_value = random.choice(chart_axis)
        y_axis_value = random.choice(measures)
        df_plot = fake_df.groupby([x_axis_value])[y_axis_value].sum().reset_index(name=y_axis_value)
       

        df_plot = df_plot.sample(frac=round(random.uniform(0.3, 0.7), 10))

        
        plt.style.use(np.random.choice(stylesArray)) 
   
        plt.pie(df_plot[y_axis_value], 
                labels = df_plot[y_axis_value]
                )
        plt.title(fake.sentence())
        #plt.legend(loc = np.random.choice(legendPosition))
        plt.savefig(str(x) + 'pie.png', dpi=40)
        plt.clf() 
        #plt.bar(x_values, y_values)
    # plt.show()

    
fig = generate_pie_chart(dimensions, measures, fake_data, num = 1000)



def generate_scatter_chart(dimensions, measures, fake_data, num):
    

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
    colormapArray = ['viridis', 'plasma', 'inferno', 'magma', 'cividis'] 
    
    fake_data = pd.DataFrame(fake_data) 

    for x in range(num):

        fake_df = fake_data
    
        x_axis_value = random.choice(measures)
        y_axis_value = np.random.choice(['Sales','Discount'])
        y_axis_value2 = np.random.choice(['Quantity','Salary'])
        df_plot = fake_df.groupby([y_axis_value2])[y_axis_value].sum().reset_index(name=y_axis_value)
       

        df_plot = df_plot.sample(frac=round(random.uniform(0.1, 0.2), 10))

        plt.style.use(np.random.choice(stylesArray)) 
        df_plot.plot.scatter(x=y_axis_value,
                              y=y_axis_value2,
                              s=random.randrange(100, 140, 2),
                              c= random.choice(colorsArray),
                              colormap= random.choice(colormapArray),
                              title = fake.sentence()
                              )
        plt.savefig(str(x) + 'scatter.png', dpi=40)
        plt.clf() 
        #plt.bar(x_values, y_values)
    # plt.show()

    
fig = generate_scatter_chart(dimensions, measures, fake_data, num = 1000)


print("--- %s seconds ---" % (time.time() - start_time))