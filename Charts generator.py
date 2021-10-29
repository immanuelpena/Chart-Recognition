import pandas as pd
import numpy as np
import os 
import time
import random
from faker import Faker
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import copy

os.mkdir('images')

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
                     'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
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
        plt.savefig('./images/' + str(x) + '_bar.png', dpi=40)
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
                     'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
     
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
        plt.savefig('./images/' + str(x) + '_line.png', dpi=40)
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
                     'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
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
        plt.savefig('./images/' + str(x) + '_pie.png', dpi=40)
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
                     'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
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
        plt.savefig('./images/' + str(x) + '_scatter.png', dpi=40)
        plt.clf() 
        #plt.bar(x_values, y_values)
    # plt.show()

    
fig = generate_scatter_chart(dimensions, measures, fake_data, num = 1000)


print("--- %s seconds ---" % (time.time() - start_time))



import os
import numpy as np
import substring
path = './images'

files = os.listdir(path)

filesArray = []

for f in files:
	filesArray.append(f)
 
filesArray = np.array(filesArray)

df = pd.DataFrame(data=filesArray, columns=["FileName"])

df['clase'] = df.apply(lambda x: substring.substringByChar(x['FileName'], startChar="_", endChar="."),axis=1)
df['clase'] = df['clase'].str.replace('_','')
df['clase'] = df['clase'].str.replace('.','')

print(df)


import pandas as pd
import math
import random
import os
import shutil


random.seed(30)


classes = df["clase"].unique()
classesFinal = [cl.replace(' ', '_') for cl in classes]
print(classesFinal)

try:
  os.mkdir('dataset')
except OSError:
  print ("No se pudo crear folder dataset")
else:
  print ("Se creó folder dataset")

try:
  os.mkdir('dataset/train')
except OSError:
  print ("No se pudo crear folder dataset")
else:
  print ("Se creó folder dataset")

try:
  os.mkdir('dataset/test')
except OSError:
  print ("No se pudo crear folder dataset")
else:
  print ("Se creó folder dataset")

try:
  os.mkdir('dataset/val')
except OSError:
  print ("No se pudo crear folder dataset")
else:
  print ("Se creó folder dataset")

for cl in classesFinal:
  try:
    os.mkdir(os.path.join('dataset', "train", cl))
  except OSError:
    print (f"No se pudo crear folder train {cl}")
  else:
    print (f"Se creó folder train {cl}")
  
  try:
    os.mkdir(os.path.join('dataset', "test", cl))
  except OSError:
    print (f"No se pudo crear folder test {cl}")
  else:
    print (f"Se creó folder test {cl}")
  
  try:
    os.mkdir(os.path.join('dataset', "val", cl))
  except OSError:
    print (f"No se pudo crear folder val {cl}")
  else:
    print (f"Se creó folder val {cl}")



train_df = df.sample(frac=0.7,random_state=200)
train_df.columns = ["imagen", "clase"]
train_df.reset_index

val_df = df.sample(frac=0.3,random_state=200)
val_df.columns = ["imagen", "clase"]
val_df.reset_index

test_df =df
test_df.columns = ["imagen", "clase"]

dataset = dict()


pathTrain = './dataset/train/'
pathSource = './images/'

#Copiar archivos de train
for index, elem in train_df.iterrows():
    imagen = elem['imagen']
    clase = elem['clase']
    shutil.copy(os.path.join(pathSource,imagen), os.path.join(pathTrain , clase ,imagen))



pathTest = './dataset/test/'
pathSource = './images/'

#Copiar archivos de test
for index, elem in test_df.iterrows():
    imagen = elem['imagen']
    clase = elem['clase']
    shutil.copy(os.path.join(pathSource,imagen), os.path.join(pathTest, clase ,imagen))



pathVal = './dataset/val/'
pathSource = './images/'

#Copiar archivos de validacion
for index, elem in val_df.iterrows():
    imagen = elem['imagen']
    clase = elem['clase']
    shutil.copy(os.path.join(pathSource,imagen), os.path.join(pathVal, clase ,imagen))



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import time
import os
import copy

pathDataset = 'dataset/'

train_dataset = torchvision.datasets.ImageFolder(pathDataset + 'train', 
                                                    transform = transforms.Compose([
                                                        transforms.RandomVerticalFlip(),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomResizedCrop(224),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std = [0.229, 0.224, 0.225])]))

val_dataset = torchvision.datasets.ImageFolder(pathDataset + 'val',
                                                    transform = transforms.Compose([ transforms.Resize(256),
                                                                    transforms.CenterCrop(224),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std = [0.229, 0.224, 0.225])]))

test_dataset = torchvision.datasets.ImageFolder(pathDataset + 'test',
                                                    transform = transforms.Compose([ transforms.Resize(224),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std = [0.229, 0.224, 0.225])]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

class_names = train_dataset.classes

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds ==  labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print('Train Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        #Validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects / len(val_dataset)
        print('Val Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    print('Best accuracy: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model

def test_model(model, criterion):
  model.eval()
  running_loss = 0.0
  running_corrects = 0.0

  for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

  epoch_loss = running_loss / len(test_dataset)
  epoch_acc = running_corrects / len(test_dataset)
  print('Test Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))



#Usamos un modelo pre-entrenado ResNet18 con dataset Imagenet
model_ft = models.resnet18(pretrained=True)

#Vamos a cambiar la última capa de la red. La red original fur entrenada con 
# 1000 clases. Ahora solo necesitamos 6 neuronas de salida.
# Cambiamos la arquitectura final de la red

num_ft = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ft, 6)

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

#La red original fue entrenada con SGD y un lr inicial de 0.1, el cual decrementaba cada vez que el 
# error se estancaba. Nosotros partimos con SGD y un lr bajo, dado que solo queremos tunear la red
# para el nuevo problema

optimizer = torch.optim.SGD(model_ft.parameters(), lr = 0.001, momentum=0.9)

# Empezamos con un número de épocas bajo. A más épocas podemos mejorar el performance
model_ft = train_model(model_ft, criterion, optimizer, num_epochs=30)

#Guardamos la mejor red, en términos de accuracy de validación
torch.save(model_ft.state_dict(), 'resnet18_finetuned.pth')