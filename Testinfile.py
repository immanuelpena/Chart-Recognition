import pandas as pd
import numpy as np
import os 
import matplotlib
import matplotlib.pyplot as plt
import time
from faker import Faker
from faker.providers import internet

fake = Faker()
fake.add_provider(color)

print(fake.country())


fake.sentence()