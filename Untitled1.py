#!/usr/bin/env python
# coding: utf-8

# In[1]:


mylist= ["One", "Two", "Three"]


# In[2]:


mylist


# In[3]:


mylist[1:]


# In[4]:


mylist[:1]


# In[5]:


mylist[:3]


# In[6]:


add_list=['Four', "Five", "Six"]


# In[7]:


new_list = mylist + add_list


# In[8]:


new_list


# In[9]:


new_list[0] = 'ONEW'


# In[10]:


new_list


# In[11]:


new_list.append("Seven")


# In[12]:


new_list


# In[13]:


new_list.remove("Seven")


# In[14]:


new_list


# In[16]:


new_list.pop(0)


# In[17]:


new_list


# In[18]:


newList = ['a', 'c', 'x', 'b', 't', 'r']


# In[19]:


newList.sort()


# In[20]:


newList


# In[22]:


dicts = {
    "key1": 1,
    "key2": [1,2,3],
    "key3":{'key_value': [4,5,6]}
}


# In[27]:


dicts["key3"]


# In[28]:


d = {'k1': ["a", "b", "c", "d"]}


# In[29]:


d


# In[31]:


d["k1"][2].upper()


# In[32]:


d["k1"] = { "e"}


# In[34]:


d = {'k1': 100, 'k2': 200, "k3": 300}


# In[36]:


d["k4"]= 400


# In[37]:


d


# In[38]:


d.keys()


# In[39]:


d.values()


# In[40]:


d.items()


# In[41]:


d["k1"] = "New value assign"


# In[42]:


d


# In[43]:


lists = set()


# In[44]:


lists.add(1)


# In[45]:


lists


# In[46]:


lists.add(2)


# In[47]:


lists


# In[48]:


lists = [1,1,1,1,1,1,2,2,2,3,3,6,5,5,4,6,4,8,7,4,6,8,10]


# In[49]:


set(lists)


# In[50]:


get_ipython().run_cell_magic('writefile', 'myfile.txt', 'this is my file\nthis is second line\nthis is third line')


# In[52]:


myfile = open("myfile.txt")


# In[54]:


myfile.read()


# myfile.read()

# In[55]:


myfile.read()


# In[56]:


myfile.seek(0)


# In[57]:


myfile.read()


# In[58]:


with open('myfile.txt') as my_new_file:
    contents = my_new_file.readlines()


# In[59]:


contents


# In[60]:


get_ipython().run_cell_magic('writefile', 'new_file.txt', 'THIS IS THE FIRST PART\nTHIS IS THE SECOND PART\nTHIS IS THE THIRD PART')


# In[64]:


with open('new_file.txt', mode='r') as f:
    print(f.read())


# In[66]:


with open('new_file.txt', mode="a") as f:
    f.write("THIS IS THE FOURTH PART")


# In[68]:


with open("new_file.txt", mode="r") as f:
    print(f.read())


# In[5]:


def calculate_sec(seconds):
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    remaining_seconds = seconds - hours * 3600 - minutes * 60
    return hours, minutes, remaining_seconds
hours , minutes, seconds = calculate_sec(5000)
print(hours, minutes, seconds)


# In[10]:


def gretting(name):
    letter = len(name) * 9
    print(f"Hi this is {name}, Learning python was fun! Huh. Your Lucky number is {letter}")


# In[12]:


gretting("Mithun")
gretting("pip")


# In[14]:


def calculate(d):
    q = 3.1416
    z = d * (q ** 2)
    print(z)
calculate(5)


# In[15]:


def calculate_area(radius):
    pi = 3.1416
    area = radius * (pi ** 2)
    print(area)
calculate(5)


# In[16]:


def userName(name):
    if len(name) < 3:
        print("Invalid username! Please try again")
    else:
        print("valid userName")


# In[17]:


userName('Mithun')


# In[20]:


v = 0.0
def fractional_part(numerator, denominator):
    if numerator == 0:
        return 0
    elif denominator == 0:
        return 0
    else:
        v = numerator/denominator
        v = v - int(v)
        if v == 0.0:
            return 0
    return v

print(fractional_part(5, 5)) # Should be 0
print(fractional_part(5, 4)) # Should be 0.25
print(fractional_part(5, 3)) # Should be 0.66...
print(fractional_part(5, 2)) # Should be 0.5
print(fractional_part(5, 0)) # Should be 0
print(fractional_part(0, 5)) # Should be 0


# In[21]:


"big" > "small"


# In[22]:


11%5


# In[23]:


x = 0
while x < 5:
    print("Not there yet, x=" + str(x))
    x = x+1


# In[29]:


def num(n):
    x = 0
    while x < n:
        print("Not there yet, x = "+str(x))
        x += 1
    print("Done")


# In[30]:


num(5)


# In[1]:


product = 1
for n in range(1,10):
    product = product * n
print(product)


# In[3]:


def to_celsius(x):
    return (x-32)*5/9
for x in range(0, 101, 10):
    print(x, to_celsius(x))


# In[4]:


for left in range(10):
    for right in range(left, 10):
        print("[" +str(left) + "|" + str(right) + "]", end=" ")
print()


# In[5]:


teams = ["Real Madrid", "Man City", "Paris Saint Germaint", "Barcelona"]
for home_teams in teams:
    for away_teams in teams:
        if home_teams != away_teams:
            print(home_teams + " vs " + away_teams)


# In[8]:


def factorial(n):
    print("Factorial called with " + str(n))
    if n < 2:
        print("returnning 1")
        return 1
    result =  n * factorial(n-1)
    print("Returnning " + str(result) + " for factorial " + str(n))
    return result


# In[9]:


factorial(9)


# In[10]:


for x in range(10):
    for y in range(x):
        print(y)


# In[12]:


def votes(params):
    for vote in params:
        print("Possible option:" + vote)


# In[15]:


votes(["yes", "no", "maybe"])


# In[16]:


def decade_counter():
    while year < 50:
        year += 10
    return year


# In[17]:


decade_counter(10)


# In[18]:


Weather = "Rainfall"


# In[22]:


print(Weather[:4])


# In[3]:


animals = ["Monkey", "Zebra", "Dog", "Cat", "Tiger"]
char = 0
for animal in animals:
    char+= len(animal)
print("Total character: {}, Avarage length {}".format(char, char/len(animal)))


# In[4]:


winners = ["Shakib", "Liton", "Tamim", "Virat"]
for index, person in enumerate(winners):
    print("{} - {}".format(index + 1, person))


# In[5]:


def full_emails(people):
    result = []
    for name, email in people:
        result.append("{}, <{}>".format(name, email))
    return result


# In[7]:


full_emails([("Shakil Hossain", "shakil@gmail.com"), ("Habib", "habib@gmail.com")])


# In[8]:


mul = []
for x in range(1,11):
    mul.append(x*7)
print(mul)


# In[9]:


mul = [x*7 for x in range(1,11)]


# In[11]:


mul


# In[3]:


file_count = {
    "JPG": 10,
    "PNG": 14,
    "PDF": 30
}
for extension in file_count:
    print(extension)


# In[5]:


for ext, amount in file_count.items():
    print("There are {} file and .{} extensions".format(amount, ext))


# In[7]:


def count_letters(text):
    letter = {}
    for letter in text:
        if letter not in result:
            result[letter] = 0
        result[letter] += 1
    return result


# In[8]:


wardrobe = {'shirt': ['red', 'blue', 'white'], 'jeans': ['blue', 'black']}
new_items = {'jeans': ['white'], 'scarf': ['yellow'], 'socks': ['black', 'brown']}
wardrobe.update(new_items)


# In[9]:


wardrobe


# In[10]:


animal = "Hippopotamus"


# In[11]:


print(animal[3:6])
print(animal[-5])
print(animal[10:])


# In[12]:


colors = ["red", "white", "blue"]
colors.insert(2, "yellow")


# In[13]:


colors


# In[14]:


host_addresses = {"router": "192.168.1.1", "localhost": "127.0.0.1", "google": "8.8.8.8"}
host_addresses.keys()


# In[1]:


class piglet:
    def speak(self):
        print("Owak Owak")


# In[3]:


helmet = piglet()
helmet.speak()


# In[6]:


class piglet:
    name = "hamlet"
    def speak(self):
        print("Owak my name is {}!".format(self.name))
hamlet=piglet()
hamlet.name = "Piglet"


# In[7]:


hamlet.speak()


# In[9]:


class piglet:
    year = 0
    def piggy_year(self):
        return self.year * 18
piggy = piglet()


# In[10]:


piggy = piglet()


# In[11]:


piggy.piggy_year()


# In[12]:


piggy.year = 2


# In[13]:


piggy.piggy_year()


# In[14]:


class Apple:
    def __init__(self, color, flavor):
        self.color = color
        self.flavor = flavor


# In[16]:


fruits = Apple("red", "sweet")
fruits.color


# In[17]:


fruits.flavor


# In[30]:


class Person:
    def __init__(self, name):
        self.name = name
    def greeting(self):
        # Should return "hi, my name is " followed by the name of the Person.
        return "hi, my name is {}".format(self.name) 

# Create a new instance with a name of your choice
some_person = Person("Mithun") 
# Call the greeting method
print(some_person.greeting())


# In[31]:


class Apple:
    def __init__(self, color, flavor):
        self.color = color
        self.flavor = flavor
    def __str__(self):
        return "This is a {} and very {}".format(self.color, self.flavor)
fruits = Apple("red", "sweet")


# In[32]:


class Repository:
    def __init__(self):
        self.packages = {}
    def add_package(self, package):
        self.packages[package.name] = package
    def total_size(self):
        result = 0
        for package in self.package.values():
            result += package.size
        return result


# In[33]:


class Clothing:
  stock={ 'name': [],'material' :[], 'amount':[]}
  def __init__(self,name):
    material = ""
    self.name = name
  def add_item(self, name, material, amount):
    Clothing.stock['name'].append(self.name)
    Clothing.stock['material'].append(self.material)
    Clothing.stock['amount'].append(amount)
  def Stock_by_Material(self, material):
    count=0
    n=0
    for item in Clothing.stock['___']:
      if item == material:
        count += Clothing.___['amount'][n]
        n+=1
    return count

class shirt(Clothing):
  material="Cotton"
class pants(Clothing):
  material="Cotton"
  
polo = shirt("Polo")
sweatpants = pants("Sweatpants")
polo.add_item(polo.name, polo.material, 4)
sweatpants.add_item(sweatpants.name, sweatpants.material, 6)
current_stock = polo.Stock_by_Material("Cotton")
print(current_stock)


# In[42]:


def get_event_date(event):
    return event.date
def current_user(events):
    events.sort(key=get_event_date)
    machines = {}
    for event in events:
        if event.machine not in machines:
            machines[event.machine] = set()
        if event.type=="login":
            machines[event.machine].add(event.user)
        elif event.type=="logout":
            machines[event.machine].remove(event.user)
    return machines
def generate_user(machines):
    for machine, users in machines.item():
        if len(users) > 0:
            user_list = " ,".join(users)
            print("{} {}".format(machine, user_list))


# In[44]:


class Event:
    def __init__(self, event_date, event_type, machine_name, user):
        self.date = event_date
        self.machine = machine_name
        self.type = event_type
        self.user = user


# In[47]:


events = [
    Event("2023-01-05 01-21-66AM", "login", "myworkstation.local", "jordana"),
    Event("2023-01-05 01-21-66AM", "login", "workstation.local", "jordane"),
    Event("2023-01-05 01-21-66AM", "login", "mywork.local", "jordand"),
    Event("2023-01-05 01-21-66AM", "login", "fuck.local", "jordans"),
    Event("2023-01-05 01-21-66AM", "login", "fuckyou.local", "jordanf"),
    Event("2023-01-05 01-21-66AM", "login", "myworfuckkstation.local", "jordanc")
]


# In[48]:


users= current_user(events)
print(users)


# In[49]:


cloud = wordcloud.WordCloud()
cloud.generate_from_frequencies(frequencies)
cloud.to_file("myfile.jpg")


# In[2]:


import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame


# In[4]:


ser = Series([1,2, 3, 4, 5])


# In[5]:


ser


# In[7]:


ser1 = Series([1000000, 250000054, 321548954, 254687456, 245879542, 4214565,], index= ["Bangladesh", "India", "China", "South-korea", "Pakistan", "Italy"])


# In[8]:


ser1


# # DataFrames

# In[10]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[18]:


import webbrowser
website = "https://www.ncaa.com/rankings/football/fbs/associated-press"
webbrowser.open(website)


# In[19]:


nfl_frame = pd.read_clipboard()


# In[20]:


nfl_frame


# In[21]:


nfl_frame.columns


# In[1]:


import pandas.io.data as pdweb


# In[4]:


get_ipython().system('pip install pandas-datareader')


# In[12]:


from pandas_datareader import data, web  # <- use this


# In[11]:





# # Excel with python

# In[1]:


import pandas as pd


# In[2]:


get_ipython().system('pip install xlrd')


# In[3]:


get_ipython().system('pip install openpyxL')


# In[4]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame


# In[5]:


df_left = DataFrame({"X": np.arange(100),"Y": np.arange(100)})


# In[6]:


df_left


# In[8]:


df_right = DataFrame({"Group_data": [10, 20]}, index =["X", "Y"])


# In[9]:


df_right


# In[12]:


arr1 = np.arange(9).reshape(3,3)
arr1


# In[13]:


np.concatenate([arr1, arr1], axis=1)


# In[14]:


ser1 = Series([0,1,2], index=["T", "U", "V"])
ser2 = Series([3,4], index=["X", "Y"])


# In[16]:


pd.concat([ser1, ser2], axis=1)


# In[17]:


pd.concat([ser1, ser2], keys=["cat", "dog"])


# In[18]:


dframe1 = DataFrame(np.random.randn(4, 3), columns = ["X","Y", "Z"])
dframe2 = DataFrame(np.random.randn(3, 3), columns = ["Y","Q", "X"])


# In[19]:


dframe1


# In[20]:


dframe2


# In[22]:


dframe3= pd.concat([dframe1, dframe2])


# In[23]:


dframe3


# In[25]:


dframe4 =pd.concat([dframe1, dframe2], ignore_index = True)


# In[26]:


dframe4


# In[31]:


import pandas.util.testing as tm; tm.N = 3


# In[33]:


def unpivot(frame):
    N, K = frame.shape
    data = {
        "value" : frame.value.ravel("F"),
        "variable" : np.asarray(frame.colums).repeat(N),
        "date" : np.title(np.asarray(frame.index), K)
    }
    return DataFrame(data, columns= ["date", "variable", "value"])
dframe = unpivot(tm.makeTimeDataFrame())


# In[29]:





# In[34]:


dframe1


# In[35]:


dframe1.describe()


# In[41]:


get_ipython().system('pip install seaborn')


# # Histograms

# In[42]:


import numpy as np
import pandas as pd
from numpy.random import randn

#Stats
from scipy import stats

#plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


data1 = randn(100)


# In[45]:


plt.hist(data1)


# In[46]:


data2 = randn(80)
plt.hist(data2, color = 'indianred')


# In[51]:


plt.hist(data1, density=True, color='indianred', alpha=.5, bins=20)
plt.hist(data2, density=True, alpha=0.5, bins=20)


# In[52]:


data1 = randn(1000)
data2 = randn(1000)


# In[54]:


sns.jointplot(data1, data2)


# In[55]:


#hex bins
sns.jointplot(data1, data2, kind="hex")


# # Kernal Density estimation plot

# In[57]:


dataset=randn(25)


# In[60]:


sns.rugplot(dataset)
plt.ylim(0,1)


# In[63]:


plt.hist(dataset, alpha= 0.3)
sns.rugplot(dataset)


# In[66]:


sns.rugplot(dataset)

#create min and max value
x_min = dataset.min() -2
x_max = dataset.max() +2

x_axis = np.linspace(x_min, x_max, 100)
bandwidth = ((4*dataset.std()**5) / (3*len(dataset))) **.2

kernel_list = []

for data_point in dataset:
    #create a karnel for each point and append it to the kernel_list
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #scale for plotting
    kernal = kernel / kernel.max()
    kernel = kernel* 0.4
    
    plt.plot(x_axis, kernel, color="grey", alpha=0.5)
plt.ylim(0,1)


# In[67]:


sum_of_kde = np.sum(kernel_list, axis=0)
fig = plt.plot(x_axis, sum_of_kde, color="indianred")
sns.rugplot(dataset)
plt.yticks([])
plt.suptitle("Sum of the basis function")


# In[68]:


sns.kdeplot(dataset)


# In[21]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


# In[22]:


data = DataFrame(np.random.randn(100, 2), columns = ["X","Y"])


# In[23]:


data


# In[56]:


X,  Y = make_regression(n_samples= 100, n_features = 1, n_informative = 1,
                      n_targets = 1,noise=50, random_state = 83)


# In[57]:


plt.figure(figsize=(10,6))
plt.scatter(X, Y)
plt.show()


# In[58]:


lreg = LinearRegression()
lreg.fit(X,Y)


# In[59]:


print(f"m= {lreg.coef_}")
print(f"b= {lreg.intercept_}")


# In[60]:


m = lreg.coef_
b = lreg.intercept_
y_pred = lreg.predict(X)
# print (Y_pred)
plt.figure(figsize=(10, 6))
plt.scatter(X,Y)
plt.show()


# In[44]:


m = 105.59434113
b = 30
y_pred_new = (m*X.ravel()+b)
print(y_pred_new)

plt.figure(figsize=(10,6))
plt.scatter(X,Y)
plt.plot(X,y_pred,color='purple', label='OLS') # Actual line from sklearn
plt.plot(X,y_pred_new, label='m=30')# our models predicted line
plt.legend()
plt.show()


# In[61]:


lr= 0.01
#slope = -2*np.sum(y-m*X-b)
slope = -2*np.sum(Y-m*X.ravel()-b)
b = b- lr*slope

y_pred_newest = m*X+b

plt.figure(figsize=(10,6))
plt.scatter(X,Y)
plt.plot(X,y_pred, color= "purple", label= "OLS")
plt.plot(X,y_pred_new, label= "b-30")
plt.plot(X,y_pred_newest,  label= f"OLS{b}")
plt.legend()
plt.show()


# In[62]:


lr = 0.01
slope = -2*np.sum(Y-m*X.ravel()-b)
b_last = b - lr*slope
 
y_pred_newest_last = m*X +b


plt.figure(figsize=(10,6))
plt.scatter(X,Y)
plt.plot(X,y_pred,color='purple', label='OLS(sklearn prediction)')
plt.plot(X,y_pred_new, label='Initial b=30')
plt.plot(X,y_pred_newest, label=f'2nd update {b}')
plt.plot(X,y_pred_newest_last, label=f'3rd update {b_last}')
plt.legend()
plt.show()


# In[63]:


all_b = np.arange(-20, 20, 0.5)
loss =[]
for b in all_b:
    loss.append(np.power(np.sum(Y-m*X.ravel()-b),2))


# In[64]:


loss = np.array(loss)


# In[65]:


plt.figure(figsize=(10,6))
plt.plot(all_b, loss)
plt.show()


# In[66]:


m = lreg.coef_
b = 25
lrate = 0.01
slope = -2*np.sum(Y-m*X.ravel()-b)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

for i in range(8):
    slope = -2*np.sum(Y-m*X.ravel()-b)
    b = b - lrate*slope
    l = np.power(np.sum(Y-m*X.ravel()-b),2) # loss
    y_pred_newest = m*X.ravel()+b
    ax1.plot(X,y_pred_newest)
    ax1.scatter(X,Y)
    ax1.plot(X,lreg.predict(X),color='red', label='OLS') # Actaul line
    ax2.plot(all_b,loss)
    ax2.scatter(b,l)
  
plt.show()


# In[ ]:




