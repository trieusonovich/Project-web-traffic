from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

data = pd.read_csv(filepath_or_buffer='web-traffic.csv', header = 0)
date= pd.to_datetime(data['date'], dayfirst = True)
# profile = ProfileReport(data, title = 'web-traffic report')
# profile.to_file('web-traffic report.html')

fig, ax = plt.subplots()
ax.plot(data['date'], data['users'],label = 'users')
plt.ylabel('users')
plt.xticks([0,50,100,150,200,250,300,350,400],rotation=45)
plt.legend()
plt.show()

def convert_data(raw_data, window_size):
    i = 1
    while i < window_size:
        raw_data['users_t-{}'.format(i)] = raw_data['users'].shift(i)
        i += 1
    raw_data = raw_data.dropna(axis=0)
    return raw_data

data = convert_data(data, window_size = 10)
x = data.drop(['date','users'],axis = 1)
y = data['users']

size = int(0.8* len(x))
# x_train = x[0:size]
# x_test = x[size:]
# y_train = y[0:size]
# y_test = y[size:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    shuffle=False,
    random_state=42)

model = KNeighborsRegressor(n_neighbors=15,weights='distance',n_jobs=-1)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# for i,j in zip(y_test,y_predict):
#     print('y thực tế:{}'.format(i),'y dự đoán:{}'.format(j))

print('MSE:{}'.format(mean_squared_error(y_test,y_predict)))
print('MAE:{}'.format(mean_absolute_error(y_test,y_predict)))
print('R2:{}'.format(r2_score(y_test,y_predict)))

def forecast_future(model, last_window, steps):
    predictions = []
    current_window = last_window.copy()
    for i in range(steps):
        next_pred = model.predict([current_window])[0]
        predictions.append(next_pred)
        current_window = np.roll(current_window, -1)
        current_window[-1] = next_pred
    return predictions

last_actual_values = data['users'].values[-9:]
y_future = forecast_future(model, last_actual_values, 100)

# Tạo timeline cho tương lai
# last_date = date.iloc[-1]
# future_date = pd.date_range(
#      start=last_date + pd.Timedelta(days=1),
#      periods=100,
#      freq='d')

all_time_index = list(range(len(data))) + list(range(len(data), len(data) + 100))

fig,ax= plt.subplots()
ax.plot(data['date'][0:size],y_train,label = 'Train')
ax.plot(data['date'][size:],y_test,label= 'Test')
ax.plot(data['date'][size:],y_predict,label='Predict')
ax.plot(all_time_index[len(data):], y_future, label='Future')
plt.xlabel('date')
plt.title('Biểu đồ lượng người dùng web')
plt.xticks([0,50,100,150,200,250,300,350,400,500,600],rotation=45)
plt.legend()
plt.show()

# with open('web traffic model.pkl', 'wb') as f:
#     pickle.dump(model,f)
