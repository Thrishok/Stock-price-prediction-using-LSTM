import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

stock="^NSEI"
df=yf.download(stock,period="60d",interval="5m")
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
df.tail()

X = df[['Open']]  
y = df['Close']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

user_input = float(input("Enter the 'Open' price: "))
input_data = pd.DataFrame({'Open': [user_input]})
predicted_close = model.predict(input_data)
print(f"Predicted 'Close' price: {predicted_close[0]}")