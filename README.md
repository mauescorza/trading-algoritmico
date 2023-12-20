# Time Series Prediction with LSTM

This Python script demonstrates the use of Long Short-Term Memory (LSTM) networks for time series prediction using the Keras library. The data used in this example is related to stock prices, specifically the midprice of a stock.

## Requirements
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Keras](https://keras.io/)
- [Google Colab](https://colab.research.google.com/)

## Usage Instructions

1. Open the script in a Jupyter notebook or Google Colab.

2. Make sure you have the required libraries installed. You can install them using:
   ```bash
   !pip install numpy matplotlib pandas scikit-learn keras

3. Load the data from the specified file path:
   ```bash
   dataset = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/hpq.us.txt', delimiter=',', usecols=['Date','Open','High','Low','Close'])

4. Explore the data and visualize it using various plots.

5. Normalize the training dataset:

   ```bash
   sc = MinMaxScaler(feature_range=(0,1))
   set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)


6. Build and train the LSTM model:
   ```bash
   modelo = Sequential()
   modelo.add(LSTM(units=na, input_shape=dim_entrada))
   modelo.add(Dense(units=dim_salida))
   modelo.compile(optimizer='rmsprop', loss='mse')
   modelo.fit(X_train, Y_train, epochs=10, batch_size=20)
   Validate the model and make predictions:

   ```bash
   prediccion = modelo.predict(X_test)
   prediccion = sc.inverse_transform(prediccion)

7. Evaluate the model performance using Mean Squared Error (MSE) and R-squared (R2) metrics.

8. Feel free to modify and experiment with the parameters to see how they affect the model's predictions.
