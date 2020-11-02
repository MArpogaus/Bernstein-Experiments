class MyMinMaxScaler():
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, Y=None):
        self.min = X.min()
        self.max = X.max()

    def transform(self, X, Y=None):
        X_std = (X - self.min) / (self.max - self.min)
        X_scaled = X_std * \
            (self.feature_range[1] - self.feature_range[0]
             ) + self.feature_range[0]
        return X_scaled if Y is None else (X_scaled, self.transform(Y))

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X, Y)


class CERDataScaler():
    def __init__(self, load_col=0, weather_col=1):
        self.load_col = load_col
        self.weather_col = weather_col
        self.load_scaler = MyMinMaxScaler(feature_range=(0, 1))
        self.weather_scaler = MyMinMaxScaler(feature_range=(-1, 1))

    def fit(self, X, Y=None):
        load_data = X[:, :, self.load_col]
        weather_data = X[:, :, self.weather_col]

        self.load_scaler.fit(load_data)
        self.weather_scaler.fit(weather_data)

    def transform(self, X, Y=None):
        load_data = X[:, :, self.load_col]
        weather_data = X[:, :, self.weather_col]

        X[:, :, self.load_col] = self.load_scaler.transform(load_data)
        X[:, :, self.weather_col] = self.weather_scaler.transform(
            weather_data)
        if Y is not None:
            Y = self.load_scaler.transform(Y)

            return X, Y
        else:
            return X

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X, Y)
