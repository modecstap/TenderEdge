from storage.repositories import MfuRepository
from categoricalEncoder import CategoricalEncoder
from neural_network import NeuralNetwork
from aiohttp import web
import pandas as pd
import numpy as np
import json

class MfuPredictor:

    def __init__(self):
        self.NN = NeuralNetwork
        self.mfuEncoder = CategoricalEncoder
        self.price_mean = 1
        self.price_std = 1
        self.supplie_cost_mean = 1
        self.supplie_cost_std = 1

    async def setup(self):

        columnsname = ["id", "name", "vender", "functional", "price", "refueling_cost", "supplie_cost", "repairability", "parts_support", "manufacturer", "efficiency", "cluster"]
        data = await MfuRepository().get_all()
        data = [obj.__dict__ for obj in data]
        data = pd.DataFrame(data, columns=columnsname)


        mfuDescriptionsDf = data.iloc[:, 2:-1]
        mfuclusters = data['cluster']

        self.mfuEncoder = CategoricalEncoder(mfuDescriptionsDf)
        mfuDescriptionsDf = self.mfuEncoder.encode_df(["vender", "functional", "repairability", "parts_support", "manufacturer", "efficiency"])

        self.price_mean = mfuDescriptionsDf['price'].mean()
        self.price_std = mfuDescriptionsDf['price'].std()
        self.supplie_cost_mean = mfuDescriptionsDf['supplie_cost'].std()
        self.supplie_cost_std = mfuDescriptionsDf['supplie_cost'].mean()
        self.NN = NeuralNetwork(mfuDescriptionsDf.shape[1], 6, mfuclusters.nunique())
        self.NN.set_weigths(
            W1=np.array([
                [ 1.27835251e+00, -2.63478987e-01,  2.23457046e+00,  1.01279001e-01,  7.12805991e-01, -2.38920499e+00],
                [-4.28985251e-02,  1.82245778e-01, -1.71167745e-02, -1.75852450e-01, -9.38482412e-02,  1.73869399e-01],
                [ 1.03171773e+00, -5.16196120e-01,  2.53882521e+00,  7.57557355e-03,  5.98338749e-01, -2.63598139e+00],
                [ 3.10737325e-01,  3.05687965e-01, -3.06093848e-02,  1.07024865e-01,  4.73362708e-01, -2.67987729e-02],
                [ 2.90502220e-01, -1.07460543e-01,  6.22677148e-02, -6.61048245e-02, -8.21998479e-02,  2.32403888e-01],
                [ 4.95609984e-01, -2.54053938e-01, -4.10172949e-01, -9.66777850e-02,  3.26348326e-01,  3.00493465e-01],
                [ 7.18662318e-01,  1.08008064e-01,  5.22071632e-02,  9.01421156e-02,  1.17226163e-01,  3.49726699e-01],
                [-2.05802905e-01, -2.07211276e-01, -2.07854905e-01, -1.65176510e-01, -3.06038667e-02, -1.79089597e-01],
                [ 4.97974890e-01, -4.02214763e-02,  1.07638444e-01, -1.34543263e-01,  1.19183180e-01,  9.82972706e-02],
                [ 7.54409574e-01, -7.13619530e-02, -1.97393874e-01, -2.16531628e-01,  8.93548351e-02,  6.72925801e-01],
                [ 4.27724111e-01, -1.39567582e-02, -2.46030610e-01,  1.52458092e-01,  4.01351765e-01,  4.09951642e-01],
                [ 7.28966343e-01, -2.13555832e-01, -1.09740627e-01, -9.44925315e-02,  2.39503324e-01, -4.04919381e-02],
                [-1.84561486e-01, -8.41330356e-02,  1.38089847e-01, -9.85257614e-02,  1.64629406e-01, -4.88988829e-02],
                [ 4.13935683e-02,  1.66749505e-01,  4.12942092e-02,  4.49928570e-02,  4.72470614e-01,  1.57708595e-01],
                [ 8.62604676e-01,  3.21270254e-02, -8.80750863e-02,  8.87976760e-02,  3.85580886e-01,  2.50189379e-01],
                [-2.14344615e-01, -2.02478026e-01,  1.05388749e-01, -4.32904981e-02, -1.73632415e-01, -5.74392087e-02],
                [ 9.78767016e-01, -6.18039954e-02, -4.26716455e-02, -9.66778817e-02,  3.84652328e-01, -2.50913040e-01],
                [ 5.35827219e-01,  2.94432279e-03, -1.52722349e-01, -1.67688600e-01,  3.31139794e-01,  4.57525880e-01],
                [ 9.57678512e-02,  1.58300140e-01, -1.30138336e-01,  2.09617491e-01, -9.03260810e-02, -1.17916426e-01],
                [ 2.61552275e-01,  1.08567667e-02, -1.44414800e-01, -1.35972818e-01,  6.09729148e-01,  7.16864775e-01],
                [ 9.17732791e-01,  6.19578152e-04, -1.37657666e-01,  1.14173137e-01,  4.95006441e-02,  1.04548989e-01]
            ]),
            b1=np.array([
                [ 0.92936045, -0.23669368, -0.21376481, -0.18079721, 0.64198364, 0.61787316]
            ]),
            W2=np.array([
                [-0.2256025,   0.562208,   -0.56655425,  0.74317006, -2.37321342,  1.39973074,  0.67450482],
                [ 0.15189027, -0.24341176, -0.01756407,  0.2399959,   0.66728548, -0.21005075,  0.29614365],
                [ 2.57697154, -0.55828854,  0.09392975, -1.66885798, -0.33248319, -1.21626759,  0.88291535],
                [-0.23416753, -0.27366185, -0.10114603,  0.082313,   -0.05712697,  0.13238359,  0.095403],
                [-0.13148715, -0.56246591, -0.53942767,  1.05669249, -0.91436794,  0.62170287,  0.19445166],
                [-0.76922972,  1.00957816,  1.6577399,  -0.15073323,  2.1772823,  -2.28446748, -0.98892823]
            ]),
            b2=np.array([
                [-0.47759437,  0.05820214, -0.57228248,  1.18714918, -1.00536868,  0.58356129, 0.17621791]
            ])
        )

    async def get_cluster(self, request):
        try:
            # Чтение данных из тела запроса
            data_param = await request.json()
            data_param['price'] = int(data_param['price'])
            data_param['refueling_cost'] = int(data_param['refueling_cost'])
            data_param['supplie_cost'] = int(data_param['supplie_cost'])
        except json.JSONDecodeError:
            return web.Response(status=400, text='Invalid JSON')


        if data_param:
            try:
                # Преобразование JSON в DataFrame
                data = pd.DataFrame({key: [value] for key, value in data_param.items()})
            except ValueError as e:
                return web.Response(status=400, text=f'Error converting JSON to DataFrame: {e}')

            # Кодирование категориальных признаков

            vector = self.mfuEncoder.encode_categorical_features(data, ["vender", "functional", "repairability", "parts_support", "manufacturer", "efficiency"])

            # Нормализация числовых признаков
            vector['price'] = (vector['price'] - self.price_mean) / self.price_std
            vector['supplie_cost'] = (vector['supplie_cost'] - self.supplie_cost_mean) / self.supplie_cost_std

            # Преобразование DataFrame в numpy array
            vector = vector.to_numpy()

            # Предсказание кластера
            cluster = self.NN.predict(vector).argmax()

            # Получение данных из БД по кластеру
            data = await self.getClusterFromDB(cluster)

            # Преобразование данных в словари
            data_dicts = [row.__dict__ for row in data]
            for item in data_dicts:
                item.pop('_sa_instance_state', None)


            return web.json_response(data_dicts)

        else:
            return web.Response(status=400, text='No data parameter provided')

    async def getClusterFromDB(self, cluster):
        return await MfuRepository().get_cluster(cluster)
