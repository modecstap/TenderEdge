from math import ceil, log
from typing import List

from pandas import DataFrame

class CategoricalEncoder:
    """
    A class to encode and decode categorical features in a DataFrame.

    Attributes:
        df (DataFrame): Input DataFrame containing categorical features.
        coded_categories (dict): Dictionary to store encoded categories.
    """

    def __init__(self, df: DataFrame):
        self.df = df
        self.coded_categories = {}

    def encode_df(self, categories: List[str]) -> DataFrame:
        """
        Encode categorical features in the DataFrame.

        Args:
            categories (List[str]): List of column names representing categorical features.

        Returns:
            DataFrame: DataFrame with encoded categorical features.
        """

        for category in categories:
            unique_values = self.df[category].unique()
            bits_needed = ceil(log(len(unique_values), 2)) + 1
            encod_table = {
                value: [0 if x == ' ' else int(x) for x in f"{i + 1:{bits_needed}b}"]
                for i, value in enumerate(unique_values)
            }

            encoded_columns = [f"{category}_{i}" for i in range(bits_needed)]

            self.coded_categories[category] = {
                'encod_table': encod_table,
                'encoded_columns': encoded_columns
            }
            self.df[encoded_columns] = self.df[category].apply(lambda x: encod_table.get(x, [0] * bits_needed)).tolist()
            self.df = self.df.drop(category, axis=1)

        return self.df

    def decode_df(self):
        """
        Decode the previously encoded categorical features in the DataFrame.

        Returns:
            DataFrame: DataFrame with decoded categorical features.
        """

        for category_name, coded_value in self.coded_categories.items():
            self.df[category_name] = self.df[coded_value['encoded_columns']].values.tolist()
            self.df[category_name] = self.df[category_name].apply(
                lambda x: self.__decode_value(x, coded_value['encod_table']))
            self.df = self.df.drop(coded_value['encoded_columns'], axis=1)

        return self.df

    def encode_categorical_features(self, df: DataFrame, categories: List[str]):
        for category in categories:
            encoded_columns = self.coded_categories[category]['encoded_columns']
            encod_table = self.coded_categories[category]['encod_table']
            bits_needed = len(self.coded_categories[category]['encoded_columns'])
            df[encoded_columns] = df[category].apply(lambda x: encod_table.get(x, [0] * bits_needed)).tolist()
            df = df.drop(category, axis=1)
        return df


    def decode_categorical_features(self, df):
        for category_name, coded_value in self.coded_categories.items():
            df[category_name] = df[coded_value['encoded_columns']].values.tolist()
            df[category_name] = df[category_name].apply(
                lambda x: self.__decode_value(x, coded_value['encod_table']))
            df = df.drop(coded_value['encoded_columns'], axis=1)
        return df


    def __decode_value(self, x, encod_table):
        return list(encod_table.keys())[list(encod_table.values()).index(x)]
