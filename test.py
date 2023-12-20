from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData

custom_obj = CustomData(0.31,61.5,58.0,4.31,4.33,2.66,'Premium','D','SI1')
data = custom_obj.Get_data_as_dataframe()

print(data)