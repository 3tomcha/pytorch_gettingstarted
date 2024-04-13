import torch
import pandas as pd
import json


# model = torch.load("1hupdown_model.pth")
# model.eval()

# with torch.no_grad():
#   predicted = model()

# # 予測したい日時
# predicted_date = pd.to_datetime("2025-04-01 12:00:00", utc=True)
# print(predicted_date)

def lambda_handler(event, context):
  return {
    "statusCode": 200,
    "body": json.dumps("Hello from lambda")
  }