"""diabetes_prediction

Returns:
    csv: output
"""
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)

x_train_data = pd.read_csv("training_data/Diabetes_XTrain.csv")

y_train_data = pd.read_csv("training_data/Diabetes_YTrain.csv")
test = pd.read_csv("test_cases/Diabetes_Xtest.csv")
logging.info(f"x_train_data shape:{x_train_data.shape}")
logging.info(f"y_train_data shape:{y_train_data.shape}")
logging.info(f"test_data shape:{test.shape}")

data_x = x_train_data.values
logging.info(f"data_x shape:{data_x.shape}, type: {type(data_x)}")

data_y = y_train_data.values
logging.info(f"data_x shape:{data_y.shape}, type: {type(data_y)}")

data_test_x = test.values
logging.info(f"data_x shape:{data_test_x.shape}, type: {type(data_test_x)}")

x_train_data_train = data_x[:, 0:]
logging.info(
    f"updated x_train_data shape:{x_train_data_train.shape}, type: {type(x_train_data_train)}"
)
y_train_data_train = data_y[:, 0]
logging.info(
    f"updated y_train_data shape:{y_train_data_train.shape}, type: {type(y_train_data_train)}"
)
x_train_data_test = data_test_x
logging.info(
    f"updated x_train_data_test shape:{x_train_data_test.shape}, type: {type(x_train_data_test)}"
)


def dist(query_point, x_train_data_i) -> float:
    """finds the distribution

    Args:
        query_point
        x_train_data_i

    Returns:
        float: value
    """
    value = np.sqrt(sum((query_point - x_train_data_i) ** 2))
    return value


# Test Time
def knn(x_train_data_train_value, y_train_data_train_value, query_point, k=5):
    """_summary_

    Args:
        x_train_data_train_value (_type_): _description_
        y_train_data (_type_): _description_
        query_point (_type_): _description_
        k (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    vals = []
    x_train_shape = x_train_data_train_value.shape[0]

    for i in range(x_train_shape):
        dist_value = dist(query_point, x_train_data_train_value[i])
        vals.append((dist_value, y_train_data_train_value[i]))

    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]

    vals = np.array(vals)

    new_vals = np.unique(vals[:, 1], return_counts=True)

    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return pred


FILE_NAME = "output.csv"
output_list = []
for row in range(0, 192):
    prediction = knn(x_train_data_train, y_train_data_train, x_train_data_test[row])
    output_list.append(int(prediction))
df = pd.DataFrame(output_list)
logging.info(f"final output dataframe: {df.shape}")
df.to_csv("output.csv", index=False)
