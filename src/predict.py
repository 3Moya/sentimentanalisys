import pandas as pd
import numpy as np

from keras.models import load_model

def predict(df):
    model = load_model('model/production/model_prod.h5')

    test = df
    X_test = np.stack(test['feat_vector'].apply(eval))

    y_pred = model.predict(X_test)

    predictions = []

    for i in range(len(y_pred)):
        predictions.append(np.argmax(y_pred[i]))

    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv('data/predictions.csv', index=False)

    return predictions_df.reset_index(inplace=True)

predict(pd.read_csv('data/test.csv'))
