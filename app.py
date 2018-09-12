import tornado.ioloop
import tornado.web
import pandas as pd
import numpy as np

from helpers.microphone import Recording
from helpers.preprocessing import FeatureExtraction
from helpers.model import Metrics


class Predictor(tornado.web.RequestHandler):
    """
    Predicts the audio recorded after performing GET request from
    http://localhost:5050/predict
    """
    def get(self):

        rec = Recording()
        rec.start_recording()
        rec.stop_recording()

        fe = FeatureExtraction()
        test_df = pd.read_csv('./recording/test_data.csv')
        test_data = test_df['fname'].progress_apply(fe.get_mfcc, path='./recording/')
        test_data['fname'] = test_df['fname']
        test_data['label'] = np.zeros((len(test_df['fname'])))

        path = fe.filepath
        test_files = test_df.fname.values
        test_features = fe.extract_features(test_files, path)

        test_data = test_data.merge(test_features, on='fname', how='left')

        metrics = Metrics()

        test_data_np = test_data.drop(['label', 'fname'], axis=1).values
        test_data_resized = np.resize(test_data_np, (1, metrics.X_fromcsv.shape[1]))
        print(metrics.model.predict(test_data_resized))
        str_preds, _ = metrics.proba2labels(metrics.model.predict_proba(test_data_resized), metrics.i2c, k=1)
        response = {
            "classification": str_preds[0]
        }
        self.write(response)
        self.finish()


def make_app():
    return tornado.web.Application([
        (r"/predict", Predictor)
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(5050)
    print("Server listening on port 5050!")
    tornado.ioloop.IOLoop.current().start()
