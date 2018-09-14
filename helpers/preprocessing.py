import os
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_pandas
import scipy
import librosa
from scipy.stats import skew

tqdm.pandas()


class Preprocessing:

    def __init__(self):
        self.SAMPLE_RATE = 45100
        self.filepath = '../recording/'
        self.filename = 'file.wav'


class FeatureExtraction(Preprocessing):

    def __init__(self):
        super().__init__()

    def get_mfcc(self, name, path):
        try:
            b, _ = librosa.core.load(path + name, sr=self.SAMPLE_RATE)
            assert _ == self.SAMPLE_RATE
            gmm = librosa.feature.mfcc(b, sr=self.SAMPLE_RATE, n_mfcc=20)
            return pd.Series(
                np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1), skew(gmm, axis=1), np.median(gmm, axis=1))))
        except:
            print('bad file')
            return pd.Series([0] * 80)

    def extract_features(self, files, path):
        features = {}

        cnt = 0
        for f in tqdm(files):
            features[f] = {}
            try:
                fs, data = scipy.io.wavfile.read(os.path.join(path, f))

                abs_data = np.abs(data)
                diff_data = np.diff(data)

                def calc_part_features(data, n=2, prefix=''):
                    f_i = 1
                    for i in range(0, len(data), len(data) // n):
                        features[f]['{}mean_{}_{}'.format(prefix, f_i, n)] = np.mean(data[i:i + len(data) // n])
                        features[f]['{}std_{}_{}'.format(prefix, f_i, n)] = np.std(data[i:i + len(data) // n])
                        features[f]['{}min_{}_{}'.format(prefix, f_i, n)] = np.min(data[i:i + len(data) // n])
                        features[f]['{}max_{}_{}'.format(prefix, f_i, n)] = np.max(data[i:i + len(data) // n])

                features[f]['len'] = len(data)
                if features[f]['len'] > 0:
                    n = 1
                    calc_part_features(data, n=n)
                    calc_part_features(abs_data, n=n, prefix='abs_')
                    calc_part_features(diff_data, n=n, prefix='diff_')

                    n = 2
                    calc_part_features(data, n=n)
                    calc_part_features(abs_data, n=n, prefix='abs_')
                    calc_part_features(diff_data, n=n, prefix='diff_')

                    n = 3
                    calc_part_features(data, n=n)
                    calc_part_features(abs_data, n=n, prefix='abs_')
                    calc_part_features(diff_data, n=n, prefix='diff_')

                cnt += 1

                # if cnt >= 1000:
                #     break

            except:
                pass

        features = pd.DataFrame(features).T.reset_index()
        features.rename(columns={'index': 'fname'}, inplace=True)

        return features


class Metrics(Preprocessing):

    def __init__(self):
        super().__init__()

    @staticmethod
    def apk(actual, predicted, k=10):
        """
        Computes the average precision at k.
        This function computes the average precision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    @staticmethod
    def mapk(actual, predicted, k=10):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        Parameters
        ----------
        actual : list
                 A list of lists of elements that are to be predicted
                 (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The mean average precision at k over the input lists
        """
        return np.mean([Metrics.apk(a, p, k) for a, p in zip(actual, predicted)])
