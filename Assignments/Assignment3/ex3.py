import pandas as pd
import datetime
import re
import nltk
import string
import math
import json
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, \
    confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
import pickle
from numpy.random.mtrand import seed
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from torch import nn
from torch.utils import data
from torch import tensor
import torch
import torch.optim as optim

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('word2vec_sample')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

stopwords_en = stopwords.words('english')
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                           reduce_len=True)
lemmatizer = WordNetLemmatizer()


def load_best_model():
    loaded_model = pickle.load(open('linear_svc_best_model.sav', 'rb'))
    return loaded_model


def train_best_model():
    df_tweets, df_splitted_tweets = load_and_preprocess_train_set()
    df_train, df_test = train_test_split(df_tweets)
    df_train_splitted, df_test_splitted = train_test_split_splitted(df_splitted_tweets)
    model = SVC(kernel="linear", C=5, class_weight='balanced', decision_function_shape='ovo', gamma='scale')
    inputs_train = np.vstack(df_train["embeddings_mean"])
    y_train = np.asarray(df_train["label"])
    model.fit(inputs_train, y_train)
    return model


def predict(m, fn):
    df_tweets = load_and_preprocess_test_set(fn)
    inputs_train = np.vstack(df_tweets["embeddings_mean"])
    predicts = m.predict(inputs_train)
    predicts = [int(x) for x in predicts]
    return predicts


def predict_and_save(m, fn):
    df_tweets = load_and_preprocess_test_set(fn)
    inputs_train = np.vstack(df_tweets["embeddings_mean"])
    predicts = m.predict(inputs_train)
    predicts = [int(x) for x in predicts]
    with open('test_results.txt', 'w') as f:
        f.write(' '.join([str(x) for x in predicts]))
    return predicts


def load_and_preprocess_train_set():
    # loading
    df_tweets = load_training_data()
    # dates
    df_tweets = format_dates(df_tweets)
    df_tweets = filter_by_date(df_tweets)
    # devices
    df_tweets = process_devices(df_tweets)
    # labeling
    df_tweets = labeling(df_tweets)
    # text cleaning and tokenization
    df_tweets = process_tweets_text(df_tweets)

    # generating same length sequences
    df_tweets = generate_splitted_tweet_tokens(df_tweets)

    df_splitted_tweets = create_splitted_df(df_tweets)
    df_splitted_tweets = get_embeddings_features_splitted(df_splitted_tweets)

    df_tweets = format_df(df_tweets)
    df_tweets = get_embeddings_features(df_tweets)
    return df_tweets, df_splitted_tweets


def load_and_preprocess_test_set(fn):
    # loading
    df_tweets = load_testing_data(fn)
    # dates
    df_tweets = format_dates(df_tweets)
    # text cleaning and tokenization
    df_tweets = process_tweets_text(df_tweets)

    df_tweets = get_embeddings_features_test(df_tweets)
    return df_tweets


def load_training_data(file_path="trump_train.tsv"):
    """Loading the data"""

    tweets = {
        "tweet_id": [],
        "user_handle": [],
        "tweet_text": [],
        "time_stamp": [],
        "device": []
    }
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line_splitted = line.split("\t")
            if len(line_splitted) > 5:
                print("line too long")
            tweets["tweet_id"].append(line_splitted[0].strip())
            tweets["user_handle"].append(line_splitted[1].strip())
            tweets["tweet_text"].append(line_splitted[2].strip())
            tweets["time_stamp"].append(line_splitted[3].strip())
            tweets["device"].append(line_splitted[4].strip())

    df = pd.DataFrame(tweets)
    return df


def load_testing_data(file_path="trump_test.tsv"):
    """Loading the data"""

    tweets = {
        "user_handle": [],
        "tweet_text": [],
        "time_stamp": [],
    }
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line_splitted = line.split("\t")
            if len(line_splitted) > 5:
                print("line too long")
            tweets["user_handle"].append(line_splitted[0].strip())
            tweets["tweet_text"].append(line_splitted[1].strip())
            tweets["time_stamp"].append(line_splitted[2].strip())
    df = pd.DataFrame(tweets)
    return df


"""##Prerocess
Cleaning and basic preprocessing
"""


# Adding a column representing the time of day
def extract_time_of_day(row):
    time_of_day = row["time_stamp"].to_pydatetime().time()
    datetime_of_day = datetime.datetime(1800, 1, 1, time_of_day.hour, time_of_day.minute, time_of_day.second)
    return datetime_of_day


def format_dates(df):
    # Formatting the date and time
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')

    df['time_of_day'] = df.apply(extract_time_of_day, axis=1)
    df['time_of_day'] = pd.to_datetime(df['time_of_day'], format='%Y-%m-%d %H:%M:%S')
    return df


def filter_by_date(df):
    # filtering tweets from before trumps presidency.
    trump_inauguration_date = datetime.datetime.strptime("2017-01-20 00:00:01", '%Y-%m-%d %H:%M:%S')
    # the date of the inauguration of Donald Trump is the date he took control over the potus twitter account.
    # This date is used to filter the tweets.
    # all the tweets from the 'PressSec', 'POTUS' handles will be discarded as they may not assist in the learning process.
    # This tweets are from the previous administration and as we aim to train a precise classifier
    # these tweets may add unwanted noise to the data.
    # These tweets may be "to easy" to separate from Trump's tweets, therefore both skewing the evaluation metrics and hindering the models' ability to separate the more similar tweets.
    return df[(df["user_handle"] == "realDonaldTrump") | (df["user_handle"].isin(['PressSec', 'POTUS'])) & (
            df["time_stamp"] > trump_inauguration_date)]


def convert_devices(row):
    a_href_regex = re.compile("<a+ (?!(?:href=[\"|']+[http:\/\/]*\.[\/]?.*?[\"|'])) *[^>]*>(.*?)<[/a][^>]>")

    a_href_device = a_href_regex.findall(row["device"])
    if len(a_href_device) > 0:  # for simplicity- all devices other than 'android' are labeled "other"
        return "another"
        # return a_href_device[0]
    else:
        return row["device"]


def process_devices(df):
    df["device"] = df.apply(convert_devices, axis=1)
    return df


"""###Label Tagging"""


def get_label(row, start_time_window=datetime.time(5, 0, 0), end_time_window=datetime.time(23, 0, 0)):
    """
    0-tweet by Trump,1- tweet by another
    """
    label = False
    if row["device"] == "android":
        label = False
    else:  # other devices
        if start_time_window <= row["time_stamp"].to_pydatetime().time() <= end_time_window:
            label = True
    return label


def labeling(df):
    df["label"] = df.apply(get_label, axis=1)
    df["tweet_length"] = df["tweet_text"].apply(lambda x: len(x))
    return df


"""###Brief data exploration

quick data exploration through visualizations
"""

# import seaborn as sns
#
# sns.set()
# sns.set(rc={'figure.figsize': (8, 8)})
# sns.displot(df_tweets, x="label", hue="device", discrete=True)
#
# """The labels are clearly unblanced. Over or under sampling may be used, or a different metric than Accuracy should be used (perhaps auc)"""
#
# sns.set(rc={'figure.figsize': (12, 8)})
# sns.histplot(df_tweets, x="time_stamp", kde=True, hue="device").set_title(
#     "Tweets distribution over dates, colored acoording to the devices used to tweet")
#
# sns.histplot(df_tweets, x="time_of_day", kde=True, hue="device").set_title(
#     "Tweets distribution over time of day, colored acoording to the devices used to tweet")
#
# sns.histplot(df_tweets, x="time_of_day", kde=True, hue="label").set_title(
#     "Tweets distribution over time of day, colored acoording to the devices used to tweet")
#
# df_tweets["tweet_length"] = df_tweets["tweet_text"].apply(lambda x: len(x))
#
# sns.histplot(df_tweets, x="tweet_length", kde=True, hue="device").set_title(
#     "Tweets' lengths distribution, colored acoording to the devices used to tweet")

"""###Features extraction

tokenization, stop words removal, length, embeddings...
"""


def clean_text(text):
    norm_text = text.lower()
    norm_text = re.sub('([.,!?()\n])', r' \1 ', norm_text)
    norm_text = re.sub('\s{2,}', ' ', norm_text)
    norm_text = re.sub(r"\b's\b", '', norm_text)
    norm_text = re.sub(r'https?:\/\/.*[\r\n]*', '', norm_text)
    norm_text = re.sub(r'#', '', norm_text)
    return norm_text


def get_tweet_tokens(tweet):
    tokens = tokenizer.tokenize(tweet)
    return tokens


def process_tweet(tweet, remove_punctuations=True, remove_stopwords=True, lemmatization=True):
    pos_translation = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
    global stopwords_en
    norm_tokens = []
    norm_tokens_pos_tag = []  # (token, lemmetize_token, stem_token pos_type_1, pos_type_2) maybe add tf-idf

    clean_tweet = clean_text(tweet)
    tweet_tokens = get_tweet_tokens(clean_tweet)
    for tweet_token in tweet_tokens:
        if remove_punctuations:
            if tweet_token in string.punctuation:
                continue  # skip punctuation
        if remove_stopwords:
            if tweet_token in stopwords_en:
                continue  # skip punctuation

        # POS Tagging
        pos_type_1 = nltk.pos_tag([tweet_token])[0][1]

        # lemmatization:
        if pos_type_1[:2] in pos_translation:
            pos_type_2 = pos_translation[pos_type_1[:2]]
        else:
            pos_type_2 = 'n'  # default to noun
        lem_token = lemmatizer.lemmatize(tweet_token, pos=pos_type_2)

        # stemming
        # stem_token = stemmer.stem(tweet_token)  # stemming word

        if lemmatization:
            norm_tokens.append(lem_token)
        else:
            norm_tokens.append(tweet_token)
        # norm_tokens_pos_tag.append((tweet_token,lem_token,stem_token,pos_type_1,pos_type_2))

    # return norm_tokens, norm_tokens_pos_tag
    # return norm_tokens_pos_tag
    return norm_tokens


def process_tweets_text(df):
    df["tweet_tokens"] = df["tweet_text"].apply(process_tweet)
    return df


"""###TF-IDF: Not used, code removed"""

"""###Creating sequences of the same length
chosen length = 10
every tweet is sliced into sequences of 10

Example:
"General Kelly is doing a great job at the border. Numbers are way down. Many are not even trying to come in anymore."

has been tokenized to:
*   ['general', 'kelly', 'great', 'job', 'border', 'number', 'way', 'many', 'even', 'try', 'come', 'anymore']

and then split into:
1.   ['general', 'kelly', 'great', 'job', 'border', 'number', 'way', 'many', 'even', 'try']
2.   ['great', 'job', 'border', 'number', 'way', 'many', 'even', 'try', 'come', 'anymore']
"""


def pad_tweet(tweet, pad_str='~', n=10):
    padded_tweet = tweet
    pad_count = n - len(tweet)
    for _ in range(pad_count):
        padded_tweet.append(pad_str)
    return padded_tweet


def split_tweet(tweet, n=10):
    splited_tweet = []
    num_of_splits = math.ceil(len(tweet) / n)

    if num_of_splits == 1:  # length of tweet is shorter than n
        splited_tweet.append(pad_tweet(tweet))
        # splited_tweet.append(list(pad_sequence(tweet, n=n)))
    else:
        for i in range(num_of_splits):
            start = i * n
            end = start + n
            if end > len(tweet):  # if the tweet is ending slice a portion
                splited_tweet.append(tweet[-n:])
            else:
                splited_tweet.append(tweet[start:end])
    return splited_tweet


def generate_splitted_tweet_tokens(df):
    df["splitted_tweet_tokens"] = df["tweet_tokens"].apply(split_tweet)
    return df


def create_splitted_df(df):
    rows = []
    for _, row in df.iterrows():
        for tweet_section in row["splitted_tweet_tokens"]:
            rows.append({'tweet_id': row['tweet_id'], 'time_stamp': row['time_stamp'], 'tokens': tweet_section,
                         'tweet_length': row['tweet_length'], 'label': row['label']})
    df_splitted = pd.DataFrame(rows)
    return df_splitted


def format_df(df):
    df = df[['tweet_id', 'time_stamp', 'tweet_tokens', 'label', 'tweet_length']]
    return df


"""###Embeddings"""

# !pip install fasttext

import pickle


# import fasttext.util

# def generate_embedding_dict(embed_path = "embeddings.pkl"):
#   fasttext.util.download_model('en', if_exists='ignore')  # English
#   ft = fasttext.load_model('cc.en.300.bin')
#   count=0
#   words_not_in_embed_vocab = set()
#   embeddings_dict = {}
#   for tweet in df_splitted_tweets["tokens"]:
#     for word in tweet:
#       if word not in ft.words:
#         words_not_in_embed_vocab.add(word)
#         count=count+1
#       embeddings_dict[word]= ft.get_word_vector(word)
#   print(count)
#   #Saving the embeddings
#   embeds_file = open(embed_path, "wb")
#   pickle.dump(embeddings_dict, embeds_file)
#   embeds_file.close()
#   return embeddings_dict, words_not_in_embed_vocab

def load_embedding_dict(embed_path="embeddings.pkl"):
    embeds_file = open(embed_path, "rb")
    embeddings_dict = pickle.load(embeds_file)
    return embeddings_dict


embeds_dict = load_embedding_dict()


def get_embeddings(tweet):
    global embeds_dict
    tweet_embeds = []
    for word in tweet:
        if word not in embeds_dict:
            # add the word not in the vocab of the embeddings model with a random vector
            embeds_dict[word] = np.random.uniform(low=-0.2, high=0.2, size=(len(embeds_dict["word"]),))
        tweet_embeds.append(embeds_dict[word])
    return np.asarray(tweet_embeds)


def get_embeddings_features_splitted(df):
    df["tokens_embeddings"] = df["tokens"].apply(get_embeddings)
    df = df.dropna()
    df["embeddings_mean"] = df["tokens_embeddings"].apply(
        lambda x: np.mean(x, axis=0) if len(x.shape) == 2 else np.zeros(300))
    return df


def get_embeddings_features(df):
    df["tokens_embeddings"] = df["tweet_tokens"].apply(get_embeddings)
    df["embeddings_mean"] = df["tokens_embeddings"].apply(
        lambda x: np.mean(x, axis=0) if len(x.shape) == 2 else np.zeros(300))
    df = df.dropna()
    return df


def get_embeddings_features_test(df):
    df["tokens_embeddings"] = df["tweet_tokens"].apply(get_embeddings)
    # df["embeddings_mean"] = df["tokens_embeddings"].apply(lambda x: np.mean(x, axis=0))
    df["embeddings_mean"] = df["tokens_embeddings"].apply(
        lambda x: np.mean(x, axis=0) if len(x.shape) == 2 else np.zeros(300))
    return df


"""##Train/Test Split
Split to train and test sets based on tweet date. the models will be trained on early tweets and evaluated on more recent ones.

"""


def train_test_split(df):
    split_by_date = datetime.datetime.strptime("2016-07-27 00:00:01", '%Y-%m-%d %H:%M:%S')
    df_train = df[df['time_stamp'] < split_by_date]
    df_test = df[df['time_stamp'] >= split_by_date]

    print("df_tweets")
    print(f"Total size: {len(df)}")
    print(f"Train set size: {len(df_train)}, Percentage: {len(df_train) / len(df) * 100}")
    print(f"Test set size: {len(df_test)}, Percentage: {len(df_test) / len(df) * 100}")
    return df_train, df_test


def train_test_split_splitted(df):
    split_by_date = datetime.datetime.strptime("2016-07-27 00:00:01", '%Y-%m-%d %H:%M:%S')
    df_train_splitted = df[df['time_stamp'] < split_by_date]
    df_test_splitted = df[df['time_stamp'] >= split_by_date]

    print("\ndf_splitted_tweets")
    print(f"Total size: {len(df)}")
    print(f"Train set size: {len(df_train_splitted)}, Percentage: {len(df_train_splitted) / len(df) * 100}")
    print(f"Test set size: {len(df_test_splitted)}, Percentage: {len(df_test_splitted) / len(df) * 100}")
    return df_train_splitted, df_test_splitted


"""##Models

###SKLearn Models
"""


# !mkdir. / trained_models
# !mkdir. / trained_models / ada_boost
# !mkdir. / trained_models / logistic_regression
# !mkdir. / trained_models / svm_linear
# !mkdir. / trained_models / svm_non_linear


class SKLearnInference(object):
    def __init__(self):
        self.model_instance = None
        self.base_path = None
        self.parameters = None
        self.best_params = None

    def fit(self, x_train, y_train):
        auc_scorer = make_scorer(roc_auc_score)
        # ~~ COUDN'T USE OVER AND UNDER SAMPLING BECAUSE OF THE ASSIGNMENT GUIDLINES!
        # over and under sampling
        # over_sampler = SMOTE()
        # under_sampler = RandomUnderSampler()
        # self.parameters['o__sampling_strategy'] = [lambda y: Counter(y_train), 0.5, 1]
        # self.parameters['u__sampling_strategy'] = [lambda y: Counter(y_train), 1]
        # pipeline
        # pipeline = Pipeline(steps=[('o', over_sampler), ('u', under_sampler), ('m', self.model_instance)])
        # Run the grid search
        self.grid_obj = GridSearchCV(self.model_instance, self.parameters,
                                     scoring=auc_scorer,
                                     return_train_score=True, n_jobs=-1, verbose=4)
        self.grid_obj = self.grid_obj.fit(x_train, y_train)

        scores = pd.DataFrame(self.grid_obj.cv_results_)
        print("mean_test_score: ", max(scores['mean_test_score']))
        best_params = self.grid_obj.best_params_
        print("best_params:")
        print(best_params)

        self.model_instance = self.grid_obj.best_estimator_
        self.model_instance.fit(x_train, y_train)
        self.best_params = best_params

    def evaluate(self, x_dev, y_dev):
        predicts_dev = self.model_instance.predict(x_dev)
        auc = roc_auc_score(y_dev, predicts_dev)
        acc = accuracy_score(y_dev, predicts_dev)
        cls_report = classification_report(y_dev, predicts_dev)
        conf_mat = confusion_matrix(y_dev, predicts_dev)
        return auc, acc, cls_report, conf_mat

    def save(self, num):
        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path)

        filename = self.base_path + "\\inference_model_{}.json".format(num)
        pickle.dump(self.model_instance, open(filename, 'wb'))

        best_params_filename = self.base_path + "\\inference_model_{}_best_params.json".format(num)
        with open(best_params_filename, 'wb') as fp:
            pickle.dump(self.grid_obj.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved model to disk num:{}".format(num))

    def load(self, num):
        if not os.path.isdir(self.base_path):
            print("no model was saved yet")
        else:
            # load json and create model
            filename = self.base_path + "\\inference_model_{}.json".format(num)
            self.model_instance = pickle.load(open(filename, 'rb'))
            print("Loaded model from disk")

    def predict(self, x, probs=False):
        if probs:
            predicts = self.model_instance.predict_proba(x)
        else:
            predicts = self.model_instance.predict(x)
        return predicts

    def predict_prob(self, x):
        predicts_prob = self.model_instance.predict_proba(x)
        return predicts_prob

    def get_best_params(self, string=False):
        if string:
            return json.dumps(self.best_params)
        else:
            return self.best_params


def train_sklearn_inference_model(model, inputs_train, y_train, save_index, init_seed, kernel=None):
    # Setting the random seed
    seed(init_seed)
    # ~~ initializing the model
    if kernel is None:
        sklearn_inference_model = model(init_seed)
    else:
        sklearn_inference_model = model(init_seed, kernel)
    # ~~ Train on the training set
    start = time.time()
    sklearn_inference_model.fit(inputs_train, y_train)
    end = time.time()
    print(f"training time (seconds) {end - start}")
    sklearn_inference_model.save(save_index)

    return sklearn_inference_model


"""### Logistic Regression"""


class LogisticRegressionInference(SKLearnInference):
    def __init__(self, random_state):
        super().__init__()
        self.model_instance = LogisticRegression(random_state=random_state)
        self.base_path = "./trained_models/random_forest"
        self.parameters = {'class_weight': [None, 'balanced'],
                           # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
                           'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25],
                           'penalty': ['l1', 'l2'],
                           'solver': ['liblinear', 'saga']
                           }


def train_and_evaluate(sklearn_model, df_train, df_test):
    random_seed = 0
    save_index = 0
    model = train_sklearn_inference_model(model=sklearn_model,
                                          inputs_train=np.vstack(df_train["embeddings_mean"]),
                                          y_train=np.asarray(df_train["label"]),
                                          save_index=save_index,
                                          init_seed=random_seed)

    auc, acc, cls_report, conf_matrix = model.evaluate(x_dev=np.vstack(df_test["embeddings_mean"]),
                                                       y_dev=np.asarray(df_test["label"]))
    print(f"auc: {auc}")
    print(f"accuracy: {acc}")
    print(cls_report)
    fig, ax = plt.subplots(figsize=(3, 3))
    ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot(values_format='d', ax=ax)


# train_data = np.array([np.array(token_embeddings) for token_embeddings in df_train_splitted.tokens_embeddings])
# train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2])
# test_data = np.array([np.array(token_embeddings) for token_embeddings in df_test_splitted.tokens_embeddings])
# test_data = test_data.reshape(test_data.shape[0], test_data.shape[1]*test_data.shape[2])

# logistic_regression_model_seq = train_sklearn_inference_model(model=LogisticRegressionInference, inputs_train=train_data, y_train=np.asarray(df_train_splitted["label"]), save_index=save_index, init_seed=random_seed)

# auc_logistic_regression_seq, acc_logistic_regression_seq, cls_report_logistic_regression_seq, conf_matrix_logistic_regression_seq = logistic_regression_model_seq.evaluate(x_dev=test_data, y_dev=np.asarray(df_test_splitted["label"]))
# print(f"auc: {auc_logistic_regression_seq}")
# print(f"accuracy: {acc_logistic_regression_seq}")
# print(cls_report_logistic_regression_seq)
# ConfusionMatrixDisplay(conf_matrix_logistic_regression_seq, display_labels=[0,1]).plot(values_format='d')

"""### SVM Linear Kernel"""

from sklearn.svm import SVC


class SVCInference(SKLearnInference):
    def __init__(self, random_state, kernel):
        super().__init__()
        self.model_instance = SVC(random_state=random_state, kernel=kernel)
        self.base_path = "./trained_models/SVC/" + kernel
        self.parameters = {'class_weight': [None, 'balanced'],
                           # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
                           'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25],
                           'decision_function_shape': ['ovo', 'ovr'],
                           'gamma': ['scale', 'auto']
                           }


# linear_svc_model = train_sklearn_inference_model(model=SVCInference,
#                                                  inputs_train=np.vstack(df_train["embeddings_mean"]),
#                                                  y_train=np.asarray(df_train["label"]), save_index=save_index,
#                                                  init_seed=random_seed, kernel='linear')


"""### SVM Non-Linear Kernel"""

# rbf_svc_model = train_sklearn_inference_model(model=SVCInference, inputs_train=np.vstack(df_train["embeddings_mean"]),
#                                               y_train=np.asarray(df_train["label"]), save_index=save_index,
#                                               init_seed=random_seed, kernel='rbf')


"""### Ada-Boost Classifier"""


class AdaBoostInference(SKLearnInference):
    def __init__(self, random_state):
        super().__init__()
        self.model_instance = AdaBoostClassifier(random_state=random_state)
        self.base_path = "./trained_models/ada_boost"
        base_estimators = []
        for max_features in ['log2', 'sqrt']:
            for criterion in ['entropy', 'gini']:
                for max_depth in [5, 10]:
                    for min_samples_split in [2, 3, 5]:
                        for min_samples_leaf in [1, 5, 8]:
                            base_estimators.append(DecisionTreeClassifier(
                                max_features=max_features, criterion=criterion, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf))
        self.parameters = {'n_estimators': [100, 150, 300],
                           'base_estimator': base_estimators
                           }

    def get_best_params(self, string=False):
        if string:
            self.best_params['base_estimator'] = str(self.best_params['base_estimator'])
            return super().get_best_params(string=string)
        else:
            return super().get_best_params(string=string)


# ada_boost_model = train_sklearn_inference_model(model=AdaBoostInference,
#                                                 inputs_train=np.vstack(df_train["embeddings_mean"]),
#                                                 y_train=np.asarray(df_train["label"]), save_index=save_index,
#                                                 init_seed=random_seed)


"""##PyTorch Models

###Neural Network Classifier
"""


class NNModel(nn.Module):
    def __init__(self, layers_sizes):  # vocab_size,
        super(NNModel, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(layers_sizes[0], layers_sizes[1]),
            nn.ReLU(),
            nn.Linear(layers_sizes[1], layers_sizes[2]),
            nn.ReLU(),
            nn.Linear(layers_sizes[2], layers_sizes[3]),
            nn.ReLU(),
            nn.Linear(layers_sizes[3], layers_sizes[4])
        )

    def forward(self, x):
        logits = torch.sigmoid(self.linear_relu_stack(x))
        return logits


def train_epoch(model, opt, crit, dataloader):
    model.train(mode=True)  # setting the model to training mode

    for idx, (x, y) in enumerate(dataloader):
        predicted_label = model(x)
        # print(predicted_label)
        loss = crit(predicted_label, y)

        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate_nn(model, x_dev, y_dev):
    model.eval()
    with torch.no_grad():
        predicts_dev = model(x_dev).round()
        # print(predicts_dev)
        auc = roc_auc_score(y_dev, predicts_dev)
        acc = accuracy_score(y_dev, predicts_dev)
        cls_report = classification_report(y_dev, predicts_dev)
        conf_mat = confusion_matrix(y_dev, predicts_dev)
        return auc, acc, cls_report, conf_mat


def evaluate_nn_auc(model, x_dev, y_dev):
    model.eval()
    with torch.no_grad():
        predicts_dev = model(x_dev).round()
        # print(predicts_dev)
        auc = roc_auc_score(y_dev, predicts_dev)
    return auc


def evaluate(model, crit, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            predicted_label = model(x)
            # print(predicted_label)
            # print(predicted_label.argmax(1))
            # print(predicted_label.round())
            # loss = crit(predicted_label, y)
            total_acc += (predicted_label.round() == y).sum().item()
            total_count += y.size(0)
    return total_acc / total_count


def train(model, opt, crit, scheduler, train_dataloader, test_dataloader, test_data_tensor, test_labels_tensor, epochs):
    total_accuracy = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_epoch(model=model, opt=opt, crit=crit, dataloader=train_dataloader)
        accuracy_val = evaluate(model=model, crit=crit, dataloader=test_dataloader)
        scheduler.step()
        # auc_val = evaluate_nn_auc(model, x_dev=test_data_tensor, y_dev=np.asarray(df_test["label"]))
        auc_val = evaluate_nn_auc(model, x_dev=test_data_tensor, y_dev=test_labels_tensor)
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accuracy_val))
        print(f"auc_val: {auc_val}, last_lr: {str(scheduler.get_last_lr())}")


# Data preperation
def arrange_data_for_nn(train_data, test_data, train_labels, test_labels, batch_size=16):
    train_data_tensor = tensor(train_data)
    test_data_tensor = tensor(test_data)

    train_labels_tensor = tensor(train_labels)
    test_labels_tensor = tensor(test_labels)
    # labels_tr = np.asarray(train_df["label"]).astype(int)
    # labels_te = np.asarray(test_df["label"]).astype(int)

    # train_label = tensor(np.eye(np.max(labels_tr) + 1)[labels_tr].astype(np.float32))
    # test_label = tensor(np.eye(np.max(labels_te) + 1)[labels_te].astype(np.float32))

    train_tensor = data.TensorDataset(train_data_tensor, train_labels_tensor)
    test_tensor = data.TensorDataset(test_data_tensor, test_labels_tensor)

    trainloader = data.DataLoader(train_tensor, batch_size=batch_size,
                                  shuffle=True)
    testloader = data.DataLoader(test_tensor, batch_size=batch_size,
                                 shuffle=False)
    return trainloader, testloader, test_data_tensor, test_labels_tensor


def train_nn_model(df_train, df_test):
    trainloader, testloader, test_data_tensor, test_labels_tensor = arrange_data_for_nn(
        train_data=np.vstack(df_train["embeddings_mean"]), test_data=np.vstack(df_test["embeddings_mean"]),
        train_labels=np.vstack(df_train["label"]).astype(np.float32),
        test_labels=np.vstack(df_test["label"]).astype(np.float32))

    # Hyperparameters
    input_size = np.vstack(df_train["embeddings_mean"]).shape[1]
    output_size = 1
    layers_sizes = [input_size, int(input_size / 2), int(input_size / 4), int(input_size / 8), output_size]

    epochs = 300  # epoch
    learning_rate = 0.1  # learning rate

    # Model initialization
    nn_model = NNModel(layers_sizes=layers_sizes)
    print(nn_model)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(nn_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)

    train(model=nn_model, opt=optimizer, crit=criterion, scheduler=scheduler, train_dataloader=trainloader,
          test_dataloader=testloader, test_data_tensor=test_data_tensor, test_labels_tensor=test_labels_tensor,
          epochs=epochs)

    auc_nn, acc_nn, cls_report_nn, conf_matrix_nn = evaluate_nn(nn_model, x_dev=test_data_tensor,
                                                                y_dev=test_labels_tensor)
    print(f"auc: {auc_nn}")
    print(f"accuracy: {acc_nn}")
    print(cls_report_nn)
    fig, ax = plt.subplots(figsize=(3, 3))
    ConfusionMatrixDisplay(conf_matrix_nn, display_labels=[0, 1]).plot(values_format='d', ax=ax)


# train_data = np.array([np.array(token_embeddings) for token_embeddings in df_train_splitted.tokens_embeddings])
# train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
# test_data = np.array([np.array(token_embeddings) for token_embeddings in df_test_splitted.tokens_embeddings])
# test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
#
# # Hyperparameters
# input_size = 3000
# output_size = 1
# layers_sizes = [input_size, int(input_size / 2), int(input_size / 4), int(input_size / 8), output_size]
#
# epochs = 300  # epoch
# learning_rate = 0.1  # learning rate
# batch_size = 16  # batch size for training
#
# # Model initialization
# nn_model_seq = NNModel(layers_sizes=layers_sizes)
# print(nn_model_seq)
#
# criterion = nn.BCELoss()
# optimizer = optim.SGD(nn_model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)
#
# trainloader, testloader, test_data_tensor, test_labels_tensor = arrange_data_for_nn(train_data=train_data,
#                                                                                     test_data=test_data,
#                                                                                     train_labels=np.vstack(
#                                                                                         df_train_splitted[
#                                                                                             "label"]).astype(
#                                                                                         np.float32),
#                                                                                     test_labels=np.vstack(
#                                                                                         df_test_splitted[
#                                                                                             "label"]).astype(
#                                                                                         np.float32))
#
# train(model=nn_model_seq, opt=optimizer, crit=criterion, scheduler=scheduler, train_dataloader=trainloader,
#       test_dataloader=testloader, test_data_tensor=test_data_tensor, test_labels_tensor=test_labels_tensor,
#       epochs=epochs)
#
# auc_nn, acc_nn, cls_report_nn, conf_matrix_nn = evaluate_nn(nn_model, x_dev=test_data_tensor, y_dev=test_labels_tensor)
# print(f"auc: {auc_nn}")
# print(f"accuracy: {acc_nn}")
# print(cls_report_nn)
# fig, ax = plt.subplots(figsize=(3, 3))
# ConfusionMatrixDisplay(conf_matrix_nn, display_labels=[0, 1]).plot(values_format='d', ax=ax)

"""###Recurent Neural Network"""


class LSTMRNNModel(nn.Module):
    def __init__(self, layers_sizes, input_size, hidden_dim, num_layers):  # embedding_dim=300, hidden_dim=200
        super(LSTMRNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_layers = num_layers
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size,
                            hidden_dim,
                            bidirectional=False,
                            batch_first=True,
                            num_layers=2)
        # x needs to be: (batch_size, seq, input_size)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(layers_sizes[0], layers_sizes[1]),
            nn.ReLU(),
            nn.Linear(layers_sizes[1], layers_sizes[2])
        )

    def forward(self, x):
        batch_len = x.size(0)
        h0 = torch.randn(self.num_layers, batch_len, self.hidden_dim)
        c0 = torch.randn(self.num_layers, batch_len, self.hidden_dim)

        lstm_out, _ = self.lstm(x, (h0, c0))
        # lstm_out: tensor of shape (batch_size, seq_length, hidden_size)
        # lstm_out: (batch_size, 10, 200)

        # we only want the output of the last time step - > (batch_size, hidden_size)
        lstm_out = lstm_out[:, -1, :]

        out = torch.sigmoid(self.linear_relu_stack(lstm_out))
        return out


def train_lstm_model(df_train_splitted, df_test_splitted):
    train_data = np.array([np.array(token_embeddings) for token_embeddings in df_train_splitted.tokens_embeddings])
    test_data = np.array([np.array(token_embeddings) for token_embeddings in df_test_splitted.tokens_embeddings])

    trainloader, testloader, test_data_tensor, test_labels_tensor = arrange_data_for_nn(
        train_data=train_data, test_data=test_data,
        train_labels=np.vstack(df_train_splitted["label"]).astype(np.float32),
        test_labels=np.vstack(df_test_splitted["label"]).astype(np.float32))

    # Hyperparameters
    input_size = 300
    output_size = 1
    hidden_dim = 128
    sequence_length = 10
    num_layers = 2
    layers_sizes = [hidden_dim, int(hidden_dim / 4), output_size]  # hidden_dim*2 for bidirectional lstm

    epochs = 300  # epoch
    learning_rate = 0.1  # learning rate
    batch_size = 16  # batch size for training

    # Model initialization
    lstm_model = LSTMRNNModel(layers_sizes=layers_sizes, input_size=input_size, hidden_dim=hidden_dim,
                              num_layers=num_layers)
    print(lstm_model)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(lstm_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.8)

    train(model=lstm_model, opt=optimizer, crit=criterion, scheduler=scheduler, train_dataloader=trainloader,
          test_dataloader=testloader, test_data_tensor=test_data_tensor, test_labels_tensor=test_labels_tensor,
          epochs=epochs)

    auc_lstm, acc_lstm, cls_report_lstm, conf_matrix_lstm = evaluate_nn(lstm_model, x_dev=test_data_tensor,
                                                                        y_dev=test_labels_tensor)
    print(f"auc: {auc_lstm}")
    print(f"accuracy: {acc_lstm}")
    print(cls_report_lstm)
    fig, ax = plt.subplots(figsize=(3, 3))
    ConfusionMatrixDisplay(conf_matrix_lstm, display_labels=[0, 1]).plot(values_format='d', ax=ax)


# if __name__ == '__main__':
#     train_best_model('PyCharm')
m = train_best_model()
preds = predict(m=m, fn="trump_test.tsv")
predict_and_save(m=m, fn="trump_test.tsv")
print(preds)
print(len(preds))
