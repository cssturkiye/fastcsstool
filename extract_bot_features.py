import pandas as pd
import numpy as np
from datetime import datetime
import pytz

import os
import pandas as pd
from multiprocessing import Pool
import onnxruntime as rt


# Adjusting the Features class to correctly use the keys from mapped_row
class Features():
    def __init__(self, t):
        # self.account_age = self.calc_user_age(t['usrLastTweetDate'], t['usrCreated'])     # this is dynamic, referenced to the last tweet date but the usrLastTweetDate field is not really represent the last tweet date
        last_tweet_time = datetime.strptime('2023-02-18 00:00:00+00:00', '%Y-%m-%d %H:%M:%S%z')
        self.account_age = self.calc_user_age(last_tweet_time, t['usrCreated']) # this is static, referenced to the API call date

        self.statuses_count = t['usrStatusesCount']
        self.followers_count = t['usrFollowersCount']
        self.friends_count = t['usrFriendsCount']
        self.favourites_count = t['usrFavouritesCount']
        self.listed_count = t['usrListedCount']

        self.profile_use_background_image = 1 if t['usrProfileBannerUrl'] else 0
        self.verified = t['usrVerified']

        self.tweet_freq = self.statuses_count / self.account_age if self.account_age != 0 else 0
        self.followers_growth_rate = self.followers_count / self.account_age if self.account_age != 0 else 0
        self.friends_growth_rate = self.friends_count / self.account_age if self.account_age != 0 else 0
        self.favourites_growth_rate = self.favourites_count / self.account_age if self.account_age != 0 else 0
        self.listed_growth_rate = self.listed_count / self.account_age if self.account_age != 0 else 0
        
        self.followers_friends_ratio = self.followers_count / (self.friends_count + 1)
        self.statuses_followers_ratio = self.statuses_count / (self.followers_count + 1)

        self.screen_name_length = len(t['usr'])
        self.num_digits_in_screen_name = self.count_numerical_chars(t['usr'])
        self.name_length = len(t['usrDn'])
        self.num_digits_in_name = self.count_numerical_chars(t['usrDn'])

        self.description_length = len(t['usrDes'])
        
        self.has_url = 1 if t['usrLink'] else 0
        self.has_desc_url = 1 if t['usrDesLinks'] else 0
        
        self.location = 1 if t['usrLocation'] else 0
        self.hour_created = t['usrCreated'].hour
        self.network = np.log(1 + self.statuses_count) * np.log(1 + self.followers_count)
        
        # self.label = None  # Placeholder as label is not provided in the dataset

    def count_numerical_chars(self, string):
        return sum(char.isnumeric() for char in string)
    
    def calc_user_age_old(self, last_tweet_time, user_creation_time):
        return (last_tweet_time - user_creation_time).total_seconds() / 86400
    
    def calc_user_age(self, last_tweet_time, user_creation_time):
        user_creation_time = user_creation_time.replace(tzinfo=pytz.UTC)
        return (last_tweet_time - user_creation_time).total_seconds() / 86400
    
# Function to handle potential NaN values in string columns
def handle_nan_string(value):
    return str(value) if not pd.isna(value) else ''


def extract_features_from_df(df):
    # Clearing previous list
    features_list = []

    # Extracting features for each row in the dataset
    for _, row in df.iterrows():
        # Correctly mapping the CSV column names to the expected keys in the Features class
        mapped_row = {
            'usrLastTweetDate': pd.to_datetime(row['created_at']),
            'usrCreated': pd.to_datetime(row['author_created_at']),
            'usrStatusesCount': row['author_tweets'],
            'usrFollowersCount': row['author_followers'],
            'usrFriendsCount': row['author_following'],
            'usrFavouritesCount': row['likes'],  # TODO: Bu bilgi yanlis bu alan kullanilmiyor.
            'usrListedCount': row['author_listed'],
            'usrProfileBannerUrl': row['author_pic'],  # TODO: Bu bilgi yanlis bu alan kullanilmiyor.
            'usrVerified': row['author_verified'],
            'usr': handle_nan_string(row['username']),
            'usrDn': handle_nan_string(row['name']),
            'usrDes': handle_nan_string(row['author_description']),
            'usrLink': 'http' in handle_nan_string(row['text']),  # TODO: Bu bilgi yanlis bu alan kullanilmiyor.
            'usrDesLinks': 'http' in handle_nan_string(row['author_description']),
            'usrLocation': handle_nan_string(row['author_location'])
        }

        feature = Features(mapped_row)
        features_list.append(vars(feature))

    # Convert the list of feature objects to a DataFrame for easier viewing
    features_df = pd.DataFrame(features_list)
    return df, features_df



# BOT DETECTION - - - - 
# Load Model
providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
sess_options = rt.SessionOptions()
sess = rt.InferenceSession("models/pipeline_xgboost_wo_rates.onnx"
                           , sess_options=sess_options, providers=providers
)


def process_batches(df_features_all, batch_size=100):
    # Calculate number of batches
    num_batches = int(np.ceil(len(df_features_all) / batch_size))

    # Initialize empty lists to store results
    bot_probs = []
    is_bots = []

    # df_features_all['created_at'] = pd.to_datetime(df_features_all['created_at']).astype('int64') / 10**9

    print(df_features_all.head())   

    for i in range(num_batches):
        # Get the current batch
        batch = df_features_all[i*batch_size : (i+1)*batch_size]
        input_ = np.array(batch.values)

        # Run model prediction
        pred_onx = sess.run(None, {"input": input_.reshape(-1, input_[0].shape[0]).astype(np.float32)})

        # Append results to lists
        bot_probs.extend([prob[1] for prob in pred_onx[1]])  # the second class is 'bot'
        is_bots.extend(pred_onx[0])

    return bot_probs, is_bots



def process_df(df):
    df, features_df = extract_features_from_df(df)
    features_df.reset_index(drop=True, inplace=True)
    features = [
        'account_age',
        'statuses_count',
        'followers_count',
        'friends_count',
        'listed_count',
        'verified',
        'followers_friends_ratio',
        'statuses_followers_ratio',
        'screen_name_length',
        'num_digits_in_screen_name',
        'name_length',
        'num_digits_in_name',
        'description_length',
        'has_desc_url',
        'location',
        'hour_created',
        'network',
        # 'favourites_count',                   # Bu veri elimizde yok
        # 'profile_use_background_image',       # Bu veri elimizde yok
        # 'tweet_freq',                         # Bu veri elimizde yok
        # 'followers_growth_rate',              # Bu veri elimizde yok
        # 'friends_growth_rate',                # Bu veri elimizde yok
        # 'favourites_growth_rate',             # Bu veri elimizde yok
        # 'listed_growth_rate',                 # Bu veri elimizde yok
        # 'has_url',                            # Bu veri elimizde yok
    ]
    features_df = features_df[features]

    bot_probs, is_bots = process_batches(features_df)

    # count true and false values inside is_bots list
    print("Bot count  :", is_bots.count(True))
    print("Human count:", is_bots.count(False))

    # Add predictions to df_all
    df.loc[:, 'bot_prob'] = bot_probs
    df.loc[:, 'is_bot']   = is_bots

    # df = pd.concat([df, features_df], axis=1)
    
    return df