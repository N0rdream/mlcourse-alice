import pickle
import numpy as np
import pandas as pd
import datetime
import os
import warnings
from scipy.sparse import csr_matrix, hstack
from sklearn import preprocessing as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore")


PATH_TO_DATA = 'alice_data'
AUTHOR = 'Vladimir_Sapachev' 


def write_to_submission_file(
    predicted_labels,
    target='target',
    index_label="session_id"
):
    predicted_df = pd.DataFrame(
        predicted_labels,
        index = np.arange(1, predicted_labels.shape[0] + 1),
        columns=[target]
    )
    predicted_df.to_csv(os.path.join(
        PATH_TO_DATA, f'submission_alice_{AUTHOR}_solution.csv'), index_label=index_label)


def get_labels(label):
    return [f'{label}{i}' for i in range(1, 11)]

times = get_labels('time')
sites = get_labels('site')
links = get_labels('link')

# load data into dataframes
train_df = pd.read_csv(
    os.path.join(PATH_TO_DATA, 'train_sessions.csv'), 
    index_col='session_id', 
    parse_dates=times
)

test_df = pd.read_csv(
    os.path.join(PATH_TO_DATA, 'test_sessions.csv'), 
    index_col='session_id', 
    parse_dates=times
)

train_df = train_df.sort_values(by='time1')
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)

# get rid of data when Alice was active at 9 o'clock
nine_oclock_date = datetime.date(2013, 4, 12)
train_df = train_df[~(train_df.time1.dt.date == nine_oclock_date)]

# add sites links to dataframes
with open(os.path.join(PATH_TO_DATA, 'site_dic.pkl'), "rb") as input_file:
    site_dict = pickle.load(input_file)
    
site_dict_reverse = {v: k for k, v in site_dict.items()}

for link, site in zip(links, sites):
    train_df[link] = train_df[site].map(site_dict_reverse)
    
for link, site in zip(links, sites):
    test_df[link] = test_df[site].map(site_dict_reverse)

# save links and sites codes into files for vectorizing
train_df[links].fillna(0).to_csv(
    os.path.join(PATH_TO_DATA, 'train_df_links.txt'), 
    sep=' ', 
    index=None, 
    header=None
)

test_df[links].fillna(0).to_csv(
    os.path.join(PATH_TO_DATA, 'test_df_links.txt'), 
    sep=' ', 
    index=None, 
    header=None
)

train_df[sites].astype('int').to_csv(
    os.path.join(PATH_TO_DATA, 'train_df_sites.txt'), 
    sep=' ', 
    index=None,
    header=None
)

test_df[sites].astype('int').to_csv(
    os.path.join(PATH_TO_DATA, 'test_df_sites.txt'), 
    sep=' ', 
    index=None,
    header=None
)

# concatenate train/test data
full_df = pd.concat([train_df, test_df])
idx_split = train_df.shape[0]
full_df['start_month'] = full_df.time1.apply(lambda ts: ((ts.year - 2013) * 12 + ts.month)).astype(int)
full_df['date'] = full_df.time1.dt.year * 10000 + full_df.time1.dt.month * 100 + full_df.time1.dt.day

# get sites that was visited only by Alice and sites that Alice did not visit at all
def get_unique_sites(n, m, is_alice=True):
    
    alice_links = set(train_df[train_df.target == 1].link1.value_counts(normalize=True).head(n).index.tolist())
    other_links = set(train_df[train_df.target == 0].link1.value_counts(normalize=True).head(m).index.tolist())
    
    if is_alice:
        return alice_links.difference(other_links)
    
    if not is_alice:
        return other_links.difference(alice_links)

alice_sites = get_unique_sites(40, 200, is_alice=True)
other_sites = get_unique_sites(1000, 20, is_alice=False)

# create list of Alice interests based on EDA
alice_interests = {
    'movie', 
    'stream', 
    'tv',
    'youtube',
    'yt3',
    'video',
    'ytimg',
    'cinema',
    'film',
    'youwatch',
    'live',
    'media'
}

sites_from_interests = {site for site in alice_sites for ai in alice_interests if ai in site}
alice_interested = alice_interests.union(alice_sites.difference(sites_from_interests))

# add to dataframe 10 features based on Alice interests
# add to dataframe 10 features based on sites that Alice did not visit
def check_alice_interests(link):
    for site in alice_interested:
        if isinstance(link, str):
            if site in link:
                return 1
    return 0


def check_other_interests(link):
    for site in other_sites:
        if isinstance(link, str):
            if site in link:
                return 1
    return 0


alice_interests_labels = get_labels('alice_interest_')
other_interests_labels = get_labels('other_interest_')

for link, check in zip(links, alice_interests_labels):
    full_df[check] = full_df[link].apply(check_alice_interests)
    
for link, check in zip(links, other_interests_labels):
    full_df[check] = full_df[link].apply(check_other_interests)

# make a schedule when Alice was usually active
def is_alice_active(code):
    weekday = code // 100000
    hour_minutes = code % 100000
     
    if weekday == 3:
        if hour_minutes in range(1550, 1820+1):
            return 1
        
    else:
        if hour_minutes in range(1200, 1350+1):
            return 1
        if hour_minutes in range(1550, 1820+1):
            return 1

    return 0


def get_nunique_sites(row):
    return len(set([d for d in row.values if d > 0]))


def get_evening(hour):
    if hour >= 19 and hour <= 23:
        return 1
    return 0


# get months and days when Alice was active
alice_months = full_df[full_df.target == 1].start_month.unique().tolist()
alice_days = full_df[full_df.target == 1].date.unique().tolist()

# create dataframe with useful features based on CV
fdf = pd.DataFrame(index=full_df.index)
fdf['alice_interested'] = full_df[alice_interests_labels].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
fdf['alice_not_interested'] = full_df[other_interests_labels].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
fdf['nunique_sites'] = full_df[sites].apply(get_nunique_sites, axis=1)
fdf['weekday'] = full_df.time1.dt.weekday + 1
fdf['is_evening'] = full_df['time1'].dt.hour.apply(get_evening)
fdf['start_month'] = full_df['start_month']
fdf['is_alice_months'] = full_df['start_month'].apply(lambda sm: 1 if sm in alice_months else 0)
fdf['is_alice_days'] = full_df['date'].apply(lambda d: 1 if d in alice_days else 0)
fdf['log_duration'] = np.log1p((full_df[times].max(axis=1) - full_df[times].min(axis=1)).dt.seconds)
fdf['weekday_hour_minutes'] = fdf.weekday * 100000 + full_df.time1.dt.hour * 100 + full_df.time1.dt.minute
fdf['is_alice_active'] = fdf['weekday_hour_minutes'].apply(is_alice_active)
weekday_df = pd.get_dummies(fdf['weekday'], prefix='weekday')

# define TF-IDF vectorizers
vect_sites = TfidfVectorizer(
    ngram_range=(1, 10), 
    max_features=1000000,
    sublinear_tf=True
)

vect_links = TfidfVectorizer(
    ngram_range=(1, 6),
    max_features=300000, 
    sublinear_tf=True
)

# create matrices with TF-IDF vectorizers
with open(os.path.join(PATH_TO_DATA, 'train_df_sites.txt')) as f:
    X_vect_sites = vect_sites.fit_transform(f)

with open(os.path.join(PATH_TO_DATA, 'train_df_links.txt')) as f:
    X_vect_links = vect_links.fit_transform(f)

with open(os.path.join(PATH_TO_DATA, 'test_df_sites.txt')) as f:
    X_vect_sites_test = vect_sites.transform(f)

with open(os.path.join(PATH_TO_DATA, 'test_df_links.txt')) as f:
    X_vect_links_test = vect_links.transform(f)

# scaling selected features
columns = [
    'start_month',
    'weekday',
    'is_alice_months',
    'is_alice_days',
    'is_evening',
    'log_duration',
    'alice_interested',
    'is_alice_active',
    'nunique_sites',
    'alice_not_interested'
]

scaled = pp.StandardScaler().fit_transform(fdf[columns])

# combine all stuff in two train/test matrices 
X = csr_matrix(hstack([
    X_vect_sites,
    X_vect_links,
    scaled[:idx_split],
    weekday_df[['weekday_7']][:idx_split],
]))
print(X.shape)
X_test = csr_matrix(hstack([
    X_vect_sites_test,
    X_vect_links_test,
    scaled[idx_split:],
    weekday_df[['weekday_7']][idx_split:],
]))

# train model and get predictions
logit = LogisticRegression(
    C=1.9, 
    random_state=17, 
    solver='liblinear', 
    class_weight={0:107, 1:1},
)
logit.fit(X, train_df['target'])
y_test = logit.predict_proba(X_test)[:, 1]

write_to_submission_file(y_test)
