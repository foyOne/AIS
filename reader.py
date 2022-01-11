import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime as dt

# datetime - title - embedding - pred
def get_train_data(embedding, usd):
    usd = usd.copy()

    usd.date = usd.date.apply(lambda x: dt.strptime(x, '%d.%m.%Y').date())
    embedding_date = embedding.publ_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S').date())

    pred = embedding_date.apply(lambda x: usd[usd.date == x])
    pred = pred.apply(lambda x: int(x.change.values[0] > 0) if x.size else pd.NA)

    train_data = pd.DataFrame()
    train_data['date'] = embedding.publ_date
    train_data['title'] = embedding.title
    train_data['embedding'] = embedding.full_embed.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    train_data['pred'] = pred

    train_data.dropna(inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    return train_data

def get_train_data_by_mean_embedding(embedding, usd):
    usd = usd.copy()

    usd.date = usd.date.apply(lambda x: dt.strptime(x, '%d.%m.%Y').date())
    embedding_date = embedding.date.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S').date())

    pred = embedding_date.apply(lambda x: usd[usd.date == x])
    pred = pred.apply(lambda x: int(x.change.values[0] > 0) if x.size else pd.NA)

    train_data = pd.DataFrame()
    train_data['date'] = embedding.date.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S'))
    train_data['embedding'] = embedding.mean_embed
    train_data['pred'] = pred

    train_data.dropna(inplace=True)
    train_data.drop_duplicates(subset=['embedding'], inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    train_data.embedding = train_data.embedding.apply(lambda x: np.frombuffer(x, dtype=np.float32))

    return train_data

# Получить все новости, в дни которых есть информация о usd
def get_relevant_rss(rss, usd):
    rss = rss.copy()
    usd = usd.copy()

    usd.date = usd.date.apply(lambda x: dt.strptime(x, '%d.%m.%Y').date())

    rss['date'] = rss.publ_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S').date())
    df = pd.merge(rss, usd, how='inner', on='date')
    df.sort_values(by='date', ascending=False, inplace=True)
    df.drop(columns=['date', 'price', 'open', 'max_price', 'min_price', 'change'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_relevant_data(rss, usd):
    usd_date_field = usd.date.apply(lambda x: dt.strptime(x, '%d.%m.%Y').date())
    usd_date_range = [usd_date_field.min(), usd_date_field.max()]

    rss_date_field = rss.publ_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S').date())
    rss_date_range = [rss_date_field.min(), rss_date_field.max()]

    date_range = [usd_date_range, rss_date_range]
    # print(date_range)
    relevant_range = None

    if rss_date_range[0] < usd_date_range[0]:
        date_range[0], date_range[1] = date_range[1], date_range[0]

    if date_range[0][1] > date_range[1][0]:
        relevant_range = [date_range[1][0], date_range[0][1]]

        if date_range[0][1] > date_range[1][1]:
            relevant_range = date_range[1]
    else:
        raise Exception('dates do not cross')
    # print(date_range)
    # print(relevant_range)
    
    check_rss_date = rss_date_field.apply(lambda x: relevant_range[0] <= x <= relevant_range[1])
    check_usd_date = usd_date_field.apply(lambda x: relevant_range[0] <= x <= relevant_range[1])
    relevant_rss = rss[check_rss_date == True]
    relevant_usd = usd[check_usd_date == True]
    return relevant_rss, relevant_usd

def read_rss(*names, limit=None):
    db = 'data/rss.sqlite3'
    conn = sqlite3.connect(db)
    columns = ', '.join(names)
    limit = '' if limit is None else f'limit {limit}'
    data = pd.read_sql_query(f'select {columns} from rss_news {limit}', conn)
    conn.close()
    return data


def read_mean_embeddings():
    db = 'data/rss_new.sqlite3'
    conn = sqlite3.connect(db)
    embeddings = pd.read_sql_query('select date, mean_embed from embeddings', conn)
    conn.close()
    embeddings.drop_duplicates(inplace=True)
    embeddings.reset_index(drop=True, inplace=True)
    return embeddings

def read_embeddings(db):
    conn = sqlite3.connect(db)
    embeddings = pd.read_sql_query('select publ_date, title, full_embed from embeddings', conn)
    conn.close()
    embeddings.drop_duplicates(inplace=True)
    return embeddings

def read_usd():
    usd_data = pd.read_csv('data/USD_RUB1d.csv', sep=';', names=['date', 'price', 'open', 'max_price', 'min_price', 'change'])
    return usd_data


columns = ['publ_date', 'title', 'description']
rss, usd = read_rss(*columns), read_usd()


# r_rss, r_usd = get_relevant_data(rss, usd)

# a = r_rss.publ_date.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S').date())
# print([a.min(), a.max()])

# a = r_usd.date.apply(lambda x: dt.strptime(x, '%d.%m.%Y').date())
# print([a.min(), a.max()])

# print(rss)
# print(r_rss)

# print(usd)
# print(r_usd)
