import tensorflow
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from datetime import datetime
import numpy as np
import reader
import writer
from datetime import datetime, date

tf.compat.v1.disable_eager_execution()

columns = ['publ_date', 'title', 'description']
rss = reader.read_rss(*columns)

rss.dropna(subset=['description', 'publ_date'], inplace=True)
values = {'title': 'no title'}
rss.fillna(value=values, inplace=True)

rss = reader.get_relevant_rss(rss, reader.read_usd())
_left = datetime.strptime('01.12.2019', '%d.%M.%Y').date()
_right = datetime.strptime('07.12.2020', '%d.%M.%Y').date()
rss['date'] = rss.publ_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())
check_date = rss.date.apply(lambda x: _left <= x <= _right)
rss = rss[check_date == True]
rss = rss.sample(frac=1)
print(rss.shape)
# rss = rss.iloc[:10000]
rss = rss.iloc[:20]
rows = rss.shape[0]
rss.reset_index(drop=True, inplace=True)


elmo = hub.Module('elmo/elmo_ru', trainable=False)

batch_size = 5

start_embed = datetime.now()

for i in range(0, rows, batch_size):
    start_time = datetime.now()

    session = tf.Session()
    embeddings = elmo(rss['description'][i:i+batch_size], signature='default', as_dict=True)['default']
    session.run(tf.global_variables_initializer())
    embs = session.run(embeddings)
    session.close()

    time_elmo = datetime.now() - start_time

    sql_time = datetime.now()
    for j in range(0, batch_size):
        date = rss['publ_date'][i + j]
        title = rss['title'][i + j]
        full_emb = embs[j]
        writer.write_embeddings(date, title, full_emb)
    time_sql = datetime.now() - sql_time

    print(i, f'elmo - {time_elmo.total_seconds()}, sql - {time_sql.total_seconds()}')

end_embed = datetime.now()
time_embed = end_embed - start_embed
print(f'embeddings - {time_embed.total_seconds() / 3600}')