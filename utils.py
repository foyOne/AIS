import reader

def get_train():
   db = 'output/embeddings_dense.sqlite3'
   embed = reader.read_embeddings(db)
   usd = reader.read_usd()
   train = reader.get_train_data(embed, usd)
   return train

def get_large_train():
    embed = reader.read_mean_embeddings()
    usd = reader.read_usd()
    train = reader.get_train_data_by_mean_embedding(embed, usd)
    return train