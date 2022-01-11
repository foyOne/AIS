import sqlite3

def write_embeddings(date, title, full_emb):
    
    db = "output/embeddings_dense1.sqlite3"
    conn = sqlite3.connect(db)
    sqlite_insert_query = 'insert into embeddings (publ_date, title, full_embed) values (?,?,?)'
    full = sqlite3.Binary(full_emb)

    cursor = conn.cursor()
    cursor.execute(sqlite_insert_query, [date, title, full])
    cursor.close()

    conn.commit()
    conn.close()