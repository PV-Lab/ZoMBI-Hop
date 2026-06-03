import os
import sqlite3

# — helper to drop every table in a SQLite file —
def _wipe_db(path):
    conn = sqlite3.connect(path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for (tbl,) in cur.fetchall():
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()

def initialize_db():

    # =============================== #
    # INITIALIZE OBJECTIVE DB FILE
    # =============================== #
    obj_db = "./sql/objective.db"
    os.makedirs(os.path.dirname(obj_db), exist_ok=True)
    _wipe_db(obj_db)

    conn   = sqlite3.connect(obj_db)
    cursor = conn.cursor()
    # recreate the two tables
    cursor.execute("CREATE TABLE objective    (id INTEGER PRIMARY KEY)")
    col_defs = ", ".join(f"col_{i} REAL" for i in range(10))
    cursor.execute(f"CREATE TABLE compositions ({col_defs})")
    # Add handshake table to objective.db
    obj_db_path = './sql/objective.db'
    handshake_conn = sqlite3.connect(obj_db_path)
    handshake_cur = handshake_conn.cursor()
    handshake_cur.execute('''CREATE TABLE IF NOT EXISTS handshake (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        new_objective_available INTEGER DEFAULT 0
    )''')
    handshake_cur.execute('INSERT OR IGNORE INTO handshake (id, new_objective_available) VALUES (1, 0)')
    handshake_conn.commit()
    handshake_conn.close()
    conn.commit()
    conn.close()
    # =============================== #


    # =============================== #
    # INITIALIZE OBJECTIVE MEMORY DB
    # =============================== #
    mem_db = "./sql/objective_memory.db"
    _wipe_db(mem_db)

    conn   = sqlite3.connect(mem_db)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE objective (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    # =============================== #


    # =============================== #
    # INITIALIZE COMPOSITIONS DB FILE
    # =============================== #
    comp_db = "./sql/compositions.db"
    os.makedirs(os.path.dirname(comp_db), exist_ok=True)
    _wipe_db(comp_db)

    conn = sqlite3.connect(comp_db)
    cur  = conn.cursor()
    col_defs = ", ".join(f"col_{i} REAL" for i in range(10))

    # create all six data tables
    for tbl in (
        "compositions", "start", "end",
        "compositions_cache", "start_cache", "end_cache"
    ):
        cur.execute(f"CREATE TABLE {tbl} ({col_defs})")

    # create timestamp table
    cur.execute("CREATE TABLE timestamp (ts REAL)")

    conn.commit()
    conn.close()
    # =============================== #
