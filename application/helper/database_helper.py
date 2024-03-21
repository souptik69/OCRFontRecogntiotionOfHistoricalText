import psycopg2
from datetime import datetime

def connect():
    """ Connect to the PostgreSQL database server """
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host="131.188.35.70",
            # host="localhost",
            database="nbb_rd",
            user="root",
            password="Sl5BouGuIWhwSXig8Xi7",
            port=8432)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def select_line_info(conn, line_entry):
    sql = f"""SELECT is_recipient FROM page_line_predictions
        WHERE book = '{line_entry.book}' AND page = '{line_entry.page}' AND id = '{line_entry.id}'"""
    row = _execute_sql(conn, sql, ret=True)
    return row


def get_recipients_for_page(conn, book, page):
    sql = f"SELECT id FROM page_line_predictions WHERE book='{book}' AND page='{page}' AND is_recipient=True"
    result = _execute_sql(conn, sql, ret=True, all=True)

    page_recipient_ids = []
    for r in result:
        page_recipient_ids.append(r[0])
    return page_recipient_ids


def insert_line_infos(conn, line_entry, content, threshold, semantic_pred, visual_pred):
    combined = semantic_pred * visual_pred
    is_recipient = combined > (threshold * 0.5)

    sql = f"""INSERT INTO page_line_predictions(book, page, id, content, is_recipient, combined_pred, semantic_pred, visual_pred)
          VALUES ('{line_entry.book}', '{line_entry.page}', '{line_entry.id}', '{content}', '{is_recipient}', '{combined}', '{semantic_pred}', '{visual_pred}');"""
    _execute_sql(conn, sql)


def toggle_line_info(conn, line_entry):
    now_string = _get_now_as_string()
    sql = f"""UPDATE page_line_predictions SET is_recipient=NOT is_recipient, changed='true', last_changed_at='{now_string}'
        WHERE  book='{line_entry.book}' AND id='{line_entry.id}' AND page='{line_entry.page}';"""

    _execute_sql(conn, sql)

def _get_now_as_string():
    f = '%Y-%m-%d %H:%M:%S'
    now = datetime.now()
    return now.strftime(f)

def _execute_sql(conn, sql, ret=False, all=False):
    cur = conn.cursor()
    cur.execute(sql)
    output = None
    if ret:
        if all:
            output = cur.fetchall()
        else:
            output = cur.fetchone()
    else:
        try:
            conn.commit()
        except:
            print("error occured. Rolling back")
            conn.rollback()
    cur.close()
    return output
