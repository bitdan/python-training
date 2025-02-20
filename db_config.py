import pymysql
from pymysql.cursors import DictCursor

def get_connection():
    """
    获取数据库连接
    """
    try:
        conn = pymysql.connect(
            host='112.74.51.xxx',
            user='root',
            password='xxxx?',
            database='xxx_erp',
            charset='utf8mb4'
        )
        return conn
    except Exception as e:
        print(f"数据库连接失败: {e}")
        raise e

def close_connection(conn):
    """
    关闭数据库连接
    """
    if conn:
        conn.close()

def execute_query(query, params=None):
    """
    执行SQL查询
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()
    finally:
        close_connection(conn)

def execute_update(query, params=None):
    """
    执行更新操作
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            result = cursor.execute(query, params or ())
            conn.commit()
            return result
    finally:
        close_connection(conn) 