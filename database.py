import sqlite3
from io import BytesIO
import base64

def create_connection(db_file):
    """ Create a database connection to the SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    """ Create the results tables if they do not exist """
    create_glaucoma_table_sql = """ CREATE TABLE IF NOT EXISTS results (
                                    id integer PRIMARY KEY,
                                    image blob,
                                    cup_area real,
                                    disk_area real,
                                    rim_area real,
                                    rim_to_disc_line_ratio real,
                                    ddls_stage integer
                                ); """
    create_cataract_table_sql = """ CREATE TABLE IF NOT EXISTS cataract_results (
                                    id integer PRIMARY KEY,
                                    image blob,
                                    red_quantity real,
                                    green_quantity real,
                                    blue_quantity real,
                                    stage text
                                ); """
    try:
        cursor = conn.cursor()
        cursor.execute(create_glaucoma_table_sql)
        cursor.execute(create_cataract_table_sql)
    except sqlite3.Error as e:
        print(e)

def save_prediction_to_db(image, cup_area, disk_area, rim_area, rim_to_disc_line_ratio, ddls_stage):
    database = "glaucoma_results.db"
    conn = create_connection(database)
    if conn:
        create_table(conn)
        sql = ''' INSERT INTO results(image, cup_area, disk_area, rim_area, rim_to_disc_line_ratio, ddls_stage)
                  VALUES(?,?,?,?,?,?) '''
        cur = conn.cursor()
        
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        cur.execute(sql, (img_bytes, cup_area, disk_area, rim_area, rim_to_disc_line_ratio, ddls_stage))
        conn.commit()
        conn.close()

def fetch_all_data(conn, table_name):
    """ Fetch all data from the given table """
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    return rows

def get_db_data(db_path_glaucoma, db_path_cataract):
    conn_glaucoma = create_connection(db_path_glaucoma)
    conn_cataract = create_connection(db_path_cataract)
    
    if conn_glaucoma and conn_cataract:
        create_table(conn_glaucoma)
        create_table(conn_cataract)
        glaucoma_data = fetch_all_data(conn_glaucoma, "results")
        cataract_data = fetch_all_data(conn_cataract, "cataract_results")
        conn_glaucoma.close()
        conn_cataract.close()
        return glaucoma_data, cataract_data
    else:
        return [], []

def clear_database(db_path, table_name):
    """Clear all records from the specified table in the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        conn.commit()
        conn.close()
        return f"All records from {table_name} table in {db_path} have been deleted."
    except Exception as e:
        return f"Error deleting records from {table_name} table in {db_path}: {str(e)}"

def format_db_data(glaucoma_data, cataract_data):
    """ Format the database data for display """
    formatted_data = ""

    if not glaucoma_data and not cataract_data:
        return "No data available in the database."

    if glaucoma_data:
        headers = ["ID", "Image", "Cup Area", "Disk Area", "Rim Area", "Rim to Disc Line Ratio", "DDLS Stage"]
        formatted_data += "<h2>Glaucoma Data</h2><table border='1'><tr>" + "".join([f"<th>{header}</th>" for header in headers]) + "</tr>"

        for row in glaucoma_data:
            image_html = "No image"
            if row[1] is not None:
                image = base64.b64encode(row[1]).decode('utf-8')
                image_html = f"<img src='data:image/png;base64,{image}' width='100'/>"
            
            formatted_data += "<tr>" + "".join([f"<td>{image_html if i == 1 else row[i]}</td>" for i in range(len(row))]) + "</tr>"

        formatted_data += "</table>"

    if cataract_data:
        headers = ["ID", "Image", "Red Quantity", "Green Quantity", "Blue Quantity", "Stage"]
        formatted_data += "<h2>Cataract Data</h2><table border='1'><tr>" + "".join([f"<th>{header}</th>" for header in headers]) + "</tr>"

        for row in cataract_data:
            image_html = "No image"
            if row[1] is not None:
                image = base64.b64encode(row[1]).decode('utf-8')
                image_html = f"<img src='data:image/png;base64,{image}' width='100'/>"
            
            formatted_data += "<tr>" + "".join([f"<td>{image_html if i == 1 else row[i]}</td>" for i in range(len(row))]) + "</tr>"

        formatted_data += "</table>"

    return formatted_data