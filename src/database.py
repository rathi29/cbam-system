import sqlite3
from datetime import datetime

def init_db(db_path='data/carbon_ledger.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS shipments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shipment_id TEXT UNIQUE,
        material_type TEXT,
        tonnage REAL,
        country_of_origin TEXT,
        direct_emissions REAL,
        audit_flag TEXT,
        created_at TIMESTAMP,
        source_document TEXT
    )''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def insert_shipment(data, db_path='data/carbon_ledger.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    try:
        c.execute('''INSERT INTO shipments 
                     (shipment_id, material_type, tonnage, country_of_origin, 
                      direct_emissions, audit_flag, created_at, source_document)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (data['shipment_id'], data['material_type'], data['tonnage'],
                   data['country_of_origin'], data['direct_emissions'],
                   data.get('audit_flag', 'PASS'), datetime.now(), 
                   data.get('source_document', 'manual')))
        conn.commit()
        return True
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        return False
    finally:
        conn.close()

def get_all_shipments(db_path='data/carbon_ledger.db'):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM shipments ORDER BY created_at DESC')
    result = [dict(row) for row in c.fetchall()]
    conn.close()
    return result