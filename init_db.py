import psycopg2

# Connect to your Render PostgreSQL
conn = psycopg2.connect(
    host="dpg-d454dsh5pdvs73c5cm40-a.oregon-postgres.render.com",
    database="tb_detector_db",
    user="tb_detector_db_user",
    password="Csl2Tebfz2IAaYJAOWMA6jBRIT6KqjKz"
)

cur = conn.cursor()

# Create 'users' table
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# Create 'tb_history' table
cur.execute("""
CREATE TABLE IF NOT EXISTS tb_history (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    filename VARCHAR(100),
    tb_probability FLOAT,
    normal_probability FLOAT,
    result_text TEXT,
    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cur.close()
conn.close()
print("✅ Database initialized successfully.")
