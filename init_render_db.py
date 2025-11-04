import psycopg2

# Connect to your Render PostgreSQL
conn = psycopg2.connect(
    host="dpg-d454dsh5pdvs73c5cm40-a.oregon-postgres.render.com",
    database="tb_detector_db",
    user="tb_detector_db_user",
    password="Csl2Tebfz2IAaYJAOWMA6jBRIT6KqjKz"
)

cur = conn.cursor()

# Create tables
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

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

# Insert default admin user (username: admin / password: admin123)
cur.execute("""
INSERT INTO users (username, email, password_hash)
VALUES ('admin', 'admin@example.com',
'$2b$12$8zsmPgbx4qO6t8gO5bMZfubq7DZoK1iS8Wm1CgxY9LUaB6LJvRqxy')
ON CONFLICT (username) DO NOTHING;
""")

conn.commit()
cur.close()
conn.close()
print("✅ Tables created and admin user added successfully!")
