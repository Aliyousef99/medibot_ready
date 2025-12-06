import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env in the current directory
load_dotenv()

# Fetch variables (prefer DB_* to avoid clashing with PORT)
USER = os.getenv("DB_USER") or os.getenv("user")
PASSWORD = os.getenv("DB_PASSWORD") or os.getenv("password")
HOST = os.getenv("DB_HOST") or os.getenv("host")
PORT = os.getenv("DB_PORT") or os.getenv("port")
DBNAME = os.getenv("DB_NAME") or os.getenv("dbname")

def main() -> None:
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME,
        )
        print("Connection successful!")

        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT NOW();")
                result = cursor.fetchone()
                print("Current Time:", result)

        connection.close()
        print("Connection closed.")

    except Exception as e:
        print(f"Failed to connect: {e}")


if __name__ == "__main__":
    main()
