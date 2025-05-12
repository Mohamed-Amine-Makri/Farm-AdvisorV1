import os
import sys

# Add the parent directory to sys.path to import modules from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import after modifying the path
from src.database.models import init_db, Base

def setup_database():
    """Initialize the database schema"""
    try:
        print("Setting up database...")
        engine = init_db()
        print("Database setup complete!")
        return True
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        return False

if __name__ == "__main__":
    setup_database()