import sqlite3
import os


def clear_database():
    """Clear all test data from the database"""
    db_path = "app/data/saskedchat.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get current counts
        cursor.execute("SELECT COUNT(*) FROM subscribers")
        sub_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM posts")
        post_count = cursor.fetchone()[0]

        print(f"Current data counts:")
        print(f"Subscribers: {sub_count}")
        print(f"Posts: {post_count}")

        # Clear tables
        cursor.execute("DELETE FROM subscribers")
        cursor.execute("DELETE FROM posts")
        conn.commit()

        print("\nAll data cleared!")

        # Verify
        cursor.execute("SELECT COUNT(*) FROM subscribers")
        cursor.execute("SELECT COUNT(*) FROM posts")
        print("Database is empty and ready for new test data")

    except Exception as e:
        print(f"Error clearing database: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    clear_database()
