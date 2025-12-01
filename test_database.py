import sqlite3
import os

def test_database():
    db_path = 'corpus_platform.db'
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='corpus_items'")
            if cursor.fetchone():
                # Get total count
                cursor.execute("SELECT COUNT(*) FROM corpus_items")
                total_count = cursor.fetchone()[0]
                print(f"Database OK: {total_count} total items")

                # Get status breakdown
                cursor.execute("SELECT status, COUNT(*) FROM corpus_items GROUP BY status")
                print("Status breakdown:")
                for status, count in cursor.fetchall():
                    print(f"  {status}: {count}")

                # Check recent items
                cursor.execute("SELECT title, word_count, date_added FROM corpus_items ORDER BY date_added DESC LIMIT 3")
                print("\nRecent items:")
                for title, words, date in cursor.fetchall():
                    print(f"  {title[:50]}... ({words} words, {date})")

            else:
                print("ERROR: corpus_items table not found")

            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    else:
        print("Database file not found")

if __name__ == "__main__":
    test_database()
