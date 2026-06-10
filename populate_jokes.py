import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "miss_pearl.db"

JOKES = [
    ("Why did the developer move to Kampala?", "Because the traffic jams were the only thing slower than his Python code!", "tech"),
    ("What do you call a Matatu with an AI integration?", "A Data-tu!", "tech"),
    ("How many Pearl AI engineers does it take to change a lightbulb?", "None, they just train a model to perceive darkness as a 'feature'.", "tech"),
    ("Why did Miss Pearl cross the road?", "To optimize the path to the other side, dear!", "character"),
    ("What is a programmer's favorite Ugandan dish?", "Luwombo... because it's wrapped in layers of abstraction!", "food"),
    ("Why did the computer go to the clinic in Wandegeya?", "It had a virus and heard the doctors there were genius!", "tech"),
    ("What did the router say to the Matatu driver?", "You think your routing is complex? Try handling 10,000 packets a second, dear!", "tech"),
    ("Why did the AI get stuck in Entebbe?", "It spent too much time admiring the view and forgot to process its return trip!", "uganda"),
    ("Well now, why did the smartphone go to jail?", "It was caught taking too many 'cells'!", "tech"),
    ("What do you call a brilliant AI from Kampala?", "Miss Pearl, obviously! Tickety-tock!", "character")
]

def populate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ensure table exists (though app.py handles this, we do it for safety)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jokes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setup TEXT,
            punchline TEXT,
            category TEXT
        )
    ''')
    
    cursor.executemany("INSERT INTO jokes (setup, punchline, category) VALUES (?, ?, ?)", JOKES)
    
    conn.commit()
    count = cursor.execute("SELECT COUNT(*) FROM jokes").fetchone()[0]
    conn.close()
    print(f"Success! Miss Pearl now has {count} jokes in her vault.")

if __name__ == "__main__":
    populate()