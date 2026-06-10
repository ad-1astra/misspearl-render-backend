import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "miss_pearl.db"

# You can scale this list to millions of entries
KNOWLEDGE_DATA = [
    ("What is Pearl AI Labs?", "Pearl AI Labs is the leading AI research and development hub in Kampala, Uganda, dedicated to African innovation.", "company"),
    ("Who are the founders?", "The brilliant Paul and Mr. Divid are the visionary founders of Pearl AI Labs.", "founders"),
    ("How does Miss Pearl work?", "I am powered by a sophisticated integration of Gemini 1.5 Pro and ElevenLabs, built by the Pearl Labs engineering team.", "technical"),
    ("Where is Kampala?", "Kampala is the vibrant capital city of Uganda, the Pearl of Africa!", "geography"),
    ("What is Luwombo?", "Luwombo is a traditional Ugandan stew steamed in banana leaves—it's absolutely delicious, dear!", "culture")
]

def populate_kb():
    """Populate the knowledge base with factual Q&A pairs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Ensure table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            category TEXT
        )
    ''')
    
    print("Populating Miss Pearl's knowledge base...")
    cursor.executemany(
        "INSERT INTO knowledge_base (question, answer, category) VALUES (?, ?, ?)", 
        KNOWLEDGE_DATA
    )
    
    conn.commit()
    count = cursor.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
    conn.close()
    print(f"Success! Miss Pearl now has {count} factual records to draw from.")

if __name__ == "__main__":
    populate_kb()