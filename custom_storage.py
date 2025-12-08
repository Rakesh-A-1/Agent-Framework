from crewai.memory.storage.interface import Storage
import json
import os

class FileStorage(Storage):
    def __init__(self, filename="agent_memory.json"):
        self.filename = filename

        # Create file if not exists
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                json.dump([], f)

        # Load data into memory
        with open(self.filename, "r") as f:
            try:
                self.memories = json.load(f)
            except:
                self.memories = []

    def save(self, value, metadata=None, agent=None):
        entry = {
            "value": value,
            "metadata": metadata,
            "agent": agent
        }

        # Append to in-memory store
        self.memories.append(entry)

        # Write back to json file
        with open(self.filename, "w") as f:
            json.dump(self.memories, f, indent=4)

        return entry

    def search(self, query, limit=10, score_threshold=0.5):
        # Optional: simple keyword search
        results = [m for m in self.memories if query.lower() in str(m["value"]).lower()]
        return results[:limit]