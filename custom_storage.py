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
            "content": value,
            "metadata": metadata,
            "agent": agent
        }
        self.memories.append(entry)
        with open(self.filename, "w") as f:
            json.dump(self.memories, f, indent=4)
        return entry

    def search(self, query, limit=10, score_threshold=0.5):
        results = []
        for m in self.memories:
            value_str = str(m.get("content", "")).lower() 
            if query.lower() in value_str:
                results.append(m)
        
        recent = self.memories[-limit:] if len(self.memories) > limit else self.memories
        combined = {id(m): m for m in results + recent}
        return list(combined.values())[:limit]