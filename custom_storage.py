from crewai.memory.storage.interface import Storage

class CustomStorage(Storage):
    def __init__(self):
        self.memories = []

    def save(self, value, metadata=None, agent=None):
        self.memories.append({
            "value": value,
            "metadata": metadata,
            "agent": agent
        })

    def search(self, query, limit=10, score_threshold=0.5):
        # Implement your search logic here
        return [m for m in self.memories if query.lower() in str(m["value"]).lower()]

    def reset(self):
        self.memories = []