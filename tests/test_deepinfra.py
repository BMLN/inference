from src.inference.providers import DeepInfraClient

if __name__ == "__main__":
    client = DeepInfraClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.2,
        max_tokens=100,
    )
    out = client.generate(
        "Antworte auf Deutsch: Was ist ein AVL-Baum in 1–2 Sätzen?"
    )
    print("=== MODEL ANTWORT ===")
    print(out)
