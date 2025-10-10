import unittest




from src.inference_providers import ollama





class OllamaTest(unittest.TestCase):

    def test_inference(self):
        ollama_url = "http://localhost:11434"
        ollama_model = "tinyllama"
        prompt = "Tell a joke!"


        cl = ollama.OllamaClient(ollama_url, ollama_model)
        self.assertEqual(type(cl.generate(prompt)), str)