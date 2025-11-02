import unittest




from src.inference.providers import ollama, deepinfra





class OllamaTest(unittest.TestCase):

    def test_inference(self):
        ollama_url = "http://localhost:11434"
        ollama_model = "tinyllama"
        prompt = "Tell a joke!"


        cl = ollama.OllamaClient(ollama_url, ollama_model)
        self.assertEqual(type(cl.generate(prompt)), str)





class DeepinfraTest(unittest.TestCase):

    def test_inference(self):
        to_test = deepinfra.DeepInfraClient.generate

        args = {
            "self": deepinfra.DeepInfraClient(model="meta-llama/Meta-Llama-3-8B-Instruct"),
            "prompt": "Antworte auf Deutsch: Was ist ein AVL-Baum in 1-2 Sätzen?"
        }
    

        #test
        result = to_test(**args)

        self.assertEqual(type(result), str)
        self.assertGreater(len(str.split(result)), 5)


    