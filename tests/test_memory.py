import unittest





from src.inference.local import models, memory








class MemoryTest(unittest.TestCase):
    
    def test_memory1(self):
        test_modelname = "TinyLlama/TinyLlama_v1.1"
        test_input = "What a great thing to do!"

        
        tk = models.Tokenizer(test_modelname)
        tk(test_input)
        
        memory.on_demand(tk)
        tk(test_input)







if __name__ == "__main__":
    unittest.main()
