import unittest



from src.inference import models







# class LoadingTest(unittest.TestCase):

#     def test_load1(self):
#         test_modelname = "TinyLlama/TinyLlama_v1.1"
        
#         models.Tokenizer(test_modelname).load(test_modelname)
#         models.Model(test_modelname).load(test_modelname)


#     def test_load2(self):
#         test_modelname = "Qwen/Qwen3-0.6B"
        
#         models.Tokenizer(test_modelname).load(test_modelname)
#         models.Model(test_modelname).load(test_modelname)


#     def test_load3(self):
#         test_modelname = "Qwen/Qwen3-8B"

#         models.Tokenizer(test_modelname).load(test_modelname)
#         models.Model(test_modelname).load(test_modelname)


#     def test_load3(self):
#         test_modelname = "Qwen/Qwen3-Embedding-4B"

#         models.Tokenizer(test_modelname).load(test_modelname)
#         models.Model(test_modelname).load(test_modelname)







class InferenceTest(unittest.TestCase):

    def test_inference1(self):
        test_modelname = "TinyLlama/TinyLlama_v1.1"
        test_text = "Tell a joke!"

        tk = models.Tokenizer(test_modelname)
        lm = models.LanguageModel(test_modelname)
  
        output = tk(test_text)
        output = lm(output)
        output = tk.decode(output[0])

        self.assertEqual(type(output), str)


    def test_inference2(self):
        test_modelname = "Qwen/Qwen3-0.6B"
        test_text = "Tell a joke!"

        tk = models.Tokenizer(test_modelname)
        lm = models.LanguageModel(test_modelname)
  
        output = tk(test_text)
        output = lm(output)
        output = tk.decode(output[0])

        self.assertEqual(type(output), str)



    def test_inference3(self):
        test_modelname = "Qwen/Qwen3-8B"
        test_text = "Tell a joke!"

        tk = models.Tokenizer(test_modelname)
        lm = models.LanguageModel(test_modelname)
  
        output = tk(test_text)
        output = lm(output)
        output = tk.decode(output[0])

        self.assertEqual(type(output), str)



    def test_inference4(self):
        test_modelname = "Qwen/Qwen3-Embedding-4B"
        test_text = "Tell a joke!"

        tk = models.Tokenizer(test_modelname)
        m = models.Model(test_modelname)

        w = tk(test_text)
        w = m(w)















if __name__ == "__main__":
    unittest.main()