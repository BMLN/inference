import unittest
from os import makedirs
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModel




from src.inference import models







class ModelClassDetectionTest(unittest.TestCase):

    def test1(self):
        #to_test
        to_test = models.get_model_class_from_cfg
        
        
        #setup
        modelname = "Qwen/Qwen3-Embedding-4B"
        makedirs("./modelcache", exist_ok=True)

        try: 
            __dl = snapshot_download(modelname, local_files_only=True)
        except:
            __dl =  snapshot_download(modelname)


        #test
        config = AutoConfig.from_pretrained(__dl)
        model = to_test(config)

        # print(model)





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

#     # def test_inference1(self):
#     #     test_modelname = "TinyLlama/TinyLlama_v1.1"
#     #     test_text = "Tell a joke!"

#     #     tk = models.Tokenizer(test_modelname)
#     #     lm = models.LanguageModel(test_modelname)
  
#     #     output = tk(test_text)
#     #     output = lm(output)
#     #     output = tk.decode(output[0])

#     #     self.assertEqual(type(output), str)


#     # def test_inference2(self):
#     #     test_modelname = "Qwen/Qwen3-0.6B"
#     #     test_text = "Tell a joke!"

#     #     tk = models.Tokenizer(test_modelname)
#     #     lm = models.LanguageModel(test_modelname)
  
#     #     output = tk(test_text)
#     #     output = lm(output)
#     #     output = tk.decode(output[0])

#     #     self.assertEqual(type(output), str)



#     # def test_inference3(self):
#     #     test_modelname = "Qwen/Qwen3-8B"
#     #     test_text = "Tell a joke!"

#     #     tk = models.Tokenizer(test_modelname)
#     #     lm = models.LanguageModel(test_modelname)
  
#     #     output = tk(test_text)
#     #     output = lm(output)
#     #     output = tk.decode(output[0])

#     #     self.assertEqual(type(output), str)


    def test_inference4(self):
        to_test = models.EmbeddingModel.__call__
        expected_dim = 2560

        model_name = "Qwen/Qwen3-Embedding-4B"
        test_text = "Tell a joke!"
        args = {
            "self": models.EmbeddingModel(model_name, force_model=AutoModel)
        } | models.Tokenizer(model_name)(test_text)
        
        
        
        self.assertEqual(len(to_test(**args)[0]), expected_dim)
        












if __name__ == "__main__":
    unittest.main()