import unittest
from os import makedirs
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModel




from src.inference.local import models







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






class TokenizerTest(unittest.TestCase):

    def test1(self):
        to_test = models.Tokenizer

        #args
        test_modelname = "Qwen/Qwen3-0.6B"
        test_text = "Tell a joke!"

        #test
        res = to_test(test_modelname)(test_text)
        self.assertIn("input_ids", res)


    def test2(self):
        to_test = models.Tokenizer

        #args
        test_modelname = "Qwen/Qwen3-0.6B"
        test_text = ["Tell a joke!", "I hope it is really funny"]

        #test
        res = to_test(test_modelname)(test_text, padding=True)
        self.assertIn("input_ids", res)
        self.assertEqual(len(res["input_ids"]), 2)




class InferenceTest(unittest.TestCase):

    def test_inference1(self):
        test_modelname = "TinyLlama/TinyLlama_v1.1"
        test_text = "Tell a joke!"

        tk = models.Tokenizer(test_modelname)
        lm = models.LanguageModel(test_modelname)
  
        output = tk(test_text)
        output = lm(**output, max_new_tokens=100)
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
        expected_dim = 768

        model_name = "facebook/dpr-question_encoder-single-nq-base"
        test_text = "Tell a joke!"
        args = {
            "self": models.EmbeddingModel(model_name, force_model=AutoModel)
        } | models.Tokenizer(model_name)(test_text)
        
        

        self.assertEqual(len(to_test(**args)[0]), expected_dim)
        


    def test_inference5(self):
        to_test = models.EmbeddingModel.__call__
        expected_dim = 2560

        model_name = "Qwen/Qwen3-Embedding-4B"
        test_text = "Tell a joke!"
        args = {
            "self": models.EmbeddingModel(model_name, force_model=AutoModel)
        } | models.Tokenizer(model_name)(test_text)
        
        
        
        self.assertEqual(len(to_test(**args)[0]), expected_dim)
        


    def test_inference6(self):
        to_test = models.LanguageModel.__call__
        
        model_name = "Qwen/Qwen3-Reranker-0.6B"



        #test
        tk = models.Tokenizer(model_name)
        args = {
            "self": models.LanguageModel(model_name), 
            "max_new_tokens": 1,
            "output_scores": True,
            "do_sample": False,
            "return_dict_in_generate": True
        } | tk("What is love?")

        self.assertEqual(to_test(**args).sequences.shape, (1,1))



    def test_inference7(self):
        exp = [[151644,   8948,    198,  60256,   3425,    279,  11789,  20027,    279,
           8502,   3118,    389,    279,  11361,    323,    279,    758,   1235,
           3897,     13,   7036,    429,    279,   4226,    646,   1172,    387,
            330,   9693,      1,    476,    330,   2152,   3263, 151645,    198,
         151644,    872,    198,     27,    641,   1235,  26818,  16246,    264,
           3482,   2711,   3239,     11,  17179,   9760,  46769,    429,   4226,
            279,   3239,    198,     27,   2859,  26818,   3555,    374,    279,
           6722,    315,   5616,   5267,  75692,  26818,    576,   6722,    315,
           5616,    374,  26549,     13, 151645,    198, 151644,  77091,    198,
         151667,    271, 151668,    271]]

        model_name = "Qwen/Qwen3-Reranker-0.6B"
        test_queries = [
            "What is the capital of China?",
        ]
        test_documents = [
            "The capital of China is Beijing.",
        ]


        #test
        def frmt(a, b, instruction=None):
            if not instruction:
                instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
            return f"<Instruct>: {instruction}\n<Query>: {a}\n<Document>: {b}".format(instruction=instruction,a=a, b=b)

        tk = models.Tokenizer(model_name)
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix = tk(prefix, add_special_tokens=False)["input_ids"][0]
        suffix = tk(suffix, add_special_tokens=False)["input_ids"][0]
        tk = models.Tokenizer(model_name, prefix=prefix, suffix=suffix)
        res = tk(frmt(test_queries[0], test_documents[0]))["input_ids"].tolist()

        self.assertEqual(res, exp)











if __name__ == "__main__":
    unittest.main()