#TODO's:
# - add MODEL_SUPPORTfor vision/audio







from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import DPRContextEncoder, DPRQuestionEncoder
from transformers import AutoConfig, AutoTokenizer

import torch

from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from os import makedirs




from typing import override










from logging import getLogger
logger = getLogger()


from transformers import logging
logging.set_verbosity_error()
getLogger("accelerate.utils.modeling").setLevel(40) #logging.ERROR = 40








MODEL_SUPPORT = {
    "CausalLM": AutoModelForCausalLM,
    "Seq2SeqLM": AutoModelForSeq2SeqLM,
    "TokenClassification": AutoModelForTokenClassification,
    "SequenceClassification": AutoModelForSequenceClassification,
    "DPRContextEncoder": DPRContextEncoder,
    "DPRQuestionEncoder": DPRQuestionEncoder
}

def get_model_class_from_cfg(cfg):
    model_name = str((cfg.architectures or [None])[0])
    
    try:
        for suffix, auto_class in MODEL_SUPPORT.items():
            if model_name.endswith(suffix):

                return auto_class
            
        raise ModuleNotFoundError

    except ModuleNotFoundError:
        logger.warning(f"no matching Model for {model_name}, using AutoModel")
        
        return AutoModel  







####
# set cache through:
# HF_HOME=
# HF_DATASETS_DOWNLOADED_DATASETS_PATH=
# HF_DATASETS_CACHE=
####


class Model():

    @classmethod
    def load(cls, modelname, modelcache="/modelcache", force_model=None):
        makedirs(modelcache, exist_ok=True)

        try: 
            __dl = snapshot_download(modelname, local_files_only=True)
        except:
            __dl =  snapshot_download(modelname)

        config = AutoConfig.from_pretrained(__dl)
        model = force_model if force_model else get_model_class_from_cfg(config)

        # tokenizer = AutoTokenizer.from_pretrained(__dl)
        return model.from_pretrained(
            __dl,
            config=config,
            device_map="auto",#"sequential",
            #max_memory={0: "8GiB", "cpu": "16GiB"},
            torch_dtype="auto",
            offload_folder=modelcache
        )
    

    def __init__(self, modelname, force_model=None):
        self.modelname = modelname
        self.force_model=force_model

        self.model = None

    def __del__(self):
        del self.model

        self.model = None

    def __call__(self, *args, **kwargs):
        # if self.tokenizer == None and self.model == None:
            # self.tokenizer, self.model = self.load(self.modelname)
        if self.model == None:
            self.model = self.load(self.modelname, force_model=self.force_model)

        return self.inference(*args, **kwargs)
                
    
    def inference(self, *args, **kwargs):
        #COMPLETELY DEFAULT CALL
        #most should demand override!
        assert self.model

        return self.model(*args, **kwargs)
    



class Tokenizer(Model):

    @override
    @classmethod
    def load(cls, modelname, modelcache="/modelcache", *args, **kwargs):
        makedirs(modelcache, exist_ok=True)

        try: 
            __dl = snapshot_download(modelname, local_files_only=True)
        except:
            __dl =  snapshot_download(modelname)

    
        return AutoTokenizer.from_pretrained(__dl)
        
        
    @override
    def inference(self, text, *args, **kwargs):
        return super().inference(*([text] + list(*args)), **(kwargs | {"return_tensors": "pt"}))


    def decode(self, *args, **kwargs):
        """
            expects a list of token_ids
        """
        if self.model == None:
            self.load(self.modelname)

        return self.model.decode(*args, **({"skip_special_tokens": True} | kwargs))



class LanguageModel(Model):
    
    @override
    def inference(self, kwargs):
        assert self.model
        assert "input_ids" in kwargs
        
        with torch.no_grad():
            return self.model.generate(
                **(
                    { 
                        "do_sample": True, 
                        "return_dict_in_generate": False
                    } | kwargs
                )
            )[:, kwargs["input_ids"].shape[-1]]



class EmbeddingModel(Model):

    #https://huggingface.co/Qwen/Qwen3-Embedding-4B
    @classmethod
    def pool_mean(cls, tensors, attention_masks):

        if attention_masks[:, -1].sum() == attention_masks.shape[0]: #if left_padding
            return tensors[:, -1]
        
        else:
            sequence_lengths = attention_masks.sum(dim=1) - 1
            batch_size = tensors.shape[0]
            return tensors[torch.arange(batch_size, device=tensors.device), sequence_lengths]




    @override
    def inference(self, *args, **kwargs): #better args passing
        return self.pool_mean(super().inference(*args, **kwargs).last_hidden_state, kwargs["attention_mask"])
