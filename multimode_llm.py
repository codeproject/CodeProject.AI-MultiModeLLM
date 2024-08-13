import os
import time

from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer, AutoConfig
from PIL import Image

class MultiModeLLM:

    def __init__(self, model_id: str, filename:str, model_dir:str, 
                 device:str, verbose: bool = True) -> None:

        self.device     = device
        self.model_path = os.path.join(model_dir, filename)

        try:
          # This will use the model we've already downloaded and cached locally

            self.model      = None
            self.processor  = None
            self.model_path = None
            raise

        except:
            try:
                # This will download the model from the repo and cache it locally
                # Handy if we didn't download during install
                attn_implementation = "eager" if device == "cpu" else "sdpa" # eager = manual

                # use_flash_attention_2 = False if device == "cpu" else True
                use_flash_attention_2 = False # Only on Windows, only on CUDA

                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                # config.gguf_file = ...
                config.attn_implementation   = attn_implementation
                config.device_map            = device
                config.torch_dtype           = 'auto'
                config.trust_remote_code     = True
                config.use_flash_attention_2 = False
                
                self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                                  attn_implementation=attn_implementation,
                                                                  config=config,
                                                                  trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

                self.model_path = self.model.name_or_path

            except Exception as ex:
                self.model      = None
                self.processor  = None
                self.model_path = None

        # get the relative path to the model file from the model itself
        # self.model_path = os.path.relpath(self.llm.model_path)


    def do_chat(self, user_prompt: str, image: Image, system_prompt: str=None,
                max_tokens: int = 1024, temperature: float = 0.4,
                stream: bool = True) -> any:
        """ 
        Generates a response from a chat / conversation prompt
        params:
            prompt:str	                    The prompt to generate text from.
            system_prompt: str=None         The description of the assistant
            max_tokens: int = 128           The maximum number of tokens to generate.
            temperature: float = 0.8        The temperature to use for sampling.
        """

        user_prompt_marker      = '<|user|>\n'
        image_marker            = '<|image_1|>\n'
        assistant_prompt_marker = '<|assistant|>\n'
        prompt_suffix           = "<|end|>\n"

        start_process_time = time.perf_counter()
        stop_reason = None

        if not system_prompt:
            system_prompt = "You're a helpful assistant who answers questions the user asks of you concisely and accurately."

        prompt = f"{user_prompt_marker}{image_marker}\n{user_prompt}{prompt_suffix}{assistant_prompt_marker}"
        # prompt = self.processor.tokenizer.apply_chat_template(messages, 
        #                                                      tokenize=False,
        #                                                      add_generation_prompt=True)

        inferenceMs = 0
        try:
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

            start_inference_time = time.perf_counter()
            generate_ids = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                               eos_token_id=self.processor.tokenizer.eos_token_id,) # note trailing ","
            inferenceMs = int((time.perf_counter() - start_inference_time) * 1000)

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, 
                                                   clean_up_tokenization_spaces=False)[0]
    
        except Exception as ex:
            return {
                "success": False, 
                "error": str(ex),
                "stop_reason": "Exception",
                "processMs": int((time.perf_counter() - start_process_time) * 1000),
                "inferenceMs": 0
            }

        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()

        if stop_reason is None:
            stop_reason = "completed"

        return {
            "success": True, 
            "reply": response,
            "stop_reason": stop_reason,
            "processMs" : int((time.perf_counter() - start_process_time) * 1000),
            "inferenceMs" : inferenceMs
        }

