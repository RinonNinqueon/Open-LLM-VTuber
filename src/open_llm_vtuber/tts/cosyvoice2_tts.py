from gradio_client import Client, handle_file
from loguru import logger
from .tts_interface import TTSInterface
import requests
import httpx

class TTSEngine(TTSInterface):
    def __init__(
        self,
        client_url="http://127.0.0.1:50000/",
        mode_checkbox_group="预训练音色",
        sft_dropdown="中文女",
        prompt_text="",
        prompt_wav_upload_url="https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
        prompt_wav_record_url="https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
        instruct_text="",
        stream=False,
        seed=0,
        speed=1.0,
        api_name="/generate_audio",
    ):
#        self.client = Client(client_url)
        self.client_url = "{}inference_zero_shot".format(client_url)
        self.mode_checkbox_group = mode_checkbox_group
        self.sft_dropdown = sft_dropdown
        self.prompt_text = prompt_text
        self.prompt_wav_upload = prompt_wav_upload_url#handle_file(prompt_wav_upload_url)
        self.prompt_wav_record = prompt_wav_upload_url#handle_file(prompt_wav_record_url)
        self.instruct_text = instruct_text
        self.stream = stream
        self.seed = seed
        self.speed = speed
        self.api_name = api_name

    def generate_audio(self, text, file_name_no_ext=None):
        if file_name_no_ext is not None:
            logger.warning(
                "Warning: customizing the temp file name with file_name_no_ext is not supported by cosyvoice2TTS and will be ignored."
            )
        result_wav_path = self.client.predict(
            tts_text=text,
            mode_checkbox_group=self.mode_checkbox_group,
            sft_dropdown=self.sft_dropdown,
            prompt_text=self.prompt_text,
            prompt_wav_upload=self.prompt_wav_upload,
            prompt_wav_record=self.prompt_wav_record,
            instruct_text=self.instruct_text,
            stream=self.stream,
            seed=self.seed,
            speed=self.speed,
            api_name=self.api_name,
        )

        return result_wav_path

    def generate_audio_post(self, text, file_name_no_ext=None) -> requests.Response:
        if file_name_no_ext is not None:
            logger.warning(
                "Warning: customizing the temp file name with file_name_no_ext is not supported by cosyvoice2TTS and will be ignored."
            )
        logger.info(
                "TTS> " + text
            )
        
        payload = {
            'tts_text': text,
            'instruct_text': self.instruct_text,
            'prompt_text': self.prompt_text,
            'prompt_wav': self.prompt_wav_upload,
            'seed': self.seed
        }
        response = requests.post(self.client_url, data=payload, stream=True)
        
        return response
    
    async def async_generate_audio_post(self, text: str, file_name_no_ext=None) -> bytes:
        if file_name_no_ext is not None:
            logger.warning(
                "Warning: customizing the temp file name with file_name_no_ext is not supported by cosyvoice2TTS and will be ignored."
            )
        logger.info(
                "TTS> " + text
            )
        
        payload = {
            'tts_text': text,
            'instruct_text': '',#self.instruct_text,
            'prompt_text': self.prompt_text,
            'prompt_wav': self.prompt_wav_upload,
            'seed': self.seed
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.client_url, data=payload, timeout=None)
            response.raise_for_status()
        
        return response.content
