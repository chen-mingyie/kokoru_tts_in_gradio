import json, os, numpy as np, gradio as gr
from onnxruntime import InferenceSession
from kokoro_onnx.tokenizer import Tokenizer

with open('kokoro_config.json', 'r', encoding="utf8") as file:
  kokoro_config = json.load(file)

class TTS_Model():
    def __init__(self, model_name: str):
        self.tokenizer = Tokenizer()
        self.sess = InferenceSession(os.path.join('onnx', model_name))

    def _tokenize(self, text: str):
        # g2p = en.G2P(trf=False, british=False, fallback=None) # no transformer, American English
        # phonemes, _ = g2p(text)
        # tokens = tokenizer.tokenize(text)
        phonemes = self.tokenizer.phonemize(text)
        tokens = [kokoro_config['vocab'].get(p, 1) for p in phonemes]
        print(text)
        print(tokens)
        print(phonemes)
        print()
        return tokens, phonemes

    def generate_audio(self, text: str, voicefile: str):
        # You can generate token ids as follows:
        #   1. Convert input text to phonemes using https://github.com/hexgrad/misaki
        #   2. Map phonemes to ids using https://huggingface.co/hexgrad/Kokoro-82M/blob/785407d1adfa7ae8fbef8ffd85f34ca127da3039/config.json#L34-L148
        # tokens = [50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]

        tokens, phonemes = self._tokenize(text)

        # Context length is 512, but leave room for the pad token 0 at the start & end
        assert len(tokens) <= 510, len(tokens)

        # Style vector based on len(tokens), ref_s has shape (1, 256)
        voices = np.fromfile(os.path.join('voices', voicefile), dtype=np.float32).reshape(-1, 1, 256)
        ref_s = voices[len(tokens)]

        # Add the pad ids, and reshape tokens, should now have shape (1, <=512)
        tokens = [[0, *tokens, 0]]

        #   model_name = 'model_q8f16.onnx' # Options: model.onnx, model_fp16.onnx, model_quantized.onnx, model_q8f16.onnx, model_uint8.onnx, model_uint8f16.onnx, model_q4.onnx, model_q4f16.onnx
        #   sess = InferenceSession(os.path.join('onnx', model_name))

        audio = self.sess.run(None, dict(
            input_ids=tokens,
            style=ref_s,
            speed=np.ones(1, dtype=np.float32),
        ))[0]

        return audio, phonemes

def create_app():
    tts_model = TTS_Model('model_q8f16.onnx')
    def create_audio(text: str, voicename: str):
        voicefile = kokoro_config['voices'].get(voicename, 'af_heart.bin')
        audio, phonemes = tts_model.generate_audio(text, voicefile)
        audio_samples = audio[0]
        sample_rate_hz = 24000
        # sf.write(f'audio.wav', audio[0], 24000) # save audio file
        return (sample_rate_hz, audio_samples), phonemes

    with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto")])) as ui:
        text_input = gr.TextArea(
            label="Input Text",
            rtl=False,
            value="Kokoro TTS. Turning words into emotion, one voice at a time!",
        )
        voice_input = gr.Dropdown(
            label="Voice", value="American female - Heart", choices=list(kokoro_config['voices'].keys())
        )
        submit_button = gr.Button("Create")
        phonemes_output = gr.Textbox(label="Phonemes")
        audio_output = gr.Audio()
        submit_button.click(
            fn=create_audio,
            inputs=[text_input, voice_input],
            outputs=[audio_output, phonemes_output],
        )
    return ui

ui = create_app()
ui.launch(debug=True, server_name="0.0.0.0") #, sharable=True)
    
