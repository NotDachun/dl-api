from IPython.utils import io
import numpy as np

def synthesize_audio(text, synthesizer, embed, vocoder, show_display=True):
    print("Synthesizing new audio...")
    with io.capture_output() as captured:
        specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    return generated_wav, synthesizer.sample_rate

def generate_audio(text, synthesizer, embed, vocoder):
    all_audio = []
    for sentence in text:
        generated_wav, sample_rate = synthesize_audio(sentence, synthesizer, embed, vocoder, show_display=False)
        all_audio.append(generated_wav)
    all_audio = np.concatenate(all_audio)
    return all_audio