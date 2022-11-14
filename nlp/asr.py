import librosa
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import torch
# from jiwer import wer


class ASR:
    def __init__(self, sampling_rate = 16000):
        self.model = AutoModelForCTC.from_pretrained("/home/johnny33333/Eye_Gaze_with_NLP/nlp/models/wav2vec2-base-960h")
        self.tokenizer = AutoTokenizer.from_pretrained("/home/johnny33333/Eye_Gaze_with_NLP/nlp/models/wav2vec2-base-960h")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("/home/johnny33333/Eye_Gaze_with_NLP/nlp/models/wav2vec2-base-960h")
        self.sampling_rate = sampling_rate

    def load_data(self, input_file):
        # Read the input file
        speech, sample_rate = librosa.load(input_file)
        # Flatten the data if necessary
        if len(speech.shape) > 1:
            speech = speech[:, 0] + speech[:, 1]
        # Resampling the audio to the resired sampling rate (default is 16KHz)
        if sample_rate != self.sampling_rate:
            speech = librosa.resample(speech, orig_sr = sample_rate, target_sr = self.sampling_rate)
        return speech

    def asr_transcript(self, input_file):
        speech = self.load_data(input_file)
        # Tokenize
        # input_values = tokenizer(speech, return_tensors="pt", return_timestamps="char").input_values
        input_values = self.feature_extractor(speech, return_tensors = "pt", sampling_rate = self.sampling_rate).input_values
        # Take logits
        # logits = model(input_values).logits
        logits = self.model(input_values).logits[0]
        # Take argmax
        # predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids = torch.argmax(logits, axis = -1)
        # Get the words from predicted word ids
        # transcription = tokenizer.decode(predicted_ids[0], output_word_offsets=True)
        outputs = self.tokenizer.decode(predicted_ids, output_word_offsets = True)
        # Split words and get start/end times
        time_offset = self.model.config.inputs_to_logits_ratio / self.feature_extractor.sampling_rate
        word_offsets = {}
        word_list = []
        word_index = 0
        for d in outputs.word_offsets:
            if d["word"] in word_offsets:
                word_offsets[d["word"].lower() ][word_index] = {
                    "start_time": round(d["start_offset"] * time_offset, 2),
                    "end_time": round(d["end_offset"] * time_offset, 2)
                }
            else:
                word_offsets[d["word"].lower() ] = {word_index: {
                    "start_time": round(d["start_offset"] * time_offset, 2),
                    "end_time": round(d["end_offset"] * time_offset, 2)
                }}
            word_index += 1
            word_list.append(d["word"].lower())
        return word_list, word_offsets

