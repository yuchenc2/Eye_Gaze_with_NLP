
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC


model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

tokenizer.save_pretrained("./nlp/models/wav2vec2-base-960h")
model.save_pretrained("./nlp/models/wav2vec2-base-960h")
feature_extractor.save_pretrained("./nlp/models/wav2vec2-base-960h")

