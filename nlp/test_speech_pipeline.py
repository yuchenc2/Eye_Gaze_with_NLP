from asr import *
from recorder import *

def match_segments_to_speech(class_names, word_offsets):
    class_name_speech_info = {}
    for name in class_names:
        if name in word_offsets and name not in class_name_speech_info:
            class_name_speech_info[name] = word_offsets[name]
    return class_name_speech_info

recording_filename = "output.wav"
# Record until silence is detected for 2 seconds
rec = Recorder(silence_threshold = 150, silence_timeout = 2)
rec_frames = rec.record_until_silence()
rec.stop_recording()
rec.save_recording(rec_frames, recording_filename)

# Transcribe the recording
asr = ASR()
# Outputs in the form [word for word in recording], {word: {index: {start_time: float, end_time: float}} for word in recording}
transcription, word_offsets = asr.asr_transcript(recording_filename)

# Print results
print("\nRESULTS: ")
print("-----------------------------------------------------------------------")
print(transcription)
print(word_offsets)
