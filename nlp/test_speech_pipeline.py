from asr import ASR
from recorder import Recorder
from threading import Thread
import time

transcription = []
word_offsets = {}

def match_segments_to_speech(class_names, word_offsets):
    class_name_speech_info = []
    for name in class_names:
        class_info = word_offsets[name] if name in word_offsets else {}
        class_name_speech_info.append((name, class_info))
    return class_name_speech_info

def record_and_transcribe(rec, recording_filename = "output.wav"):
    # Record until silence is detected for 2 seconds
    global transcription
    global word_offsets
    rec_frames = rec.record_until_silence()
    rec.stop_recording()
    rec.save_recording(rec_frames, recording_filename)
    # Transcribe the recording
    asr = ASR()
    # Outputs in the form [word for word in recording], {word: {index: {start_time: float, end_time: float}} for word in recording}
    transcription, word_offsets = asr.asr_transcript(recording_filename)
    print("Thread done")
    return transcription, word_offsets

rec = Recorder(silence_threshold = 150, silence_timeout = 2)
rec_thread = Thread(target = record_and_transcribe, args = (rec,), daemon = True)
rec_thread.start()
while not rec.finished_rec:
    time.sleep(1)
print("Main thread: ", time.time())
rec_thread.join()

# Print results
print("\nRESULTS: ")
print("-----------------------------------------------------------------------")
print(transcription)
print(word_offsets)

