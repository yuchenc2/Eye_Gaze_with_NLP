from asr import *
from recorder import *

recording_filename = "output.wav"
# Record until silence is detected for 2 seconds
rec = Recorder(silence_threshold = 150, silence_timeout = 2)
rec_frames = rec.record_until_silence()
rec.stop_recording()
rec.save_recording(rec_frames, recording_filename)

# Transcribe the recording
asr = ASR()
# Outputs in the form [{'word': string, 'start_time': float, 'end_time': float}]
transcription = asr.asr_transcript(recording_filename)

# Print results
print("\nRESULTS: ")
print("-----------------------------------------------------------------------")
print(transcription)
