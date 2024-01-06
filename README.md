# Pyannote - Whisper


## Description
This repo handles diarization and transcription word by word.  
It is an alternative to [pyannote-whisper](https://github.com/yinruiqing/pyannote-whisper)  
The main difference is in the way the words are matched with segments. In this project we look word by word if it belong to the segment or not.  
Important to pass the argument "word_timestamps=True"

## Installation
1. Install [Whisper](https://github.com/openai/whisper)
2. Install Pyannote https://huggingface.co/pyannote/speaker-diarization-3.1

## Usage 

```python
import whisper
from pyannote.audio import Pipeline
from utils import words_per_segment

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token="your/token"
)

model = whisper.load_model("base")
diarization_result = pipeline("path/to/audio.wav")
transcription_result = model.transcribe("path/to/audio.wav")

final_result = words_per_segment(transcription_result, diarization_result)

for _, segment in final_result.items():
    print(f'{segment["start"]:.3f}\t{segment["end"]:.3f}\t {segment["speaker"]}\t{segment["text"]}')

```

```bash
Start   End      Speaker     Text
0.705	1.587	 SPEAKER_01	 Hola,  soy  José.
2.063	4.185	 SPEAKER_00	 José.  Soy  Cris.  ¿Qué  tal  estás?
4.270	5.255	 SPEAKER_01	 Muy  bien.  ¿Y  tú  qué  tal?
5.543	6.732	 SPEAKER_00	 muy  bien  también.
6.698	7.309	 SPEAKER_01	 ¿Qué  hiciste  ayer?
7.937	10.586	 SPEAKER_00	 Estuve  en  un  balneario  con  una  piscina  y  tú.
```