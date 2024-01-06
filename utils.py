import pyannote


def get_words_timestamps(result_transcription: dict) -> dict:
    """Get all words their start and end times into a dict"""
    words = {}
    word_counter = 0
    for segment in result_transcription["segments"]:
        for word in segment["words"]:
            words[f"word_{word_counter}"] = {
                "text": word["word"],
                "start": word["start"],
                "end": word["end"],
            }
            word_counter += 1
    return words


def words_per_segment(
    res_transcription: dict, res_diarization: pyannote.core.Annotation
) -> dict:
    """Get all words per segment and their start and end times into a dict

    Args:
        res_transcription (dict): The transcription result from the whisper library
        res_diarization (pyannote.core.Annotation): The diarization result from the pyannote library

    Returns:
        dict: A dict containing all words per segment and their start and end times and the speaker
    """
    res_trans_dia = {}

    for idx, _ in enumerate(res_diarization.itersegments()):
        total_segments = idx

    words = get_words_timestamps(res_transcription)

    for idx, (segment, _, speaker) in enumerate(
        res_diarization.itertracks(yield_label=True)
    ):
        segment_words = []
        for _, word in words.items():
            # For the first segment, include words from the beginning of the audio, to handle some missmatches or errors
            if idx == 0 and word["end"] <= segment.end:
                segment_words.append(word["text"])
            # For the last segment, include all words after the segment start
            elif idx == total_segments and word["start"] >= segment.start:
                segment_words.append(word["text"])
            # For all other segments, include words within the segment
            elif word["start"] > segment.start and word["end"] < segment.end:
                segment_words.append(word["text"])
            # Break the loop if the word starts after the segment end to handle some missmatches or errors
            if word["start"] >= segment.end:
                break

        res_trans_dia[f"segment_{idx}"] = {
            "speaker": speaker,
            "text": " ".join(segment_words),
            "start": segment.start,
            "end": segment.end,
        }
    return res_trans_dia
