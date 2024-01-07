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
    res_transcription: dict,
    res_diarization: pyannote.core.Annotation,
    add_buffer: bool = False,
    fixed_margin: float = 0.5,  # Default fixed buffer value in seconds
    gap_scale_factor: float = 0.3,  # Default scale factor for dynamic buffer
) -> dict:
    """Get all words per segment and their start and end times into a dict

    Args:
        res_transcription (dict): The transcription result from the whisper library
        res_diarization (pyannote.core.Annotation): The diarization result from the pyannote library
        add_buffer (bool): Whether to add buffer time to segment start and end
        fixed_margin (float): The fixed buffer time in seconds
        gap_scale_factor (float): The scale factor for the dynamic buffer

    Returns:
        dict: A dict containing all words per segment and their start and end times and the speaker
    """

    def calculate_dynamic_buffer(idx, segments):
        """Calculate the buffer time based on the previous and current segment"""
        if idx == 0 or idx == len(segments) - 1:
            return fixed_margin
        previous_end = segments[idx - 1].end
        current_start = segments[idx].start
        return (current_start - previous_end) * gap_scale_factor

    res_trans_dia = {}
    segments = list(res_diarization.itersegments())

    words = get_words_timestamps(res_transcription)

    for idx, (segment, _, speaker) in enumerate(
        res_diarization.itertracks(yield_label=True)
    ):
        buffer_time = calculate_dynamic_buffer(idx, segments) if add_buffer else 0

        adjusted_start = max(0, segment.start - buffer_time) if idx != 0 else 0
        adjusted_end = (
            segment.end + buffer_time if idx != len(segments) - 1 else segment.end
        )

        segment_words = []
        for _, word in words.items():
            if word["start"] >= adjusted_start and word["end"] <= adjusted_end:
                segment_words.append(word["text"])
            if word["start"] >= adjusted_end:
                break

        res_trans_dia[f"segment_{idx}"] = {
            "speaker": speaker,
            "text": " ".join(segment_words),
            "start": adjusted_start,
            "end": adjusted_end,
        }
    return res_trans_dia
