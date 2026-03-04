import collections
import numpy as np
import torch
import torchaudio

TOKEN_EOS = {'。', '?', '!'}
TOKEN_COMMA = {'、', ','}
TOKEN_PUNC = TOKEN_EOS | TOKEN_COMMA
PHONEMIC_BREAK = 8000
CHARS_PER_SEGMENT = 15

def ctc_decode(model, samples):
    """Get character probabilities per frame using CTC network"""

    # Prepare audio data for encode()
    speech = torch.tensor(samples).unsqueeze(0)
    length = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

    # Convert to appropriate types
    dtype = getattr(torch, model.dtype)
    speech = speech.to(device=model.device, dtype=dtype)
    length = length.to(device=model.device)

    # Pass audio data through CTC network
    enc = model.asr_model.encode(speech, length)[0]
    lpz = model.asr_model.ctc.softmax(enc)
    return lpz.detach().squeeze(0).cpu().numpy()

def find_blank(model, samples, threshold=0.98):
    """Find no-speech segment in audio stream.

    The entire point of this function is to detect a reasonable
    audio segment for ASR tasks, and to increase the accuracy of
    ASR tasks.

    See also: arXiv:2002.00551
    """
    Blank = collections.namedtuple('Blank', ['start', 'end'])
    blank_id = model.asr_model.blank_id
    nsamples = len(samples)

    # Get character probability matrix using CTC
    lpz = ctc_decode(model, samples)

    # Now find all the consecutive nospeech segment
    blanks = [Blank(nsamples, nsamples)]
    start = None
    for idx, prob in enumerate(lpz.T[blank_id]):
        if prob > threshold:
            if start is None:
                start = int(idx / (lpz.shape[0] + 1) * nsamples)
        else:
            if start and start > 0:
                end = int(idx / (lpz.shape[0] + 1) * nsamples)
                blanks.append(Blank(start, end))
            start = None

    return max(blanks, key=lambda b: b.end - b.start)

def get_timings(model, samples, text):
    """Compute playback timing of each character.

    Uses torchaudio forced_align to map text to audio frames.

    Args:
        model: The ReazonSpeech ESPnet model.
        samples: The audio waveform array.
        text: The transcribed text string.

    Returns:
        np.ndarray: Array of timings in audio samples for each character.
    """
    lpz = ctc_decode(model, samples)

    token_list = model.asr_model.token_list
    blank_id = model.asr_model.blank_id

    # 1. Map text characters to model token IDs using greedy matching.
    # This handles single-character and multi-character (BPE) tokens
    # safely, mimicking original ctc_segmentation.prepare_text behavior.
    tokens = []
    char_to_token_idx =[]
    max_token_len = max(len(t) for t in token_list)

    i = 0
    while i < len(text):
        match_found = False
        for length in range(max_token_len, 0, -1):
            if i + length <= len(text):
                span = text[i:i+length]
                if span in token_list:
                    tokens.append(token_list.index(span))
                    char_to_token_idx.append(i)
                    i += length
                    match_found = True
                    break
        if not match_found:
            i += 1

    if not tokens:
        # Raise an error to trigger split_text's fallback mechanism
        raise ValueError("No valid tokens found in the text.")


    # 2. Prepare tensors for forced_align
    # Use clamp to avoid -inf from log(0)
    log_probs = torch.from_numpy(lpz).clamp(min=1e-7).log().unsqueeze(0)
    targets = torch.tensor([tokens], dtype=torch.long)

    in_lens = torch.tensor([log_probs.shape[1]], dtype=torch.long)
    tgt_lens = torch.tensor([targets.shape[1]], dtype=torch.long)

    # 3. Perform forced alignment natively
    alignments, _ = torchaudio.functional.forced_align(
        log_probs,
        targets,
        input_lengths=in_lens,
        target_lengths=tgt_lens,
        blank=blank_id
    )

    alignments = alignments[0].tolist() # Extract path of length T

    # 4. Reconstruct character timings from the alignment path
    timings = np.zeros(len(text))
    target_idx = 0
    prev_token_id = blank_id

    for frame_idx, token_id in enumerate(alignments):
        if target_idx < len(tokens):
            expected = tokens[target_idx]

            # New emission happens if we see the expected token AND
            # it's the first time after a blank or different token.
            if token_id == expected and prev_token_id != expected:
                orig_char_idx = char_to_token_idx[target_idx]
                timings[orig_char_idx] = frame_idx
                target_idx += 1

        prev_token_id = token_id

    # 5. Forward-fill timings for skipped characters (e.g. punctuation)
    char_to_token_set = set(char_to_token_idx)
    for i in range(len(text)):
        if i not in char_to_token_set:
            if i > 0:
                timings[i] = timings[i-1]
            else:
                timings[i] = 0

    # 6. Convert timings from frame indices to audio samples
    index_duration = len(samples) / (lpz.shape[0] + 1)
    timings = timings * index_duration

    return timings

def find_end_of_segment(text, timings, start):
    nchar = len(text)
    for idx in range(start, nchar):
        if idx < nchar - 1:
            cur = text[idx]
            nex = text[idx + 1]
            if nex not in TOKEN_PUNC:
                if cur in TOKEN_EOS:
                    break
                elif idx  - start >= CHARS_PER_SEGMENT:
                    if cur in TOKEN_COMMA or timings[idx + 1] - timings[idx] > PHONEMIC_BREAK:
                        break
    return idx

def split_text(model, samples, text):
    """Split texts into segments (with timestamps)"""
    try:
        timings = get_timings(model, samples, text)
    except Exception:
        return [(0, len(samples), text)]

    ret = []
    start = 0
    while start < len(text):
        end = find_end_of_segment(text, timings, start)
        ret.append((timings[start], timings[end], text[start:end + 1]))
        start = end + 1
    return ret
