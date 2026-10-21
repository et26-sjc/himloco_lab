"""Current frame first, followed by past frames; zero padding at episode boundaries."""
import torch

def stack_frame(frame, previous, past_frames, reset_ids=None):
    if previous.shape != (frame.shape[0], frame.shape[1] * (past_frames + 1)):
        raise ValueError("Invalid history shape")
    result = torch.cat((frame, previous[:, :past_frames * frame.shape[1]]), dim=-1)
    if reset_ids is not None:
        result[reset_ids, frame.shape[1]:] = 0
    return result
