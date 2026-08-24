"""ZeroMQ message encoding/decoding and payload serialization.

All inter-process messages are two-frame ZMQ multipart lists:
    frame 0 — fixed-width header: [msg_type (1 byte) | sender_id (4 bytes, signed)]
    frame 1 — variable-length payload

Message types in the wall-clock protocol:
    MODEL_STATE  node i → node j  (PUSH/PULL)   model weights for this round
    METRICS      node   → monitor (PUSH/PULL)   evaluation results per round

Round synchronisation is handled by the system clock — there are no control
messages from a coordinator to nodes.  Each node sleeps until its pre-agreed
wall-clock round-start time and proceeds independently.
"""

import enum
import io
import pickle
import struct
from typing import Any, Dict, List, Tuple

import torch


class MsgType(enum.IntEnum):
    MODEL_STATE = 0  # node → neighbour: model weights for a specific round
    METRICS     = 1  # node → monitor:   evaluation results for a specific round
    TOPO_CLAIM  = 2  # node → neighbour: signed topology observation (DMTT only)
    SKETCH      = 3  # node → neighbour: commitment digest + Count-Sketch (screening phase)
    FETCH_REQ   = 4  # node → neighbour: "you passed my screen, send your full model"
    FETCH_RESP  = 5  # node → neighbour: full model state, in reply to a FETCH_REQ


# Header: 1-byte msg_type + 4-byte signed int sender_id
_HDR_FMT  = "!Bi"
_HDR_SIZE = struct.calcsize(_HDR_FMT)

MONITOR_ID = -1  # sentinel used in headers sent by the monitor (if ever needed)


def encode(msg_type: MsgType, sender_id: int, payload: bytes) -> List[bytes]:
    """Pack a message into a two-frame ZMQ multipart list."""
    header = struct.pack(_HDR_FMT, int(msg_type), sender_id)
    return [header, payload]


def decode(frames: List[bytes]) -> Tuple[MsgType, int, bytes]:
    """Unpack a two-frame ZMQ multipart list.

    Returns:
        (msg_type, sender_id, payload)
    """
    header, payload = frames[0], frames[1]
    raw_type, sender_id = struct.unpack(_HDR_FMT, header)
    return MsgType(raw_type), sender_id, payload


# ---------------------------------------------------------------------------
# Payload serializers
# ---------------------------------------------------------------------------

def pack_state(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a PyTorch state dict to bytes (via torch.save)."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def unpack_state(data: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize bytes back to a CPU state dict."""
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def pack_obj(obj: Any) -> bytes:
    """Serialize any picklable Python object."""
    return pickle.dumps(obj, protocol=4)


def unpack_obj(data: bytes) -> Any:
    """Deserialize a pickle payload."""
    return pickle.loads(data)


# Screening phase (communication-efficient aggregation): a SKETCH payload is a fixed 32-byte
# commitment digest followed by the raw float32 Count-Sketch. This is O(k) on the wire, versus
# O(d) for a full MODEL_STATE, and is what makes selective fetch cheaper than full exchange.
_COMMIT_BYTES = 32  # SHA-256 digest


def pack_sketch(commitment: bytes, sketch: "Any") -> bytes:
    """Serialize (commitment, sketch) -> bytes.  `sketch` is a 1-D float32 numpy array."""
    import numpy as np
    if len(commitment) != _COMMIT_BYTES:
        raise ValueError(f"commitment must be {_COMMIT_BYTES} bytes, got {len(commitment)}")
    return commitment + np.asarray(sketch, dtype=np.float32).tobytes()


def unpack_sketch(data: bytes) -> Tuple[bytes, "Any"]:
    """Deserialize bytes -> (commitment, float32 sketch array)."""
    import numpy as np
    commitment = data[:_COMMIT_BYTES]
    sketch = np.frombuffer(data[_COMMIT_BYTES:], dtype=np.float32)
    return commitment, sketch
