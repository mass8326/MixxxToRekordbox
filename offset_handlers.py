# Core logic borrowed from https://github.com/FrankwaP/mixxx-utils

from dataclasses import dataclass
from logging import ERROR
from pathlib import Path
from typing import BinaryIO, Iterator, Literal
import struct
import sys

import eyed3.mp3.headers  # type: ignore


Mp3Decoder = Literal["MAD", "CoreAudio", "FFmpeg"]
ACCEPTED_MP3_DECODERS: list[Mp3Decoder] = ["MAD", "CoreAudio", "FFmpeg"]


eyed3.core.log.setLevel(ERROR)
eyed3.id3.frames.log.setLevel(ERROR)
eyed3.mp3.headers.log.setLevel(ERROR)

OFFSET_ERROR_MESSAGES: list[str] = []


def has_xing_info(audiofile: eyed3.mp3.Mp3AudioFile) -> bool:
    return audiofile.info.xing_header is not None


def has_lame_tag(audiofile: eyed3.mp3.Mp3AudioFile) -> bool:
    return len(audiofile.info.lame_tag) > 0


def has_valid_CRC_tag(audiofile: eyed3.mp3.Mp3AudioFile) -> bool:
    try:
        return audiofile.info.lame_tag["music_crc"] > 0
    except KeyError:
        return False


def get_case_mp3(audiofile: eyed3.mp3.Mp3AudioFile) -> Literal["A", "B", "C", "D"]:
    if not has_xing_info(audiofile):
        return "A"
    elif not has_lame_tag(audiofile):
        return "B"
    elif not has_valid_CRC_tag(audiofile):
        return "C"
    else:
        return "D"


def get_offset_mp3(audiofile: eyed3.mp3.Mp3AudioFile, mp3_decoder: Mp3Decoder) -> int:
    check_mp3_decoder_value(mp3_decoder)
    #
    case = get_case_mp3(audiofile)
    if mp3_decoder == "MAD":
        if case == "A" or case == "D":
            return 26
    if mp3_decoder == "CoreAudio":
        if case == "A":
            return 13
        if case == "B":
            return 11
        if case == "C":
            return 26
        if case == "D":
            return 50
    if mp3_decoder == "FFmpeg":
        if case == "D":
            return 26
    return 0


def check_mp3_decoder_value(mp3_decoder: str) -> None:
    if mp3_decoder not in ACCEPTED_MP3_DECODERS:
        raise ValueError(
            "Incorrect value for Mixxx encoder: expecting {ACCEPTED_MP3_DECODERS}"
        )


MP4_FORMATS = (".m4a", ".mp4")
# Encoder delay assumed for AAC files that don't record their own
# This is the iTunes/Apple encoder default
DEFAULT_AAC_PRIMING_SAMPLES = 2112
# The above priming assuming 44.1kHz sample rate
FALLBACK_MP4_OFFSET_MS = 48
# Assumes FFmpeg's AAC priming of 1024 samples at 44.1kHz sample rate
# This is likely what the stem files are encoded with
FALLBACK_STEM_OFFSET_MS = 24


@dataclass
class Mp4AudioInfo:
    codec: str  # sample entry fourcc, e.g. "mp4a" (AAC) or "alac"
    timescale: int  # media timescale, which is the sample rate for audio
    edit_list_start: int | None  # first edit's media_time, in timescale units
    itunsmpb_priming: int | None  # priming samples from the iTunSMPB tag

    @property
    def priming_samples(self) -> int:
        if self.edit_list_start:
            return self.edit_list_start
        if self.itunsmpb_priming:
            return self.itunsmpb_priming
        if self.codec == "mp4a":
            return DEFAULT_AAC_PRIMING_SAMPLES
        return 0

    @property
    def priming_ms(self) -> float:
        return self.priming_samples * 1000.0 / self.timescale


def _iter_boxes(f: BinaryIO, start: int, end: int) -> Iterator[tuple[str, int, int]]:
    """Yield (type, content start, content end) for each box in [start, end)."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, box_type = struct.unpack(">I4s", f.read(8))
        header = 8
        if size == 1:
            (size,) = struct.unpack(">Q", f.read(8))
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            raise ValueError(f"Invalid MP4 box size {size}")
        yield box_type.decode("latin-1"), pos + header, min(pos + size, end)
        pos += size


def _find_box(f: BinaryIO, start: int, end: int, *path: str) -> tuple[int, int] | None:
    for box_type, box_start, box_end in _iter_boxes(f, start, end):
        if box_type == path[0]:
            if len(path) == 1:
                return box_start, box_end
            return _find_box(f, box_start, box_end, *path[1:])
    return None


def _read_box(f: BinaryIO, box: tuple[int, int]) -> bytes:
    f.seek(box[0])
    return f.read(box[1] - box[0])


def _parse_mdhd_timescale(data: bytes) -> int:
    # version(1) flags(3), then creation/modification times (4 or 8 bytes each)
    offset = 20 if data[0] == 1 else 12
    return struct.unpack_from(">I", data, offset)[0]


def _parse_elst_start(data: bytes) -> int | None:
    version = data[0]
    (entry_count,) = struct.unpack_from(">I", data, 4)
    offset = 8
    for _ in range(entry_count):
        if version == 1:
            _, media_time, _ = struct.unpack_from(">QqI", data, offset)
            offset += 20
        else:
            _, media_time, _ = struct.unpack_from(">IiI", data, offset)
            offset += 12
        if media_time != -1:  # -1 marks an empty edit
            return media_time
    return None


def _parse_itunsmpb(f: BinaryIO, moov: tuple[int, int]) -> int | None:
    meta = _find_box(f, *moov, "udta", "meta")
    if not meta:
        return None
    # iTunes-style meta is a full box with 4 bytes of version/flags before
    # its children, QuickTime-style meta is not
    f.seek(meta[0] + 4)
    meta_start = meta[0] if f.read(4) == b"hdlr" else meta[0] + 4
    ilst = _find_box(f, meta_start, meta[1], "ilst")
    if not ilst:
        return None
    for box_type, box_start, box_end in _iter_boxes(f, *ilst):
        if box_type != "----":
            continue
        name = data = None
        for child_type, child_start, child_end in _iter_boxes(f, box_start, box_end):
            if child_type == "name":
                # version(1) flags(3) then the name
                name = _read_box(f, (child_start, child_end))[4:]
            elif child_type == "data":
                # type(4) locale(4) then the value
                data = _read_box(f, (child_start, child_end))[8:]
        if name == b"iTunSMPB" and data:
            # " 00000000 <priming> <padding> <sample count> ..." in hex
            return int(data.decode("ascii").split()[1], 16)
    return None


def read_mp4_audio_info(path: str | Path) -> Mp4AudioInfo:
    """Read codec, timescale, and encoder delay of the first audio track.

    For stem files, the first audio track is the stereo master mix.
    """
    with open(path, "rb") as f:
        f.seek(0, 2)
        moov = _find_box(f, 0, f.tell(), "moov")
        if not moov:
            raise ValueError("No moov box found")
        for box_type, trak_start, trak_end in _iter_boxes(f, *moov):
            if box_type != "trak":
                continue
            hdlr = _find_box(f, trak_start, trak_end, "mdia", "hdlr")
            # version(1) flags(3) pre_defined(4) handler_type(4)
            if not hdlr or _read_box(f, hdlr)[8:12] != b"soun":
                continue
            mdhd = _find_box(f, trak_start, trak_end, "mdia", "mdhd")
            stsd = _find_box(f, trak_start, trak_end, "mdia", "minf", "stbl", "stsd")
            if not mdhd or not stsd:
                raise ValueError("Audio track is missing mdhd or stsd")
            # version(1) flags(3) entry_count(4) then the first sample
            # entry's size(4) and format(4)
            codec = _read_box(f, stsd)[12:16].decode("latin-1")
            elst = _find_box(f, trak_start, trak_end, "edts", "elst")
            return Mp4AudioInfo(
                codec=codec,
                timescale=_parse_mdhd_timescale(_read_box(f, mdhd)),
                edit_list_start=_parse_elst_start(_read_box(f, elst)) if elst else None,
                itunsmpb_priming=_parse_itunsmpb(f, moov),
            )
    raise ValueError("No audio track found")


def get_offset_mp4(track_path: str | Path, source_path: str | Path) -> float:
    """Offset for MP4 audio: the encoder delay (priming samples) at the start
    of the stream, which Mixxx skips and Rekordbox does not."""
    # With a virtual out dir the exported file may not exist locally, but if
    # it was copied rather than transcoded the source has the same content
    readable_path = track_path
    if not Path(track_path).exists() and Path(source_path).suffix == Path(track_path).suffix:
        readable_path = source_path
    try:
        return read_mp4_audio_info(readable_path).priming_ms
    except Exception as ex:
        OFFSET_ERROR_MESSAGES.append(f"{readable_path}: {ex}")
        if Path(track_path).name.lower().endswith(tuple(f".stem{ext}" for ext in MP4_FORMATS)):
            return FALLBACK_STEM_OFFSET_MS
        return FALLBACK_MP4_OFFSET_MS


def get_offset_ms(
    track_path: str | Path, source_path: str | Path, mp3_decoder: Mp3Decoder
) -> float:
    in_format = Path(source_path).suffix
    out_format = Path(track_path).suffix
    if out_format in MP4_FORMATS:
        return get_offset_mp4(track_path, source_path)
    elif in_format == out_format == ".mp3":
        try:
            audiofile = eyed3.load(track_path)
            return get_offset_mp3(audiofile, mp3_decoder)
        except Exception as ex:
            OFFSET_ERROR_MESSAGES.append(f"{track_path}: {ex}")
            return 0
    elif in_format == ".flac" and out_format == ".wav":
        return 26
    elif in_format == ".flac" and out_format == ".mp3":
        return 26

    print(
        f"Warning: transcoding from {in_format} to {out_format} is experimental and may result in misaligned cues",
        file=sys.stderr
    )

    return 0


def get_offset_sec(
    track_path: str | Path, source_path: str | Path, mp3_decoder: Mp3Decoder = "MAD"
) -> float:
    return get_offset_ms(track_path, source_path, mp3_decoder) / 1000.0


def flush_offset_errors() -> None:
    if not OFFSET_ERROR_MESSAGES:
        return
    print("Unable to determine offsets for the following tracks:")
    for error_message in OFFSET_ERROR_MESSAGES:
        print(error_message)
    OFFSET_ERROR_MESSAGES.clear()
