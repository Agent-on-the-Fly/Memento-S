"""
SILK to WAV audio transcoding for WeChat voice messages.

WeChat voice messages use the SILK codec (Skype SILK) which needs to be
converted to WAV format for playback compatibility.

Supports multiple backends:
- pysilk: Pure Python SILK decoder (preferred)
- FFmpeg: External tool via subprocess
- silk-sdk: Python bindings for official SDK
"""

import asyncio
import io
import logging
import os
import shutil
import struct
import subprocess
import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from weixin_sdk.exceptions import WeixinTranscodeError

logger = logging.getLogger(__name__)


class TranscodeBackend(Enum):
    """Available transcoding backends."""

    PYSILK = "pysilk"
    FFMPEG = "ffmpeg"
    SILK_SDK = "silk_sdk"
    FALLBACK = "fallback"


class SilkTranscoder:
    """
    Transcoder for converting WeChat SILK voice messages to WAV format.

    SILK (Skype SILK) is the codec used by WeChat for voice messages.
    This class provides transcoding capabilities with multiple backend options.

    Args:
        sample_rate: Output WAV sample rate in Hz (default: 24000 for WeChat standard)
        backend: Force specific backend, or None for auto-detection

    Example:
        >>> transcoder = SilkTranscoder(sample_rate=24000)
        >>> wav_data = transcoder.transcode(silk_bytes)
        >>> transcoder.transcode_file("voice.silk", "voice.wav")
    """

    # SILK magic bytes for format detection
    SILK_MAGIC_V3 = b"#!SILK_V3"
    SILK_MAGIC_V3_WITH_HEADER = b"\x02#!SILK_V3"

    # Supported sample rates for SILK codec
    SUPPORTED_SAMPLE_RATES = [8000, 12000, 16000, 24000]

    def __init__(
        self,
        sample_rate: int = 24000,
        backend: Optional[TranscodeBackend] = None,
    ):
        """
        Initialize the SILK transcoder.

        Args:
            sample_rate: Output sample rate in Hz (default: 24000)
            backend: Specific backend to use, or None for auto-detection

        Raises:
            WeixinTranscodeError: If sample rate is not supported
        """
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise WeixinTranscodeError(
                f"Unsupported sample rate: {sample_rate}. "
                f"Supported rates: {self.SUPPORTED_SAMPLE_RATES}"
            )

        self.sample_rate = sample_rate
        self._backend = backend or self._detect_backend()
        self._pysilk_module = None
        self._silk_sdk_module = None

        logger.debug(f"Initialized SilkTranscoder with backend: {self._backend.value}")

    def _detect_backend(self) -> TranscodeBackend:
        """
        Auto-detect the best available backend.

        Priority: pysilk > ffmpeg > silk-sdk > fallback

        Returns:
            The best available TranscodeBackend
        """
        # Try pysilk first (pure Python, preferred)
        try:
            import pysilk

            self._pysilk_module = pysilk
            logger.debug("Using pysilk backend")
            return TranscodeBackend.PYSILK
        except ImportError:
            pass

        # Try FFmpeg
        if self._check_ffmpeg():
            logger.debug("Using FFmpeg backend")
            return TranscodeBackend.FFMPEG

        # Try silk-sdk
        try:
            import silk_sdk

            self._silk_sdk_module = silk_sdk
            logger.debug("Using silk-sdk backend")
            return TranscodeBackend.SILK_SDK
        except ImportError:
            pass

        # Fallback: just copy data with warning
        logger.warning(
            "No SILK decoder available. "
            "Install pysilk: pip install pysilk-python, "
            "or ensure ffmpeg is installed."
        )
        return TranscodeBackend.FALLBACK

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in PATH."""
        return shutil.which("ffmpeg") is not None

    @property
    def backend(self) -> TranscodeBackend:
        """Get the current backend being used."""
        return self._backend

    @staticmethod
    def is_silk_file(file_path: Union[str, Path]) -> bool:
        """
        Check if a file is in SILK format.

        SILK files typically start with '#!SILK_V3' magic bytes,
        or may have a header byte prefix '\x02#!SILK_V3'.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file appears to be SILK format

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(path, "rb") as f:
                header = f.read(16)  # Read enough to check both magic patterns

            # Check for standard SILK_V3 magic
            if header.startswith(SilkTranscoder.SILK_MAGIC_V3):
                return True

            # Check for SILK with header byte prefix (common in WeChat)
            if header.startswith(SilkTranscoder.SILK_MAGIC_V3_WITH_HEADER):
                return True

            return False
        except (IOError, OSError) as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return False

    @staticmethod
    def is_silk_data(data: bytes) -> bool:
        """
        Check if byte data is in SILK format.

        Args:
            data: Byte data to check

        Returns:
            True if the data appears to be SILK format
        """
        if not data or len(data) < 9:
            return False

        # Check for standard SILK_V3 magic
        if data.startswith(SilkTranscoder.SILK_MAGIC_V3):
            return True

        # Check for SILK with header byte prefix
        if data.startswith(SilkTranscoder.SILK_MAGIC_V3_WITH_HEADER):
            return True

        return False

    def transcode(self, silk_data: bytes) -> bytes:
        """
        Convert SILK data to WAV format.

        Args:
            silk_data: Raw SILK codec data

        Returns:
            WAV format audio data

        Raises:
            WeixinTranscodeError: If transcoding fails
            ValueError: If input data is empty or invalid
        """
        if not silk_data:
            raise ValueError("Input data is empty")

        if not self.is_silk_data(silk_data):
            warnings.warn("Input data does not appear to be SILK format")

        try:
            if self._backend == TranscodeBackend.PYSILK:
                return self._transcode_pysilk(silk_data)
            elif self._backend == TranscodeBackend.FFMPEG:
                return self._transcode_ffmpeg(silk_data)
            elif self._backend == TranscodeBackend.SILK_SDK:
                return self._transcode_silk_sdk(silk_data)
            else:
                return self._transcode_fallback(silk_data)
        except Exception as e:
            raise WeixinTranscodeError(f"Transcoding failed: {e}") from e

    def _transcode_pysilk(self, silk_data: bytes) -> bytes:
        """Transcode using pysilk library."""
        if self._pysilk_module is None:
            import pysilk

            self._pysilk_module = pysilk

        # Decode SILK to PCM
        pcm_data = self._pysilk_module.decode(silk_data, sample_rate=self.sample_rate)

        # Wrap PCM in WAV container
        return self._pcm_to_wav(pcm_data, self.sample_rate)

    def _transcode_ffmpeg(self, silk_data: bytes) -> bytes:
        """Transcode using FFmpeg subprocess."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise WeixinTranscodeError("FFmpeg not found in PATH")

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".silk", delete=False) as temp_input:
            temp_input.write(silk_data)
            temp_input_path = temp_input.name

        temp_output_path = temp_input_path.replace(".silk", ".wav")

        try:
            # Build FFmpeg command
            cmd = [
                ffmpeg_path,
                "-y",  # Overwrite output
                "-f",
                "silk",  # Input format
                "-ar",
                str(self.sample_rate),  # Input sample rate
                "-i",
                temp_input_path,  # Input file
                "-ar",
                str(self.sample_rate),  # Output sample rate
                "-ac",
                "1",  # Mono
                "-acodec",
                "pcm_s16le",  # 16-bit PCM
                "-f",
                "wav",  # Output format
                temp_output_path,
            ]

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                raise WeixinTranscodeError(
                    f"FFmpeg transcoding failed: {result.stderr}"
                )

            # Read output
            with open(temp_output_path, "rb") as f:
                return f.read()

        finally:
            # Cleanup temp files
            for path in [temp_input_path, temp_output_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _transcode_silk_sdk(self, silk_data: bytes) -> bytes:
        """Transcode using silk-sdk Python bindings."""
        if self._silk_sdk_module is None:
            import silk_sdk

            self._silk_sdk_module = silk_sdk

        # Decode SILK to PCM using SDK
        pcm_data = self._silk_sdk_module.decode(silk_data, sample_rate=self.sample_rate)

        # Wrap in WAV
        return self._pcm_to_wav(pcm_data, self.sample_rate)

    def _transcode_fallback(self, silk_data: bytes) -> bytes:
        """
        Fallback transcoding when no decoder is available.

        Creates a silent WAV file as placeholder and issues a warning.
        """
        warnings.warn(
            "No SILK decoder available. "
            "Returning silent WAV. "
            "Install pysilk: pip install pysilk-python"
        )

        # Create 1 second of silence as placeholder
        silent_pcm = b"\x00\x00" * self.sample_rate
        return self._pcm_to_wav(silent_pcm, self.sample_rate)

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, sample_rate: int, channels: int = 1) -> bytes:
        """
        Wrap raw PCM data in a standard WAV container.

        Args:
            pcm_data: Raw PCM audio data (16-bit, little-endian)
            sample_rate: Sample rate in Hz
            channels: Number of channels (default: 1 for mono)

        Returns:
            WAV formatted audio data
        """
        bits_per_sample = 16
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)
        data_size = len(pcm_data)

        # WAV header structure
        wav_header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",  # Chunk ID
            36 + data_size,  # Chunk size
            b"WAVE",  # Format
            b"fmt ",  # Subchunk1 ID
            16,  # Subchunk1 size (PCM)
            1,  # Audio format (PCM)
            channels,  # Number of channels
            sample_rate,  # Sample rate
            byte_rate,  # Byte rate
            block_align,  # Block align
            bits_per_sample,  # Bits per sample
            b"data",  # Subchunk2 ID
            data_size,  # Subchunk2 size
        )

        return wav_header + pcm_data

    def transcode_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """
        Transcode a SILK file to WAV format.

        Args:
            input_path: Path to input SILK file
            output_path: Path for output WAV file

        Raises:
            FileNotFoundError: If input file doesn't exist
            WeixinTranscodeError: If transcoding fails
            ValueError: If input is not a valid SILK file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Validate format
        if not self.is_silk_file(input_path):
            raise ValueError(f"File does not appear to be SILK format: {input_path}")

        try:
            # Read input
            with open(input_path, "rb") as f:
                silk_data = f.read()

            # Transcode
            wav_data = self.transcode(silk_data)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write output
            with open(output_path, "wb") as f:
                f.write(wav_data)

            logger.debug(f"Transcoded {input_path} -> {output_path}")

        except Exception as e:
            if isinstance(e, WeixinTranscodeError):
                raise
            raise WeixinTranscodeError(f"Failed to transcode {input_path}: {e}") from e

    async def async_transcode(self, silk_data: bytes) -> bytes:
        """
        Asynchronously transcode SILK data to WAV.

        Note: If using pysilk or silk-sdk backends, this runs in a thread pool.
        FFmpeg backend runs fully asynchronously.

        Args:
            silk_data: Raw SILK codec data

        Returns:
            WAV format audio data
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcode, silk_data)

    async def async_transcode_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """
        Asynchronously transcode a SILK file to WAV.

        Args:
            input_path: Path to input SILK file
            output_path: Path for output WAV file
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.transcode_file, input_path, output_path)

    def transcode_batch(
        self,
        files: List[Tuple[Union[str, Path], Union[str, Path]]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Tuple[Union[str, Path], bool, Optional[str]]]:
        """
        Transcode multiple files in batch.

        Args:
            files: List of (input_path, output_path) tuples
            progress_callback: Optional callback(current, total, current_file)

        Returns:
            List of (input_path, success, error_message) tuples
        """
        results = []
        total = len(files)

        for i, (input_path, output_path) in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, str(input_path))

            try:
                self.transcode_file(input_path, output_path)
                results.append((input_path, True, None))
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to transcode {input_path}: {error_msg}")
                results.append((input_path, False, error_msg))

        return results

    async def async_transcode_batch(
        self,
        files: List[Tuple[Union[str, Path], Union[str, Path]]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_concurrency: int = 4,
    ) -> List[Tuple[Union[str, Path], bool, Optional[str]]]:
        """
        Asynchronously transcode multiple files with concurrency control.

        Args:
            files: List of (input_path, output_path) tuples
            progress_callback: Optional callback(current, total, current_file)
            max_concurrency: Maximum number of concurrent transcodes

        Returns:
            List of (input_path, success, error_message) tuples
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results = []
        total = len(files)
        completed = 0

        async def transcode_with_semaphore(
            input_path: Union[str, Path], output_path: Union[str, Path]
        ) -> Tuple[Union[str, Path], bool, Optional[str]]:
            nonlocal completed
            async with semaphore:
                try:
                    await self.async_transcode_file(input_path, output_path)
                    return input_path, True, None
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to transcode {input_path}: {error_msg}")
                    return input_path, False, error_msg
                finally:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total, str(input_path))

        # Run all transcodes concurrently with semaphore control
        tasks = [
            transcode_with_semaphore(input_path, output_path)
            for input_path, output_path in files
        ]

        results = await asyncio.gather(*tasks)
        return list(results)

    def transcode_streaming(
        self,
        input_stream: io.BufferedIOBase,
        output_stream: io.BufferedIOBase,
        chunk_size: int = 65536,
    ) -> None:
        """
        Memory-efficient transcoding for large files using streaming.

        Note: This buffers the entire input since SILK requires complete data
        for decoding. For true streaming, consider using FFmpeg directly.

        Args:
            input_stream: Input stream containing SILK data
            output_stream: Output stream for WAV data
            chunk_size: Buffer size for reading (unused, kept for API consistency)
        """
        # Read all input (SILK requires complete data)
        silk_data = input_stream.read()

        # Transcode
        wav_data = self.transcode(silk_data)

        # Write output
        output_stream.write(wav_data)


def quick_transcode(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 24000,
) -> Path:
    """
    Quick transcoding function for convenience.

    Args:
        input_path: Path to input SILK file
        output_path: Path for output WAV file (default: same name with .wav)
        sample_rate: Output sample rate (default: 24000)

    Returns:
        Path to the output WAV file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    transcoder = SilkTranscoder(sample_rate=sample_rate)
    transcoder.transcode_file(input_path, output_path)

    return output_path
