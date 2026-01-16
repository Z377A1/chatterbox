try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")


from .tts import ChatterboxTTS as ChatterboxTTS
from .vc import ChatterboxVC as ChatterboxVC
from .mtl_tts import (
    ChatterboxMultilingualTTS as ChatterboxMultilingualTTS,
    SUPPORTED_LANGUAGES as SUPPORTED_LANGUAGES,
)
