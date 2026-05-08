try:
    import perth

    if not hasattr(perth, "PerthImplicitWatermarker"):
        class _DummyImplicitWatermarker:
            """No-op replacement for perth.PerthImplicitWatermarker."""
            def __init__(self, *args, **kwargs):
                pass

            def apply_watermark(self, wav, *args, **kwargs):
                return wav

            def __call__(self, wav, *args, **kwargs):
                return wav

        perth.PerthImplicitWatermarker = _DummyImplicitWatermarker
        print("[patch_perth] Injected dummy PerthImplicitWatermarker")
    else:
        # Already has the class, no-op
        pass

except ImportError:
    # perth not installed at all — let chatterbox fail with clearer message
    pass
