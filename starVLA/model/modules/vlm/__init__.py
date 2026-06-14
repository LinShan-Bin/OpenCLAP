def get_vlm_model(config):
    """Dispatch to the appropriate VLM interface based on the model path.

    OpenCLAP ships only the Qwen2.5-VL and Qwen3-VL backbones (the two used
    in the CLAP paper). Add additional dispatch branches if you bring more
    backbones over from upstream starVLA.
    """
    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen2.5-VL" in vlm_name:
        from .QWen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name or "clap" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
    else:
        raise NotImplementedError(
            f"VLM backbone {vlm_name} is not supported in OpenCLAP. Bring the "
            "corresponding interface in from upstream starVLA if you need it."
        )
