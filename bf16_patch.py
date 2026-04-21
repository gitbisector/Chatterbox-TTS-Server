"""Patch engine.py to wrap generate() with BF16 autocast for ~2x speedup."""

with open("/app/engine.py", "r") as f:
    content = f.read()

old = """        # Call the core model's generate method
        # Multilingual model requires language_id parameter
        if loaded_model_type == "multilingual":
            wav_tensor = chatterbox_model.generate(
                text=text,
                language_id=language,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        else:
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )"""

new = """        # Call the core model's generate method with BF16 autocast for speed
        import torch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(model_device == "cuda")):
            # Multilingual model requires language_id parameter
            if loaded_model_type == "multilingual":
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    language_id=language,
                    audio_prompt_path=audio_prompt_path,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
            else:
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    audio_prompt_path=audio_prompt_path,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )"""

if old in content:
    content = content.replace(old, new)
    with open("/app/engine.py", "w") as f:
        f.write(content)
    print("BF16 autocast patch applied")
else:
    print("WARNING: patch target not found, engine.py may have changed")
