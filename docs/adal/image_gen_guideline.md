# Image Generation Architecture Guideline

**Date:** 2025-04-27
**Author:** AdaL
**TL;DR:** The image generation pipeline uses `ModelClient` → `Generator` → `GeneratorOutput` — the same unified flow as text generation. The `ImageModelAdapter` layer is redundant and should be eliminated by having each `ModelClient` handle `IMAGE_GENERATION` natively in `convert_inputs_to_api_kwargs`, `acall`, and response parsing.

---

## 1. The Correct Architecture (Google Client = Reference)

The Google client (`GoogleGenAIClient`) implements image generation as a first-class citizen of the existing `ModelClient` → `Generator` pipeline:

```
ImageGenerationTool.generate_image()
  └─ Generator.acall(prompt_kwargs, model_kwargs)
       ├─ _pre_call()
       │    └─ model_client.convert_inputs_to_api_kwargs(input, model_kwargs, IMAGE_GENERATION)
       │         → Translates tool params (aspect_ratio, image_size, n, images)
       │           into provider-specific API kwargs
       ├─ model_client.acall(api_kwargs, IMAGE_GENERATION)
       │    → Routes to provider's image API (generate_content / images.generate)
       └─ _async_post_call()
            └─ model_client.parse_chat_completion(completion)
                 → Detects image response → parse_image_generation_response()
                 → Returns GeneratorOutput(data=text, images=[{b64_json, mime_type}])
```

### Key design principle

**Image generation reuses the exact same Generator flow as text generation.** The only differences are:

| Concern | Text (LLM) | Image (IMAGE_GENERATION) |
|---------|-------------|--------------------------|
| `convert_inputs_to_api_kwargs` | Builds messages/input | Builds prompt + image config |
| `acall` routing | `responses.create` / `generate_content` | `images.generate` / `generate_content(modalities=IMAGE)` |
| Response parser | Extracts text → `data` | Extracts text → `data`, images → `images` |
| `GeneratorOutput.data` | Text content | Text content (or None for image-only) |
| `GeneratorOutput.images` | None | List of image dicts |

---

## 2. GeneratorOutput Contract for Images

```python
class GeneratorOutput:
    data: Optional[str]           # Text content (None if image-only)
    images: Optional[List[dict]]  # Image data (None if text-only)
    raw_response: str             # Lightweight — NEVER store base64 here
    metadata: Optional[dict]      # Grounding, usage, etc.
```

### Image dict format (MUST follow)

The `images` field must contain dicts in one of these formats:

```python
# Format 1: Base64-encoded image data
{"b64_json": "<base64_string>", "mime_type": "image/png"}

# Format 2: URL reference
{"url": "https://..."}
```

This is the contract that `ImageGenerationTool._save_image_to_file()` depends on:

```python
def _save_image_to_file(self, image_data, filepath):
    if isinstance(image_data, dict):
        if "b64_json" in image_data:
            img_bytes = base64.b64decode(image_data["b64_json"])
            # write bytes to file
        elif "url" in image_data:
            urllib.request.urlretrieve(image_data["url"], filepath)
    elif isinstance(image_data, bytes):
        # write raw bytes to file
```

### Single vs multiple images

```python
# Single image (n=1): unwrap from list
images = image_dict            # dict, not list

# Multiple images (n>1): keep as list
images = [image_dict_0, image_dict_1, ...]
```

---

## 3. What Each ModelClient Must Implement

### 3.1 `convert_inputs_to_api_kwargs` — Handle IMAGE_GENERATION

The client does NOT need to translate or unify parameters. The caller (agent/tool) already knows which provider it's targeting and passes provider-native params directly. The client just needs to:
1. Accept `IMAGE_GENERATION` without raising `ValueError`
2. Set the prompt from `input`
3. Strip text-gen params that don't apply (temperature, max_tokens)

**Google:**
```python
elif model_type == ModelType.IMAGE_GENERATION:
    prompt = input if input else api_kwargs.get("prompt", "")
    api_kwargs["prompt"] = prompt
    images = api_kwargs.pop("images", None)
    if images:
        api_kwargs["input_images"] = images
    api_kwargs.pop("temperature", None)
    api_kwargs.pop("max_tokens", None)
```

**OpenAI:**
```python
elif model_type == ModelType.IMAGE_GENERATION:
    api_kwargs["prompt"] = input
    # The caller passes OpenAI-native params directly:
    #   size="2048x1152", output_format="png", quality="high", n=1
    # No translation needed — the agent knows what provider it's calling.
    api_kwargs.pop("temperature", None)
    api_kwargs.pop("max_tokens", None)
```

**Key principle:** Parameters do NOT have to be unified across providers. The agent can use different parameter sets for OpenAI vs Google. Each provider has its own API shape — the model client just passes them through. Translation (like aspect_ratio → pixel size) belongs in the caller (tool/agent), not the model client.

### 3.2 `acall` — Route to image API

**Google:**
```python
elif model_type == ModelType.IMAGE_GENERATION:
    return await self._async_impl_call_image_generation(api_kwargs)
```

**OpenAI:**
```python
elif model_type == ModelType.IMAGE_GENERATION:
    if "image" in api_kwargs:
        response = await self.async_client.images.edit(**api_kwargs)
    else:
        response = await self.async_client.images.generate(**api_kwargs)
    return response.data
```

### 3.3 `parse_image_generation_response` — Return correct GeneratorOutput

**Google (reference):**
```python
def parse_image_generation_response(self, response):
    parsed_images = []   # [{"b64_json": str, "mime_type": str}, ...]
    parsed_texts = []    # [str, ...]

    # ... extract from response.candidates[].content.parts ...

    return GeneratorOutput(
        data="\n".join(parsed_texts) if parsed_texts else None,
        images=parsed_images[0] if len(parsed_images) == 1 else parsed_images,
        raw_response="",  # Never store base64 here
    )
```

**OpenAI (fixed):**
```python
def parse_image_generation_response(self, response: List[Image]):
    images = []
    for img in response:
        if img.b64_json:
            images.append({"b64_json": img.b64_json})
        elif img.url:
            images.append({"url": img.url})
    return GeneratorOutput(
        data=None,  # OpenAI image API returns no text
        images=images[0] if len(images) == 1 else images,
        raw_response="",
    )
```

---

## 4. Why ImageModelAdapter Exists and Why It Should Be Eliminated

### The core design principle

The `ImageGenerationTool` should be a **thin orchestration layer** that:
1. Resolves which model to use
2. Passes raw `model_kwargs` + prompt to the `Generator`
3. Saves the resulting images to disk

It should **NOT** handle provider-specific input translation — that's the model client's job via `convert_inputs_to_api_kwargs`, just like it works for text generation.

### Why the adapter was introduced (the actual reason)

The adapter exists because **OpenAI's `convert_inputs_to_api_kwargs` doesn't handle `ModelType.IMAGE_GENERATION`**:

```python
# openai_client.py L934-985 (current state)
def convert_inputs_to_api_kwargs(self, input, model_kwargs, model_type):
    if model_type == ModelType.EMBEDDER:
        ...
    elif model_type == ModelType.LLM or model_type == ModelType.LLM_REASONING:
        ...
    else:
        raise ValueError(f"model_type {model_type} is not supported")  # ← IMAGE_GENERATION hits this!
```

Since the OpenAI client can't translate image params, someone created the adapter to do it *before* the Generator flow. The adapter became a **bypass layer** that pre-translates kwargs so the model client doesn't have to.

### What the adapter actually does

```python
# OpenAIImageAdapter.build_model_kwargs() does:
# 1. Convert aspect_ratio + image_size → "2048x1152" pixel string (30 lines of math)
# 2. Infer output_format from file extension (.png → "png")
# 3. Open input_images as file handles (OpenAI SDK needs IO[bytes])
# 4. Validate google_search_grounding is not used
# 5. Set quality="low" for budget model variant

# GoogleImageAdapter.build_model_kwargs() does:
# 1. Pass through aspect_ratio, image_size, n as-is
# 2. Add google_search_grounding to kwargs
# 3. Resolve input_images paths
```

### The current (broken) flow

```
ImageGenerationTool.generate_image()
  │
  ├─ adapter.build_model_kwargs()          ← Provider translation done HERE (tool layer)
  │    └─ Returns fully-translated kwargs (e.g., size="2048x1152")
  │
  ├─ Generator._pre_call()
  │    └─ client.convert_inputs_to_api_kwargs(IMAGE_GENERATION)
  │         └─ OpenAI: raise ValueError!   ← BROKEN — doesn't handle IMAGE_GENERATION
  │            Google: strips temperature, moves images to input_images
  │
  └─ (never reached for OpenAI because ValueError kills the flow)
```

**This means OpenAI image generation through this flow has a latent bug:**
- The adapter pre-translates kwargs correctly
- But `convert_inputs_to_api_kwargs` raises `ValueError("model_type IMAGE_GENERATION is not supported")`
- The tool catches the exception and returns `ToolOutput(observation="Error: ...")`
- It only "works" if there's additional error handling or if the flow bypasses `convert_inputs_to_api_kwargs`

### The problems with the adapter approach

1. **Tool does provider-specific work it shouldn't**: The `ImageGenerationTool` calls `adapter.build_model_kwargs()` which does OpenAI pixel math, file handle management, format inference. This provider logic belongs in the model client, not the tool.

2. **Split ownership creates contract violations**: The adapter handles INPUT translation, but OUTPUT parsing lives in the model client. No shared interface enforces that they agree on format. This is exactly how the OpenAI `parse_image_generation_response` bug happened — it returned `data=base64_string` instead of `images=[{"b64_json": ...}]` because nobody enforced the contract.

3. **Google doesn't need an adapter at all**: The `GoogleGenAIClient` already handles IMAGE_GENERATION natively in `convert_inputs_to_api_kwargs` (L1030-1037) and `acall` (L1191-1193). The `GoogleImageAdapter` is just a passthrough wrapper.

4. **Duplicated responsibility**: For Google, both the adapter's `build_model_kwargs()` and the client's `convert_inputs_to_api_kwargs()` translate the same params. For OpenAI, the adapter does all the work but `convert_inputs_to_api_kwargs` raises ValueError anyway.

### The correct flow (no adapter)

```
ImageGenerationTool.generate_image()
  │
  │  # The tool/agent knows which provider it's calling and passes
  │  # provider-native params directly — no unification needed.
  │
  ├─ (OpenAI) Generator.acall(prompt_kwargs={"prompt": prompt}, model_kwargs={
  │      "model": "gpt-image-2",
  │      "size": "2048x1152",         ← OpenAI-native param
  │      "output_format": "png",      ← OpenAI-native param
  │      "n": 1,
  │  })
  │
  ├─ (Google) Generator.acall(prompt_kwargs={"prompt": prompt}, model_kwargs={
  │      "model": "gemini-3.1-flash-image-preview",
  │      "aspect_ratio": "16:9",      ← Google-native param
  │      "image_size": "2K",          ← Google-native param
  │      "n": 1,
  │  })
  │
  ├─ Generator._pre_call()
  │    └─ client.convert_inputs_to_api_kwargs(input, kwargs, IMAGE_GENERATION)
  │         └─ Sets prompt, strips temperature/max_tokens — passthrough otherwise
  │
  ├─ client.acall(api_kwargs, IMAGE_GENERATION)
  │    ├─ OpenAI: images.generate(**api_kwargs) → response.data
  │    └─ Google: generate_content(modalities=IMAGE) → response
  │
  └─ client.parse_chat_completion(completion)
       └─ parse_image_generation_response()
            → GeneratorOutput(data=text, images=[{"b64_json": ...}])
```

**Key principle: parameters do NOT have to be unified.** The agent/tool knows which provider it's targeting and can pass provider-native params directly. Each provider has its own API shape. The model client's job for IMAGE_GENERATION is:
1. Accept the model type without raising ValueError
2. Set the prompt from input
3. Route to the correct API endpoint
4. Parse the response into the unified `GeneratorOutput` format

Provider-specific param translation (e.g., aspect_ratio → pixel dimensions for OpenAI) belongs in the **tool/agent layer**, not the model client.

### Migration path

1. **OpenAI client**: Add `IMAGE_GENERATION` case to `convert_inputs_to_api_kwargs` (just accept it, set prompt, strip text-gen params). No pixel math needed — the caller passes `size="2048x1152"` directly.
2. **Google client**: Already done — `convert_inputs_to_api_kwargs` handles IMAGE_GENERATION at L1030.
3. **ImageGenerationTool**: Move provider-specific param building (OpenAI's pixel math, file handle opening, output_format inference) into the tool itself, keyed by which provider is being called. The tool already knows the provider — it just needs to build the right kwargs before passing to Generator.
4. **Delete adapters**: Remove `image_generation_adapters.py`. The tool handles param building, the client handles API routing + response parsing.

---

## 5. OpenAI Image Generation API Reference

> **Guide:** [Image generation | OpenAI API](https://developers.openai.com/api/docs/guides/image-generation?api=image)
> **API Ref:** [Create Image](https://developers.openai.com/api/reference/resources/images/methods/generate/)
> **SDK:** `openai.resources.images.Images.generate()` ([source](https://github.com/openai/openai-python))

### 5.1 Available Models

| Model | Use Case | Notes |
|-------|----------|-------|
| `gpt-image-2` | Latest, best quality | Flexible resolution, always returns b64. **No transparent backgrounds.** |
| `gpt-image-1.5` | Previous generation | Supports transparent backgrounds |
| `gpt-image-1` | Original GPT Image | Supports transparent backgrounds |
| `gpt-image-1-mini` | Budget / fast | Lower cost |
| `dall-e-3` | Legacy | Only n=1, response_format url/b64_json |
| `dall-e-2` | Legacy | 256/512/1024 only |

### 5.2 `images.generate()` — Request Parameters

| Parameter | Type | Required | Values | Applies To |
|-----------|------|----------|--------|------------|
| `prompt` | `str` | ✅ | Max 32000 chars (GPT), 4000 (dall-e-3), 1000 (dall-e-2) | All |
| `model` | `str` | ❌ | See model table above | All |
| `n` | `int` | ❌ | 1–10 (dall-e-3: only 1) | All |
| `size` | `str` | ❌ | See size section below. Default: `"auto"` | All |
| `quality` | `str` | ❌ | `"auto"` (default), `"low"`, `"medium"`, `"high"` | GPT models |
| | | | `"standard"`, `"hd"` | dall-e-3 |
| `output_format` | `str` | ❌ | `"png"` (default), `"jpeg"`, `"webp"` | GPT models only |
| `output_compression` | `int` | ❌ | 0–100 (default: 100). Only for webp/jpeg | GPT models only |
| `background` | `str` | ❌ | `"auto"` (default), `"transparent"`, `"opaque"` | GPT models (NOT gpt-image-2) |
| `moderation` | `str` | ❌ | `"auto"` (default), `"low"` (less restrictive) | GPT models only |
| `response_format` | `str` | ❌ | `"url"`, `"b64_json"` | dall-e-2/3 only (GPT always returns b64) |
| `style` | `str` | ❌ | `"vivid"`, `"natural"` | dall-e-3 only |
| `stream` | `bool` | ❌ | `true`, `false` (default) | GPT models only |
| `partial_images` | `int` | ❌ | 0–3 (for streaming) | GPT models only |
| `user` | `str` | ❌ | End-user identifier for abuse monitoring | All |

### 5.3 Size Options — `gpt-image-2` Accepts Flexible Resolutions

**`gpt-image-2` accepts ANY resolution** that satisfies these constraints:

| Constraint | Value |
|-----------|-------|
| Max edge length | ≤ 3840px |
| Both edges | Must be multiples of 16px |
| Aspect ratio | Long edge : short edge ≤ 3:1 |
| Min total pixels | ≥ 655,360 (roughly 810×810) |
| Max total pixels | ≤ 8,294,400 (roughly 2880×2880) |

**Popular sizes:**

| Size | Aspect | Description |
|------|--------|-------------|
| `"auto"` | auto | **Default.** Model picks best size for the prompt |
| `"1024x1024"` | 1:1 | Square (fastest) |
| `"1536x1024"` | 3:2 | Landscape |
| `"1024x1536"` | 2:3 | Portrait |
| `"2048x2048"` | 1:1 | 2K square |
| `"2048x1152"` | 16:9 | 2K landscape |
| `"3840x2160"` | 16:9 | 4K landscape |
| `"2160x3840"` | 9:16 | 4K portrait |

> ⚠️ Outputs above 2560×1440 (~3.7M pixels, "2K") are considered **experimental** by OpenAI.
>
> ⚠️ `"jpeg"` is faster than `"png"` — prioritize jpeg when latency matters.

**Size options for other models:**

| Model | Supported Sizes |
|-------|----------------|
| dall-e-2 | `"256x256"`, `"512x512"`, `"1024x1024"` |
| dall-e-3 | `"1024x1024"`, `"1792x1024"`, `"1024x1792"` |
| gpt-image-1/1.5/mini | `"1024x1024"`, `"1536x1024"`, `"1024x1536"`, `"auto"` |

### 5.4 `images.generate()` — Response (`ImagesResponse`)

```python
{
    "created": 1713833628,           # Unix timestamp
    "background": "opaque",          # "transparent" | "opaque"
    "output_format": "png",          # "png" | "webp" | "jpeg"
    "quality": "high",               # "low" | "medium" | "high"
    "size": "1024x1024",             # Actual size used
    "data": [                        # List of Image objects
        {
            "b64_json": "iVBOR...",  # Base64 image (GPT models always)
            "revised_prompt": "...", # dall-e-3 only
            "url": "https://..."    # dall-e-2/3 only (when response_format="url")
        }
    ],
    "usage": {                       # GPT models only
        "input_tokens": 50,
        "output_tokens": 50,
        "total_tokens": 100,
        "input_tokens_details": {"text_tokens": 10, "image_tokens": 40},
        "output_tokens_details": {"image_tokens": 45, "text_tokens": 5}
    }
}
```

### 5.5 `images.edit()` — Additional Parameters

| Parameter | Type | Values | Notes |
|-----------|------|--------|-------|
| `image` | `FileTypes \| List[FileTypes]` | File object(s) | Required. GPT: up to 16 images, png/webp/jpg < 50MB each. dall-e-2: 1 square png < 4MB |
| `prompt` | `str` | Editing instruction | Required. Max 32000 chars (GPT), 1000 (dall-e-2) |
| `mask` | `FileTypes` | PNG with alpha channel | Optional. Transparent areas = where to edit. Same dimensions as image |
| `input_fidelity` | `str` | `"high"`, `"low"` (default) | GPT models only (NOT gpt-image-2 which is always high). How closely to match input style/features |

### 5.6 Cost & Latency

**`gpt-image-2` pricing (output tokens → cost):**

| Quality | 1024×1024 | 1024×1536 | 1536×1024 |
|---------|-----------|-----------|-----------|
| Low | $0.006 | $0.005 | $0.005 |
| Medium | $0.053 | $0.041 | $0.041 |
| High | $0.211 | $0.165 | $0.165 |

- Use `quality: "low"` for fast drafts, thumbnails, quick iterations
- Complex prompts may take up to 2 minutes
- `jpeg` output is faster than `png`
- Streaming partial images: each partial adds ~100 output tokens

### 5.7 Key Behaviors & Gotchas

1. **GPT models always return `b64_json`** — the `response_format` param is ignored; URLs are not available. Our `parse_image_generation_response` must check `img.b64_json` first.

2. **`gpt-image-2` does NOT support transparent backgrounds** — `background: "transparent"` will error. Use `gpt-image-1.5` or `gpt-image-1` for transparency.

3. **`size="auto"` is the default** — the model picks the best size. Our adapter's pixel math is only needed for explicit size control.

4. **`gpt-image-2` accepts flexible resolutions** — not just the 3 preset sizes. Any `WxH` string meeting the constraints (multiples of 16, max 3840 edge, ratio ≤ 3:1, 655K-8.3M pixels) is valid.

5. **`gpt-image-2` always processes input images at high fidelity** — the `input_fidelity` parameter is ignored. This means edit requests with reference images use more input tokens.

6. **OpenAI SDK `images.edit()` requires `IO[bytes]`** — file paths must be `open(path, "rb")`.

7. **Model naming in our codebase:** We use `gpt-image-2` as an alias. The actual API model IDs are: `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, `gpt-image-1-mini`.

8. **Mask requirements for editing:** Image and mask must be same format and size (< 50MB). Mask must have an alpha channel (transparent areas = edit zones).

9. **Streaming:** Set `stream=True` + `partial_images=N` (0-3) to get progressive image updates. Each partial adds ~100 tokens to cost.

---

## 6. Common Bugs & How to Avoid Them

### Bug 1: Images in `data` instead of `images` (OpenAI bug, fixed 2025-04-27)

```python
# WRONG — downstream tool sees images=None, leaks base64 as "text"
return GeneratorOutput(data=base64_string, raw_response=str(response))

# CORRECT — images in images field, data for text only
return GeneratorOutput(data=None, images={"b64_json": b64}, raw_response="")
```

### Bug 2: Wrong image dict format

```python
# WRONG — plain string, _save_image_to_file can't handle it
images = [img.b64_json for img in response]

# CORRECT — dict with "b64_json" key
images = [{"b64_json": img.b64_json} for img in response]
```

### Bug 3: Storing base64 in raw_response

```python
# WRONG — megabytes of base64 in raw_response, leaks to logs/UI
raw_response=str(response)  # contains full base64 data

# CORRECT — empty or text-only summary
raw_response=""
```

### Bug 4: Not unwrapping single image from list

```python
# WRONG — always returns list, breaks isinstance(images_data, dict) check
images = [{"b64_json": b64}]

# CORRECT — unwrap single image
images = images[0] if len(images) == 1 else images
```

---

## 7. Testing Checklist for New Image Provider

When adding a new provider's image generation support:

- [ ] `convert_inputs_to_api_kwargs` handles `ModelType.IMAGE_GENERATION`
- [ ] `acall` routes `IMAGE_GENERATION` to the correct API
- [ ] Response parser returns `GeneratorOutput` with `images` field (not `data`)
- [ ] Image dicts use `{"b64_json": str}` or `{"url": str}` format
- [ ] `raw_response` does NOT contain base64 data
- [ ] Single image is unwrapped from list (dict, not [dict])
- [ ] Text-only params (`temperature`, `max_tokens`) are stripped
- [ ] Error responses set `GeneratorOutput.error`, not crash
- [ ] End-to-end test: tool saves image to disk, file is valid image
- [ ] End-to-end test: tool returns correct `ToolOutput.observation` (path, model name)
- [ ] End-to-end test: `ToolOutput.metadata.text` is None or actual text (never base64)

---

## 8. File Reference

| File | Role |
|------|------|
| `adalflow/core/types.py` | `GeneratorOutput` — unified output type |
| `adalflow/core/generator.py` | `Generator` — orchestrates pre_call → acall → post_call |
| `adalflow/components/model_client/google_client.py` | Reference implementation (correct) |
| `adalflow/components/model_client/openai_client.py` | OpenAI implementation (fixed 2025-04-27) |
| `tools/src/tools/image_generation.py` | `ImageGenerationTool` — save pipeline |
| `tools/src/tools/image_generation_adapters.py` | Adapters (to be eliminated) |
