#!/usr/bin/env python3
"""LTX-2 prompt generator - generates cinematographic video prompts optimized for LTX-2.

Aligned with the official LTX-2 Gemma prompt enhancer format:
- "Style: <style>," prefix, then single flowing paragraph
- Present-progressive verbs ("is walking", "is speaking")
- Audio/soundscape integrated chronologically alongside actions
- Speech in quotes with voice characteristics
- Restrained language, no dramatic/exaggerated terms
- ≤200 words, no headings/markdown/timestamps
"""
import argparse
import csv
import json
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

# ============== 硬编码配置 ==============
API_KEY = "sk-Zl0CgSxEPikuXrmGFcn2yV83IZ37ahGzHEQ7Ncs075zwZsL3"
API_BASE = "https://api.xcode.best/v1"
MODEL = "gpt-5.4"
# ========================================
OUTPUT_CSV = "ltx_prompts.csv"
OUTPUT_CATEGORIZED_CSV = OUTPUT_CSV.replace(".csv", "_categorized.csv")

TARGET_COUNT = 12000
WORKERS = 8
PROMPTS_PER_REQUEST = 4
PROMPTS_TO_KEEP_PER_REQUEST = 2
TEMPERATURE = 0.8
MAX_TOKENS = 4000
TIMEOUT = 120

# ============== LTX-2 分类体系 ==============
CATEGORIES = {
    "Human_CloseUp": {
        "desc": "Close-up or medium close-up of a single person - facial expressions, emotions, subtle gestures, intimate moments",
        "weight": 15,
    },
    "Human_Action": {
        "desc": "People performing activities - cooking, sports, crafts, work, walking, running, dancing",
        "weight": 20,
    },
    "Human_Dialogue": {
        "desc": "1-2 people talking, conversing, or presenting - natural speech with body language and gestures",
        "weight": 15,
    },
    "Cinematic_Scene": {
        "desc": "Cinematic narrative moments - dramatic scenes, story beats, character interactions with cinematic framing",
        "weight": 15,
    },
    "Nature_Landscape": {
        "desc": "Natural environments - weather, landscapes, water, sky, forests, mountains, ocean - with dynamic elements",
        "weight": 10,
    },
    "Animal": {
        "desc": "Animals in natural habitats - wildlife behavior, pets, nature documentary style",
        "weight": 5,
    },
    "Urban_Architecture": {
        "desc": "City scenes, streets, buildings, vehicles, urban life - architectural details and city dynamics",
        "weight": 5,
    },
    "Stylized": {
        "desc": "Stylized or artistic shots - film noir, period drama, sci-fi, fantasy aesthetics, specific film looks",
        "weight": 5,
    },
    "Object_Product": {
        "desc": "Object close-ups, product shots, food, still life with camera movement and lighting",
        "weight": 5,
    },
    "Music_Performance": {
        "desc": "Musical performances - singing, instrument playing, dancing to music, concert scenes",
        "weight": 5,
    },
}

# ============== LTX-2 风格示例 ==============
CATEGORY_EXAMPLES = {
    "Human_CloseUp": [
        'Style: cinematic-realistic, In a medium close-up at shoulder height, a woman in her mid-30s with short dark hair is standing at a kitchen counter as soft afternoon light is streaming through a nearby window. Her right hand is resting on the counter while her left hand is reaching up to brush a strand of hair from her forehead. The soft hum of a refrigerator and distant traffic sounds are filling the quiet space. Her eyebrows are lifting slightly and her lips are parting as if about to speak, while copper pots are hanging softly blurred in the background behind her.',
        'Style: warm intimate, An elderly man with deep-set wrinkles and silver stubble is sitting in a leather armchair, bathed in warm lamp light. Shot on an 85mm lens at f/2.0, the camera is holding steady in a tight close-up. His watery blue eyes are scanning an unseen page as a grandfather clock is ticking steadily nearby. Then his eyes are slowly closing and his head is tilting back slightly. His weathered hands are gripping the armrests, knuckles whitening briefly before relaxing as dust particles are drifting through the amber light beam.',
        'Style: naturalistic overcast, A young woman with braided hair and subtle freckles is leaning against a rain-streaked window, her breath is fogging the glass. The camera is sitting at eye level in a medium close-up, 50mm lens, as overcast light is casting soft shadows across her face. Rain is pattering against the window pane while distant thunder is rumbling outside. She is tracing a finger along the condensation, drawing an absent-minded circle, then turning to look directly into the lens as a faint smile is forming at the corner of her mouth.',
    ],
    "Human_Action": [
        'Style: cinematic-realistic, In a smooth tracking shot at waist height, a chef in a white coat is entering a busy restaurant kitchen, weaving between prep stations as steam is rising from simmering pots. The clatter of pans and the hiss of a hot griddle are filling the warm space. He is reaching his station, tying his apron with practiced hands, and beginning to chop carrots with rapid precise strokes. His knife is striking the wooden board in a steady rhythm while other kitchen staff are moving purposefully in the softly blurred background under warm overhead lights.',
        'Style: golden-hour cinematic, A skateboarder in a grey hoodie is pushing down an empty concrete plaza as his shadow is stretching long across the warm-toned pavement. Shot on a 35mm lens, the camera is tracking alongside at matching speed. The scrape of urethane wheels on rough concrete is filling the air. He is crouching low, then popping the tail with a sharp snap as the board is spinning beneath him in a clean kickflip before he is landing with a solid clack, arms extending briefly for balance as he rolls onward.',
        'Style: soft morning naturalistic, A woman in a yellow raincoat is kneeling in a lush garden at dawn, gently parting wet leaves to check ripening tomatoes on the vine. Birds are singing nearby as a thin mist is drifting between the garden rows. The camera is holding in a medium shot from slightly above, capturing morning dew glistening on her sleeves. She is plucking a red tomato, turning it in her fingers, then placing it carefully in a wicker basket beside her knee as early sunlight is filtering through a wooden trellis.',
    ],
    "Human_Dialogue": [
        'Style: cinematic-realistic, Two women are sitting across from each other at a sunlit sidewalk cafe, espresso cups between them on a small marble table. The camera is framing them in a medium two-shot at eye level, 50mm lens with shallow depth of field softening the busy street behind them. Ambient street noise is mingling with the clink of porcelain. The woman on the left, with curly auburn hair and a cream cardigan, is leaning forward and speaking in a warm, animated voice, "I think we should go for it." Her companion, wearing tortoiseshell glasses and a denim jacket, is nodding slowly, then responding with a soft laugh, "You always say that."',
        'Style: tense dramatic, A man in a tailored grey suit is standing face to face with a woman in a black dress in a dimly lit hallway. The camera is capturing them in a medium shot from a low angle as fluorescent lights are flickering overhead, casting cool blue tones across their faces. He is speaking in a firm, measured tone, "We need to talk about this now," his jaw tightening as he makes his point, hands clasped before him. She is uncrossing her arms and tilting her head slightly, responding quietly, "Not here." The tension between them is visible in their rigid postures.',
    ],
    "Cinematic_Scene": [
        'Style: sci-fi cinematic, A pair of metallic elevator doors are sliding open inside a spaceship corridor as thin mist is rolling out from floor vents. In a stationary wide shot, a tall figure in a white uniform is stepping forward through the haze. Blue accent lights are lining the corridor walls, casting geometric patterns on the polished floor. Their footsteps are echoing with a hollow metallic ring as a low mechanical hum is filling the space, punctuated by occasional electronic beeps. The camera is gliding sideways in a smooth tracking shot, following their stride past illuminated wall panels and sealed doorways.',
        'Style: film noir, A detective in a rumpled trench coat is pushing open a heavy wooden door into a dimly lit bar. The camera is following from behind in a steady dolly shot as cigarette smoke is curling through shafts of amber light from overhead fixtures. Glasses are clinking softly and low murmured conversation is creating a heavy atmosphere. He is pausing at the threshold, scanning the room where a lone pianist is playing a melancholic tune in the corner. The camera is continuing forward past his shoulder into a medium shot as he is approaching the bar counter, rain dripping from his coat onto the worn hardwood floor.',
    ],
    "Nature_Landscape": [
        'Style: cinematic-realistic, Massive ocean waves are crashing against a weathered rocky coastline at dawn, sending white spray high into the misty air. The camera is holding in a wide shot from an elevated position, capturing the full force of the surf against dark jagged boulders. Wind is howling across the scene as foam is rushing up the rocks and retreating in rhythmic pulses. Seabirds are wheeling overhead against a pale pink sky. The camera is slowly panning right, revealing a distant lighthouse perched on a cliff edge as thunder is rumbling from gathering storm clouds on the horizon.',
        'Style: moody naturalistic, A dense forest clearing is receiving a heavy downpour, rain is hammering the canopy and cascading through layers of green leaves. Shot in a medium-wide frame with a 35mm lens, the camera is remaining static as water is pooling across the muddy forest floor. The steady drumming of rain on leaves blends with the rush of a swollen stream passing moss-covered rocks in the foreground. Lightning is flashing briefly, illuminating the dark treeline before thunder is rolling across the landscape.',
    ],
    "Animal": [
        'Style: nature documentary, A red fox is trotting across a snow-covered meadow at twilight, its bushy tail trailing behind against the white landscape. The camera is tracking alongside in a smooth lateral movement, 85mm telephoto lens compressing the blue-tinged snowy background. Crunching footsteps and distant wind are carrying across the quiet field. The fox is pausing, ears swiveling forward, then pouncing nose-first into the snow, emerging moments later with a small rodent in its jaws as snow crystals are scattering from its whiskers. Its amber eyes are glinting in the fading light.',
        'Style: macro naturalistic, A hummingbird is hovering before a cluster of red trumpet flowers, its iridescent green plumage is shimmering as wings are beating in a rapid blur. The camera is holding in an extreme close-up with shallow depth of field isolating the tiny bird against a soft bokeh of garden colors. The rapid whirring of wings is blending with the lazy buzz of nearby bees. Its long curved beak is dipping into a blossom, withdrawing, then the bird is pivoting mid-air to approach the next flower as sunlight is catching its ruby throat patch.',
    ],
    "Urban_Architecture": [
        'Style: cinematic urban, Early morning fog is rolling through the steel and glass canyon of a downtown financial district as the first commuters are emerging from a subway entrance. The camera is beginning in a low-angle wide shot looking up between two skyscrapers, then tilting down and pushing forward into the street-level crowd. Car horns are punctuating the steady hum of the waking city. Office lights are flickering on in windows high above while a yellow taxi is idling at a red light, exhaust rising in the cold air. The wet sidewalk is reflecting signage and traffic signals in blurred streaks of color.',
    ],
    "Stylized": [
        'Style: 1940s period noir with warm tungsten tones, A woman in a red dress is descending a spiral staircase in a grand art deco hotel lobby, one gloved hand trailing along the brass banister. The camera is capturing her from below in a slow upward crane shot, 35mm anamorphic lens. Her heels are clicking rhythmically on marble steps as warm light is bathing the scene in amber and deep shadows. She is pausing halfway down, looking over her shoulder with a measured expression as cigarette smoke is curling from her other hand. A jazz trio is playing softly from an unseen lounge, bass notes reverberating through the cavernous space.',
    ],
    "Object_Product": [
        'Style: warm minimalist, A ceramic coffee cup is sitting on a rough wooden table as morning sunlight is creeping across the surface, steam rising in delicate spirals from the dark liquid. The camera is beginning in a tight close-up on the glazed rim, 100mm macro lens at f/2.8, then slowly pulling back to reveal the full table setting. The soft gurgle of a coffee maker is audible in the background. A hand is entering from the right side of the frame, wrapping around the warm cup, and lifting it from the saucer with a soft clink. Cream-colored linen and a folded newspaper are occupying the background in soft focus.',
    ],
    "Music_Performance": [
        'Style: warm jazz club, A jazz pianist is sitting at a glossy black grand piano on a dimly lit club stage, his fingers are rolling through a warm ascending arpeggio. The camera is starting in a close-up on his weathered hands dancing across the ivory keys, 50mm lens at f/1.8, then slowly pulling back to a medium shot revealing his closed eyes and gently swaying torso. Each note is ringing with natural reverb as his foot is working the sustain pedal. Amber stage light from above is creating a pool of warmth around the instrument while a single spotlight is catching wisps of smoke drifting above the piano.',
        'Style: intimate concert, A young woman with flowing black hair is standing center stage gripping a vintage microphone, her eyes are closed and her body is swaying. The camera is holding in a medium close-up as soft backlighting is creating a halo around her silhouette while a warm amber spot is lighting the front. She is leaning into a sustained note, her voice filling the small venue with rich tones over a gentle piano accompaniment. Then she is pulling back and opening her eyes as the audience beyond is remaining in shadow, their silhouettes barely visible.',
    ],
}

# ============== 多样性变量池 ==============
DIVERSITY_POOLS = {
    "time_of_day": ["early morning", "mid-morning", "noon", "afternoon", "golden hour", "dusk", "evening", "night", "midnight", "dawn", "twilight", "blue hour"],
    "weather": ["sunny", "overcast", "rainy", "foggy", "snowy", "windy", "stormy", "humid", "crisp autumn", "hot summer", "misty", "hazy"],
    "lighting": ["natural sunlight", "warm golden hour", "harsh midday sun", "soft diffused light", "dramatic shadows", "neon lights", "candlelight", "moonlight", "fluorescent", "dim ambient", "backlighting", "rim light"],
    "mood": ["energetic", "calm", "tense", "joyful", "melancholic", "mysterious", "romantic", "urgent", "contemplative", "playful", "suspenseful", "serene"],
    "camera_movement": ["static tripod", "slow dolly forward", "smooth tracking left", "gentle crane up", "slow pull-back", "orbit around subject", "steady push-in", "lateral dolly", "tilt up reveal", "handheld subtle sway"],
    "lens": ["24mm wide-angle", "35mm standard", "50mm normal", "85mm portrait", "100mm macro", "200mm telephoto", "anamorphic widescreen", "fisheye extreme"],
    "film_look": ["Kodak Portra warm tones", "Fuji Velvia saturated", "desaturated documentary", "high-contrast noir", "pastel soft grade", "teal and orange blockbuster", "natural ungraded", "vintage 8mm grain"],
    "setting_style": ["modern", "vintage", "rustic", "industrial", "minimalist", "cluttered", "luxurious", "worn-down", "futuristic", "traditional"],
}

CATEGORY_DIVERSITY_DIMS = {
    "Human_CloseUp": ["time_of_day", "lighting", "mood", "lens", "film_look"],
    "Human_Action": ["time_of_day", "weather", "lighting", "camera_movement", "lens"],
    "Human_Dialogue": ["time_of_day", "lighting", "mood", "setting_style", "lens"],
    "Cinematic_Scene": ["time_of_day", "lighting", "mood", "camera_movement", "film_look", "lens"],
    "Nature_Landscape": ["time_of_day", "weather", "lighting", "camera_movement", "lens"],
    "Animal": ["time_of_day", "weather", "lighting", "lens", "camera_movement"],
    "Urban_Architecture": ["time_of_day", "weather", "lighting", "camera_movement", "film_look"],
    "Stylized": ["lighting", "mood", "film_look", "lens", "camera_movement"],
    "Object_Product": ["time_of_day", "lighting", "lens", "camera_movement", "film_look"],
    "Music_Performance": ["lighting", "mood", "lens", "camera_movement", "setting_style"],
}


def get_diversity_hint(category: str) -> str:
    available_dims = CATEGORY_DIVERSITY_DIMS.get(category, list(DIVERSITY_POOLS.keys()))
    num_dims = min(random.randint(2, 3), len(available_dims))
    dims = random.sample(available_dims, k=num_dims)
    hints = []
    for dim in dims:
        value = random.choice(DIVERSITY_POOLS[dim])
        hints.append(f"{dim.replace('_', ' ')}: {value}")
    return "Creative suggestions (for inspiration only, feel free to deviate): " + ", ".join(hints)


# ============== LTX-2 系统提示词模板 ==============

LTX2_BASE_SYSTEM = """You are a creative prompt writer for the LTX-2 video generation model. Your output MUST match the exact format of LTX-2's official Gemma prompt enhancer.

## FORMAT RULES (STRICT — match the official enhancer output):

1. **Style prefix**: ALWAYS start with "Style: <visual_style>, " then continue with the scene description.
   - Examples: "Style: cinematic-realistic, ", "Style: film noir with high contrast, ", "Style: warm documentary, "
   - If the style is standard cinematic, use "Style: cinematic-realistic, "

2. **Single flowing paragraph**: Everything after the Style prefix is ONE continuous paragraph. No bullet points, no line breaks, no lists, no headings.

3. **200 words maximum**: Keep each prompt under 200 words total (including Style prefix).

4. **Present-progressive verbs**: Use "is walking", "is speaking", "is reaching" — NOT "walks" or "walked".
   - Example: "A woman is sitting at a table, her fingers are tracing the rim of a cup"

5. **Chronological flow**: Use temporal connectors: "as", "then", "while", "meanwhile".

6. **Audio integrated chronologically**: Weave soundscape descriptions INTO the action flow — do NOT put audio at the end as a separate section.
   - GOOD: "She is tapping the table rhythmically, the soft thuds blending with the hum of the espresso machine behind her"
   - BAD: "She taps the table. Audio: tapping sounds, espresso machine."

7. **Speech format** (only when dialogue is needed): Use quotes with voice characteristics.
   - Example: The man speaks in a warm, gravelly voice, "I've been waiting for you."

8. **Restrained language**: Use mild, natural, understated phrasing.
   - Colors: Plain terms ("red dress", "blue sky") — NOT "vibrant crimson", "brilliant azure"
   - Lighting: Neutral descriptions ("soft overhead light", "warm side light") — NOT "blinding", "dazzling"
   - Features: Delicate modifiers ("subtle freckles", "slight smile")

9. **Show, don't tell emotions**: Describe observable physical cues, not emotional labels.

10. **Camera language**: Include specific shot type and lens when applicable. Do NOT invent camera movement unless it serves the scene.

## DO NOT:
- Start with "The scene opens with..." or "The video starts..." — start directly with "Style:"
- Use readable text, brand names, logos, signage in the scene
- Describe complex physics (juggling, collisions, liquid simulations)
- Overload the scene (too many characters/actions/objects)
- Use conflicting light sources
- Use timestamps or describe scene cuts
- Use dramatic/exaggerated language
- Start output with punctuation or special characters after "Style:"
- Put audio/sound descriptions at the end — weave them in chronologically
"""

SYS_RENDER_HUMAN_CLOSEUP = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: facial detail, emotional subtlety, camera precision, lighting quality, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: HUMAN CLOSE-UP
Generate {count} prompts for intimate close-up or medium close-up shots of a SINGLE person.
Focus on: subtle facial expressions, micro-gestures, emotional transitions shown through physical cues.
Camera should be close — emphasize shallow depth of field, precise lens choices (85mm, 50mm).
Weave ambient sounds chronologically into the action — do NOT separate them.

## REFERENCE EXAMPLE (match this format EXACTLY — note the "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is sitting", "is reaching", "is looking"
- Exactly 1 person
- Close-up or medium close-up framing
- Audio/sounds woven into the description, not at the end
- REALISTIC settings only

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_HUMAN_ACTION = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: action fluidity, movement precision, camera work, environmental detail, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: HUMAN ACTION
Generate {count} prompts for scenes of people performing activities or actions.
Focus on: fluid natural movements, specific physical details of the activity, cause-and-effect sequences.
Camera should track or follow the action — use dolly, tracking shots, or steady medium shots.
Describe the full person (face, body, expression) — not just hands.

## REFERENCE EXAMPLE (match this format EXACTLY — note the "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is chopping", "is running", "is reaching"
- 1-2 people performing a clear activity
- Activity-specific sounds woven chronologically into the description
- REALISTIC settings only
- Vary activities: sports, cooking, crafts, work, walking, running, gardening, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_HUMAN_DIALOGUE = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: character distinction, body language, spatial relationship, dialogue integration, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: HUMAN DIALOGUE
Generate {count} prompts for 1-2 people conversing, presenting, or speaking.
Focus on: natural speech delivery, body language, gestures, facial reactions during conversation.
Use dialogue in quotation marks WITH voice characteristics (e.g., speaks in a warm tone, "I've been waiting for you").
Show emotion through physical cues — posture shifts, hand gestures, eye movements.

## REFERENCE EXAMPLE (match this format EXACTLY — note the "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is speaking", "is nodding", "is leaning"
- 1-2 people speaking
- Dialogue in quotation marks with voice characteristics (keep short — under 15 words total dialogue)
- Ambient sounds woven in chronologically
- REALISTIC settings only

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_CINEMATIC = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: narrative tension, cinematic framing, lighting drama, camera choreography, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: CINEMATIC SCENE
Generate {count} prompts for dramatic, narrative-driven cinematic moments.
Think like a film director — these are story beats with tension, atmosphere, and purpose.
Use genre-appropriate language (noir, thriller, drama, sci-fi, period piece).
Camera choreography should be deliberate — match camera movement to emotional rhythm.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, " (e.g., "Style: film noir, ", "Style: sci-fi cinematic, ")
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is pushing", "is stepping", "is scanning"
- Ambient sounds woven chronologically into the action
- Can include brief dialogue in quotes with voice characteristics (under 10 words)
- Vary genres: noir, thriller, drama, romance, sci-fi, western, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_NATURE = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: environmental dynamism, visual drama, sound design, camera work, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: NATURE & LANDSCAPE
Generate {count} prompts for dynamic natural scenes with visible motion and atmospheric energy.
Focus on: weather dynamics, water movement, light changes, wind effects — scenes with ENERGY.
Camera should capture the scale and power of nature — wide shots, slow pans, dramatic framing.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is crashing", "is rolling", "is streaming"
- NO humans as the main subject
- Must have visible dynamic motion (waves, rain, wind, clouds, etc.)
- Natural sounds woven chronologically into the action
- REALISTIC only
- Vary: ocean, mountains, forests, deserts, prairies, rivers, storms, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_ANIMAL = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: animal behavior accuracy, visual dynamics, camera precision, sound design, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: ANIMAL
Generate {count} prompts for animal scenes — wildlife, pets, nature documentary style.
Focus on: natural behavior, fluid animal movement, specific physical details of the animal.
Camera work should match wildlife cinematography — telephoto lens, patient tracking, revealing moments.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is trotting", "is hovering", "is pouncing"
- Animals as main subject (no humans in focus)
- Realistic animal behavior — no anthropomorphizing
- Animal sounds and environmental audio woven chronologically
- Vary species: mammals, birds, marine life, reptiles, insects, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_URBAN = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: urban atmosphere, architectural detail, city dynamics, camera work, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: URBAN & ARCHITECTURE
Generate {count} prompts for city scenes, streets, buildings, vehicles, and urban life.
Focus on: architectural details, city dynamics, reflections, traffic, pedestrian flow, signage atmosphere.
Camera work should emphasize scale, geometry, and the rhythm of city life.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is rolling", "are emerging", "is idling"
- Urban/city environment focus
- Dynamic elements (traffic, people moving, lights changing, weather effects)
- City soundscape woven chronologically into the description
- REALISTIC settings only
- Vary: downtown, suburbs, alleys, bridges, markets, transit, parks, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_STYLIZED = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: stylistic coherence, visual distinctiveness, genre authenticity, camera artistry, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: STYLIZED / ARTISTIC
Generate {count} prompts with strong visual style — film noir, period drama, sci-fi, cyberpunk, surreal, painterly.
Focus on: genre-specific visual language, distinctive color grading, artistic lighting, unique atmospheres.
Use film-specific terminology: anamorphic, Kodak film stock, high-contrast, desaturated, etc.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <specific_style>, " (e.g., "Style: 1940s period noir, ", "Style: cyberpunk neon, ")
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is descending", "is clicking", "is curling"
- Strong visual style / genre identity
- Specific film or art-style references encouraged
- Sound design woven chronologically into the action
- Vary styles: noir, retro, cyberpunk, period, fantasy, anime-inspired, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_OBJECT = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: detail precision, lighting beauty, camera movement elegance, texture quality, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: OBJECT / PRODUCT
Generate {count} prompts for object close-ups, product shots, food, still life with elegant camera movement.
Focus on: surface textures, material quality, precise lighting, macro details, reflections.
Camera work: slow reveals, macro push-ins, orbit shots, pull-backs revealing context.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is sitting", "is creeping", "is rising"
- Object as the main subject
- Precise material/texture descriptions
- Elegant camera movement (slow, deliberate)
- Subtle ambient sounds woven chronologically into the description
- Vary: food, beverages, crafts, technology, jewelry, instruments, tools, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_MUSIC = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: performance authenticity, music-visual sync, atmosphere, camera work, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: MUSIC PERFORMANCE
Generate {count} prompts for musical performances — singing, instrument playing, dancing to music, concerts.
Focus on: the physical act of making music, body rhythm, instrument details, stage/venue atmosphere.
Camera should capture the energy and emotion of live performance.
LTX-2 generates synchronized audio-video, so describe the music and sounds prominently.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, " (e.g., "Style: warm jazz club, ", "Style: intimate concert, ")
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs: "is sitting", "is rolling", "is swaying"
- Music/performance as the main focus
- Describe the music woven into the action (genre, mood, rhythm)
- Can include brief sung lyrics in quotes (under 10 words)
- Vary: jazz, rock, classical, hip-hop, folk, electronic, etc.

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

SYS_RENDER_GENERIC = LTX2_BASE_SYSTEM + """
## SELECTION PROCESS:
1. Silently brainstorm {draft_count} diverse candidates (do NOT output these)
2. Evaluate on: creativity, visual richness, camera precision, atmosphere, uniqueness
3. Output ONLY the TOP {count} highest-quality prompts

## CATEGORY: {category}

Generate {count} UNIQUE, highly detailed prompts following the LTX-2 style.
Be wildly creative — each prompt should feel like a unique cinematic moment.

## REFERENCE EXAMPLE (match this format EXACTLY — note "Style:" prefix and present-progressive verbs):
{example}

## CONSTRAINTS:
- MUST start with "Style: <style>, "
- Each prompt: single flowing paragraph, under 200 words
- Use present-progressive verbs throughout: "is walking", "is reaching"
- Include specific camera and lens language
- Audio/sounds woven chronologically into the action
- REALISTIC or genre-appropriate settings

## OUTPUT:
Strict JSON: {{"prompts": ["prompt1", "prompt2", ...]}}
Return exactly {count} prompts."""

# 类别到模板的映射
CATEGORY_TEMPLATES = {
    "Human_CloseUp": SYS_RENDER_HUMAN_CLOSEUP,
    "Human_Action": SYS_RENDER_HUMAN_ACTION,
    "Human_Dialogue": SYS_RENDER_HUMAN_DIALOGUE,
    "Cinematic_Scene": SYS_RENDER_CINEMATIC,
    "Nature_Landscape": SYS_RENDER_NATURE,
    "Animal": SYS_RENDER_ANIMAL,
    "Urban_Architecture": SYS_RENDER_URBAN,
    "Stylized": SYS_RENDER_STYLIZED,
    "Object_Product": SYS_RENDER_OBJECT,
    "Music_Performance": SYS_RENDER_MUSIC,
}

# 某些类别需要更多候选
CATEGORY_DRAFT_COUNT = {
    "Nature_Landscape": 6,
    "Animal": 5,
    "Stylized": 5,
}


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        depth, end = 0, start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from: {text[:200]}...")


def api_call(system: str, user: str, max_tokens: int = None) -> Dict[str, Any]:
    base = API_BASE.rstrip("/")
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    tokens = max_tokens if max_tokens else MAX_TOKENS

    for attempt in range(5):
        seed = random.randint(0, 2**31 - 1)
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": TEMPERATURE,
                    "max_tokens": tokens,
                    "seed": seed,
                },
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return extract_json(text)
        except Exception as e:
            print(f"  Retry {attempt+1}/5: {e}")
            time.sleep(2**attempt)
    raise RuntimeError("API failed after retries")


def validate_prompt_with_reason(prompt: str) -> Tuple[bool, str]:
    """Validate a prompt follows LTX-2 Gemma enhancer format rules."""
    if not prompt:
        return False, "empty"

    # 字符数检查 (200 words ≈ 800-1400 chars, 允许一些余量)
    if len(prompt) < 150:
        return False, f"too_short:{len(prompt)}_chars"
    if len(prompt) > 2000:
        return False, f"too_long:{len(prompt)}_chars"

    # 词数检查: ≤ 200 words (允许少量超出到 220)
    words = prompt.split()
    if len(words) > 220:
        return False, f"word_count_over_200:{len(words)}"
    if len(words) < 30:
        return False, f"word_count_too_low:{len(words)}"

    # 必须以 "Style:" 开头
    if not prompt.startswith("Style:"):
        return False, "missing_style_prefix"

    # 必须是单段落（不能有多个换行）
    if "\n\n" in prompt:
        return False, "multiple_paragraphs"

    # 不能有列表格式
    if re.search(r"^[\s]*[-*•]\s", prompt, re.MULTILINE):
        return False, "contains_bullet_points"

    # 不应包含 causAV 风格的标签
    if "<S>" in prompt or "<E>" in prompt:
        return False, "contains_causav_speech_tags"

    # 不应有独立的 "Audio:" 段落标记
    if re.search(r"\bAudio:\s", prompt):
        return False, "contains_audio_section_marker"

    # 不应以 "The scene opens" / "The video starts" 开头（在 Style: 之后）
    style_body = prompt.split(",", 1)[1].strip() if "," in prompt else ""
    if re.match(r"^(The scene opens|The video starts|The video opens)", style_body, re.IGNORECASE):
        return False, "starts_with_scene_opens"

    # 不应大量使用过去时态（was/were/walked 等）
    past_tense_count = len(re.findall(
        r"\b(was|were|had been|walked|turned|looked|stood|sat|said|spoke|moved|entered|paused)\b",
        prompt, re.IGNORECASE
    ))
    if past_tense_count > 3:
        return False, f"too_many_past_tense:{past_tense_count}"

    # 鼓励但不强制：检查是否有 present-progressive 用法 (is/are + -ing)
    progressive_count = len(re.findall(r"\b(is|are)\s+\w+ing\b", prompt, re.IGNORECASE))
    if progressive_count < 2:
        return False, f"too_few_progressive_verbs:{progressive_count}"

    return True, "ok"


def validate_prompt(prompt: str) -> bool:
    ok, _ = validate_prompt_with_reason(prompt)
    return ok


def shuffle_csv(csv_path: str, categorized_path: str):
    """Shuffle both output CSVs in place after generation completes."""
    for path, has_category in [(csv_path, False), (categorized_path, True)]:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        random.shuffle(rows)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Shuffled {len(rows)} rows in {path}")


def load_existing_prompts() -> Set[str]:
    if not os.path.exists(OUTPUT_CSV):
        return set()
    prompts = set()
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                prompts.add(row[0])
    return prompts


def load_existing_category_counts() -> Dict[str, int]:
    counts = {}
    if not os.path.exists(OUTPUT_CATEGORIZED_CSV):
        return counts
    with open(OUTPUT_CATEGORIZED_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                cat = row[0]
                counts[cat] = counts.get(cat, 0) + 1
    return counts


def save_prompts(prompts: List[str]):
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["text_prompt"])
        for p in prompts:
            writer.writerow([p])


def save_prompts_categorized(prompts: List[str], category: str):
    file_exists = os.path.exists(OUTPUT_CATEGORIZED_CSV)
    with open(OUTPUT_CATEGORIZED_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["category", "text_prompt"])
        for p in prompts:
            writer.writerow([category, p])


def calculate_category_counts(total_count: int) -> Dict[str, int]:
    total_weight = sum(c["weight"] for c in CATEGORIES.values())
    counts = {}
    remaining = total_count
    category_list = list(CATEGORIES.keys())
    for i, name in enumerate(category_list):
        if i == len(category_list) - 1:
            counts[name] = remaining
        else:
            weight = CATEGORIES[name]["weight"]
            count = round(total_count * weight / total_weight)
            counts[name] = count
            remaining -= count
    return counts


def _get_category_prompt(l1_name: str, draft_count: int, example: str) -> str:
    template = CATEGORY_TEMPLATES.get(l1_name)
    if template:
        return template.format(
            draft_count=draft_count,
            count=PROMPTS_TO_KEEP_PER_REQUEST,
            example=example,
        )
    # Fallback to generic
    cat = CATEGORIES[l1_name]
    return SYS_RENDER_GENERIC.format(
        draft_count=draft_count,
        count=PROMPTS_TO_KEEP_PER_REQUEST,
        category=f"{l1_name} - {cat['desc']}",
        example=example,
    )


def generate_prompts_by_category(target_count: int = None):
    """依次按类别生成 prompts，每个类别内部多线程并行"""
    if target_count is None:
        target_count = TARGET_COUNT

    existing = load_existing_prompts()
    existing_cat_counts = load_existing_category_counts()

    category_counts = calculate_category_counts(target_count)
    category_stats = {name: 0 for name in CATEGORIES.keys()}
    total_produced = 0
    reject_stats = {}  # 记录各类拒绝原因

    # 总进度条
    pbar_total = tqdm(
        total=target_count,
        desc="Total",
        unit="prompt",
        position=0,
        bar_format="{l_bar}{bar:30}{r_bar}",
        colour="green",
    )
    pbar_total.update(len(existing))
    pbar_total.set_postfix(existing=len(existing))

    if len(existing) >= target_count:
        print("Target reached from existing prompts")
        pbar_total.close()
        return

    for l1_name, target_for_category in category_counts.items():
        existing_for_cat = existing_cat_counts.get(l1_name, 0)
        target_for_category -= existing_for_cat
        if target_for_category <= 0:
            print(f"Skipping {l1_name}, already has {existing_for_cat} prompts (target was {category_counts[l1_name]})")
            continue

        draft_count = CATEGORY_DRAFT_COUNT.get(l1_name, PROMPTS_PER_REQUEST)
        lock = threading.Lock()
        produced_lock = [0]

        # 每个类别的子进度条
        pbar_cat = tqdm(
            total=target_for_category,
            desc=f"  {l1_name}",
            unit="prompt",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar:25}{r_bar}",
            colour="cyan",
        )

        def make_single_request(cat_name=l1_name, cat_draft_count=draft_count):
            examples = CATEGORY_EXAMPLES.get(cat_name, CATEGORY_EXAMPLES["Human_CloseUp"])
            example = random.choice(examples)
            prompt = _get_category_prompt(cat_name, cat_draft_count, example)
            diversity_hint = get_diversity_hint(cat_name)
            user_msg = f"{diversity_hint}\n\nGenerate prompts now. Be creative and use the suggestions above as optional inspiration."

            try:
                result = api_call(prompt, user_msg)
                raw_prompts = result.get("prompts", [])
                valid = []
                with lock:
                    for p in raw_prompts:
                        ok, reason = validate_prompt_with_reason(p)
                        if ok and p not in existing:
                            valid.append(p)
                            existing.add(p)
                        else:
                            # 统计拒绝原因
                            base_reason = reason.split(":")[0] if ":" in reason else reason
                            reject_stats[base_reason] = reject_stats.get(base_reason, 0) + 1
                return valid
            except Exception as e:
                reject_stats["api_error"] = reject_stats.get("api_error", 0) + 1
                return []

        def process_batch(cat_name=l1_name, cat_target=target_for_category):
            with lock:
                if produced_lock[0] >= cat_target:
                    return 0
            valid = make_single_request()
            if not valid:
                return 0
            with lock:
                if produced_lock[0] >= cat_target:
                    return 0
                need = cat_target - produced_lock[0]
                to_save = valid[:need]
                save_prompts(to_save)
                save_prompts_categorized(to_save, cat_name)
                produced_lock[0] += len(to_save)
                pbar_cat.update(len(to_save))
                pbar_total.update(len(to_save))
                pbar_total.set_postfix(cat=cat_name, last=f"+{len(to_save)}")
                return len(to_save)

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            while produced_lock[0] < target_for_category:
                remaining = target_for_category - produced_lock[0]
                batches_needed = (remaining + PROMPTS_TO_KEEP_PER_REQUEST - 1) // PROMPTS_TO_KEEP_PER_REQUEST
                num_workers = min(WORKERS, batches_needed)
                if num_workers <= 0:
                    break
                futures = [executor.submit(process_batch) for _ in range(num_workers)]
                for fut in as_completed(futures):
                    total_produced += fut.result()

        category_stats[l1_name] = produced_lock[0]
        pbar_cat.close()

    pbar_total.close()

    # Shuffle 输出文件
    shuffle_csv(OUTPUT_CSV, OUTPUT_CATEGORIZED_CSV)

    # 最终统计
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total generated: {total_produced}")
    print("\nCategory breakdown:")
    for name, count in category_stats.items():
        if count > 0:
            print(f"  {name}: {count}")
    if reject_stats:
        print("\nReject reasons:")
        for reason, count in sorted(reject_stats.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


def generate_prompts(target_category: Optional[str] = None, target_count: int = None):
    """Random sampling mode - weighted random category selection."""
    if target_count is None:
        target_count = TARGET_COUNT

    existing = load_existing_prompts()

    if len(existing) >= target_count:
        print("Target reached")
        return

    produced = 0
    lock = threading.Lock()
    batch_num = [0]
    saved_count = [len(existing)]
    reject_stats = {}

    if target_category:
        if target_category not in CATEGORIES:
            print(f"Error: Unknown category '{target_category}'")
            print(f"Available categories: {list(CATEGORIES.keys())}")
            return
        category_names = [target_category]
        category_weights = [1.0]
    else:
        category_names = list(CATEGORIES.keys())
        category_weights = [CATEGORIES[name]["weight"] for name in category_names]

    category_stats = {name: 0 for name in CATEGORIES.keys()}

    # 进度条
    pbar = tqdm(
        total=target_count,
        initial=len(existing),
        desc="Generating",
        unit="prompt",
        bar_format="{l_bar}{bar:30}{r_bar}",
        colour="green",
    )

    def make_request():
        l1_name = random.choices(category_names, weights=category_weights, k=1)[0]
        draft_count = CATEGORY_DRAFT_COUNT.get(l1_name, PROMPTS_PER_REQUEST)
        examples = CATEGORY_EXAMPLES.get(l1_name, CATEGORY_EXAMPLES["Human_CloseUp"])
        example = random.choice(examples)
        prompt = _get_category_prompt(l1_name, draft_count, example)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                diversity_hint = get_diversity_hint(l1_name)
                user_msg = f"{diversity_hint}\n\nGenerate prompts now. Be creative and use the suggestions above as optional inspiration."
                result = api_call(prompt, user_msg)
                raw_prompts = result.get("prompts", [])
                valid = []
                with lock:
                    for p in raw_prompts:
                        ok, reason = validate_prompt_with_reason(p)
                        if ok and p not in existing:
                            valid.append(p)
                            existing.add(p)
                        else:
                            base_reason = reason.split(":")[0] if ":" in reason else reason
                            reject_stats[base_reason] = reject_stats.get(base_reason, 0) + 1
                if valid:
                    return valid[:PROMPTS_TO_KEEP_PER_REQUEST], l1_name
                elif attempt < max_retries - 1:
                    pass  # 静默重试，不刷屏
            except Exception as e:
                reject_stats["api_error"] = reject_stats.get("api_error", 0) + 1
                if attempt < max_retries - 1:
                    pass
        return [], l1_name

    def process_batch():
        nonlocal produced
        with lock:
            if saved_count[0] >= target_count:
                return 0
            batch_num[0] += 1

        prompts, l1_name = make_request()

        with lock:
            if not prompts or saved_count[0] >= target_count:
                return 0
            need = target_count - saved_count[0]
            to_save = prompts[:need]
            save_prompts(to_save)
            save_prompts_categorized(to_save, l1_name)
            saved_count[0] += len(to_save)
            category_stats[l1_name] += len(to_save)
            pbar.update(len(to_save))
            pbar.set_postfix(cat=l1_name, last=f"+{len(to_save)}")
            return len(to_save)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        while saved_count[0] < target_count:
            remaining = target_count - saved_count[0]
            avg_yield = PROMPTS_TO_KEEP_PER_REQUEST
            batches_needed = (remaining + avg_yield - 1) // avg_yield
            num_workers = min(WORKERS, batches_needed)
            if num_workers <= 0:
                break
            futures = [ex.submit(process_batch) for _ in range(num_workers)]
            for fut in as_completed(futures):
                produced += fut.result()

    pbar.close()

    # Shuffle 输出文件
    shuffle_csv(OUTPUT_CSV, OUTPUT_CATEGORIZED_CSV)

    # 最终统计
    print(f"\nDone. New: {produced}, Total: {len(existing)}")
    print("\n=== Category Statistics ===")
    for cat_name in category_names:
        count = category_stats.get(cat_name, 0)
        if count > 0:
            print(f"  {cat_name}: {count}")
    if reject_stats:
        print("\nReject reasons:")
        for reason, count in sorted(reject_stats.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LTX-2 Prompt Generator - Generate cinematographic video prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available categories:
  {', '.join(CATEGORIES.keys())}

Examples:
  python gen_prompts.py                         # 按类别轮流生成（默认）
  python gen_prompts.py --random                # 随机采样模式
  python gen_prompts.py -c Animal               # 只生成 Animal 类
  python gen_prompts.py -n 200                  # 生成200条
  python gen_prompts.py -o my_ltx_prompts       # 输出到 my_ltx_prompts.csv
""",
    )
    parser.add_argument("-c", "--category", type=str, default=None, help="指定生成的类别")
    parser.add_argument("--random", action="store_true", help="使用随机采样模式")
    parser.add_argument("-n", "--count", type=int, default=None, help=f"目标生成数量（默认: {TARGET_COUNT}）")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出文件名（不含扩展名）")
    parser.add_argument("--shuffle-only", action="store_true", help="只对已有 CSV 做 shuffle，不生成新 prompt")
    return parser.parse_args()


def main():
    args = parse_args()

    global OUTPUT_CSV, OUTPUT_CATEGORIZED_CSV

    target_count = args.count if args.count else TARGET_COUNT
    if args.output:
        output_base = args.output
    else:
        output_base = f"ltx_prompts_{target_count}"

    OUTPUT_CSV = f"{output_base}.csv"
    OUTPUT_CATEGORIZED_CSV = f"{output_base}_categorized.csv"

    if args.shuffle_only:
        shuffle_csv(OUTPUT_CSV, OUTPUT_CATEGORIZED_CSV)
        return 0

    if API_KEY == "REPLACE_WITH_YOUR_API_KEY" or API_BASE == "https://your-api-base-url":
        print("Please edit API_KEY and API_BASE at the top of the file!")
        return 1

    print(f"=== LTX-2 Prompt Generator ===")
    print(f"Target: {target_count} prompts -> {OUTPUT_CSV}")
    print(f"Categorized output -> {OUTPUT_CATEGORIZED_CSV}")
    total_weight = sum(c["weight"] for c in CATEGORIES.values())

    if args.random:
        print(f"Mode: RANDOM SAMPLING")
        print(f"Categories: {len(CATEGORIES)}")
        for name, cat in CATEGORIES.items():
            prob = cat["weight"] / total_weight * 100
            print(f"  - {name}: {prob:.1f}%")
        print()
        generate_prompts(target_category=args.category, target_count=target_count)
    else:
        print(f"Mode: BY-CATEGORY ({WORKERS} workers per category)")
        print(f"Categories: {len(CATEGORIES)}")
        category_counts = calculate_category_counts(target_count)
        for name, count in category_counts.items():
            print(f"  - {name}: {count} prompts")
        print()
        generate_prompts_by_category(target_count=target_count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
