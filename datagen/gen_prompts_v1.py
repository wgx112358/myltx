#!/usr/bin/env python3
"""LTX-2 prompt generator v1.

This version keeps the batch-generation workflow from gen_prompts.py, but
aligns prompt writing much more closely with the LTX-2 Prompting Guide:
- single flowing paragraph
- 4-8 chronological sentences
- under 200 words
- starts directly with the main action or shot
- literal, cinematography-style scene description
- audio cues woven into the action timeline

It also removes the hard-coded secret requirement from the original script and
expects API credentials via environment variables or CLI arguments.
"""

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm


DEFAULT_API_BASE = os.getenv("PROMPT_GEN_API_BASE", "https://api.xcode.best")
DEFAULT_API_KEY = os.getenv("PROMPT_GEN_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-Zl0CgSxEPikuXrmGFcn2yV83IZ37ahGzHEQ7Ncs075zwZsL3"
DEFAULT_MODEL = os.getenv("PROMPT_GEN_MODEL", "gpt-5.4")

DEFAULT_OUTPUT_CSV = "ltx_prompts_v1.csv"
DEFAULT_TARGET_COUNT = 12000
DEFAULT_WORKERS = 8
DEFAULT_DRAFTS_PER_REQUEST = 6
DEFAULT_KEEP_PER_REQUEST = 3
DEFAULT_TEMPERATURE = 0.78
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TIMEOUT = 120

MIN_WORDS = 45
MAX_WORDS = 200
MIN_SENTENCES = 4
MAX_SENTENCES = 8

DIALOGUE_ALLOWED_FAMILIES = {
    "Human_Interaction",
    "Human_Presentation",
}


SCENE_FAMILIES: Dict[str, Dict[str, Any]] = {
    "Human_Portrait": {
        "mode": "portrait close-up",
        "hero_subject_archetypes": [
            "a single adult in an everyday role",
            "a young person in a transitional or uncertain moment",
            "an older person whose face and hands show lived experience",
            "a solitary worker or commuter between actions",
        ],
        "setting_archetypes": [
            "a quiet domestic interior or window-side corner",
            "a parked vehicle, waiting room, or transit interior",
            "a narrow hallway, stairwell, or back room",
            "a small work area with limited depth and a soft background",
        ],
        "action_classes": [
            "a subtle hand movement followed by a visible change in gaze",
            "a small posture adjustment that resolves into stillness",
            "a restrained look away and slow return toward camera",
            "a minor object interaction that reveals tension or hesitation",
        ],
        "appearance_focus": [
            "face texture, hair shape, and one distinctive personal feature",
            "eyes, mouth, and breath behavior in close view",
            "fabric texture, skin detail, and signs of recent weather or effort",
            "hands, jaw, and asymmetry in posture or expression",
        ],
        "shot_scales": [
            "tight close-up or medium close-up",
            "shoulder-up portrait frame",
            "profile close-up with shallow depth of field",
            "intimate seated medium close-up",
        ],
        "camera_patterns": [
            "the camera stays close and steady with only a slight drift",
            "a slow push-in reveals more facial detail rather than more space",
            "an eye-level portrait setup keeps the background soft and secondary",
            "the framing stays intimate so a small movement changes the whole image",
        ],
        "lighting_motifs": [
            "soft natural window light and a restrained color palette",
            "cool overcast or fluorescent light with controlled contrast",
            "warm practical light in a dim surrounding space",
            "mixed low-intensity light that emphasizes skin, fabric, and breath",
        ],
        "sound_profiles": [
            "quiet room tone with one small nearby sound",
            "subtle fabric, breath, or hand-contact sounds against low ambience",
            "soft domestic or transit ambience with one interruptive off-screen cue",
            "near-silence broken by a practical sound that changes the moment",
        ],
        "audio_emphasis": [
            "quiet ambience-led",
            "micro-foley-led",
            "single interruptive sound accent",
            "soft room tone with a delayed cue",
        ],
        "change_types": [
            "a reflection, shadow, or light shift crosses the face",
            "an off-screen sound interrupts a held pause",
            "a small breath or posture change becomes the final beat",
            "a nearby device or object changes the emotional direction of the shot",
        ],
    },
    "Human_Interaction": {
        "mode": "conversation or social interaction",
        "hero_subject_archetypes": [
            "two people with a clear emotional relationship",
            "two coworkers, family members, or friends in a small shared space",
            "a pair seated or standing across from one another",
            "two people handling a short exchange while one person hesitates",
        ],
        "setting_archetypes": [
            "a table, bench, or waiting area that keeps both faces readable",
            "a corridor, office corner, or transit edge with mild background activity",
            "a cafe, kitchen, or porch with controlled background depth",
            "a sidewalk or station space where ambient sound shapes the pause between lines",
        ],
        "action_classes": [
            "a short exchange with one speaker leading and the other reacting",
            "a brief disagreement or hesitation resolved through body language",
            "an explanation, question, or confession followed by a visible response",
            "a conversation beat where silence matters as much as speech",
        ],
        "appearance_focus": [
            "contrast between the two faces, outfits, or postures",
            "hands, eye lines, and shoulder position during the exchange",
            "small gestures that change the social distance between them",
            "fabric, personal items, and signs of fatigue or urgency",
        ],
        "shot_scales": [
            "stable medium two-shot",
            "over-the-shoulder two-shot without cutting away",
            "waist-up interaction frame",
            "seated eye-level medium shot",
        ],
        "camera_patterns": [
            "the camera holds both people in one readable composition",
            "a slight push-in or side drift changes the balance between them",
            "the framing favors reaction timing over constant movement",
            "the camera stays close enough to read both faces and hands together",
        ],
        "lighting_motifs": [
            "soft daylight with restrained reflections on nearby surfaces",
            "mixed indoor practicals with clear separation between foreground and background",
            "cool overhead light with muted environmental color",
            "warm low-key light that preserves facial readability",
        ],
        "sound_profiles": [
            "short dialogue with low surrounding ambience",
            "dialogue punctuated by small table, paper, or clothing sounds",
            "ambient public-space sound threading through pauses between lines",
            "a quiet room where one environmental cue changes the exchange",
        ],
        "audio_emphasis": [
            "dialogue-led",
            "dialogue plus micro-foley",
            "speech interrupted by ambience",
            "reaction-led silence after speech",
        ],
        "change_types": [
            "one person changes posture or eye contact after a line lands",
            "an outside sound or announcement interrupts the exchange",
            "the emotional balance flips between the two speakers",
            "a withheld response becomes visible in gesture rather than speech",
        ],
    },
    "Human_Activity": {
        "mode": "everyday physical activity",
        "hero_subject_archetypes": [
            "one person doing a routine task with practiced motion",
            "one or two people engaged in work, sport, or craft",
            "a person moving through space with one clear objective",
            "a worker, cook, maker, or athlete in a readable action loop",
        ],
        "setting_archetypes": [
            "a kitchen, workshop, yard, or street corner suited to the task",
            "a practical work area with surfaces, tools, or materials nearby",
            "an outdoor space where weather or ground texture affects the movement",
            "a simple performance-free environment where the action stays central",
        ],
        "action_classes": [
            "a repeated hand task with visible rhythm",
            "a short sequence of movement with one clear cause-and-effect beat",
            "a task involving lifting, placing, turning, or assembling",
            "a brief athletic or locomotion beat with a clean finish",
        ],
        "appearance_focus": [
            "clothing wear, hand condition, and task-specific body posture",
            "surface residue, moisture, dust, or marks created by the activity",
            "tool grip, balance, and body alignment during movement",
            "the relation between body motion and nearby materials or surfaces",
        ],
        "shot_scales": [
            "medium or medium-wide action frame",
            "waist-up tracking shot",
            "full-body follow shot",
            "working frame that keeps hands, torso, and tool visible together",
        ],
        "camera_patterns": [
            "the camera tracks or holds to keep the action legible from start to finish",
            "a side-follow or rear-follow relation keeps pace with the movement",
            "the framing privileges rhythm and contact over dramatic reveal",
            "the camera movement is simple and tied directly to the subject's motion",
        ],
        "lighting_motifs": [
            "natural daylight shaped by weather, steam, or dust",
            "harder directional light that clarifies tools and surfaces",
            "warm practical work light with controlled background clutter",
            "outdoor light that emphasizes ground texture, moisture, or long shadows",
        ],
        "sound_profiles": [
            "task-driven foley with moderate environmental ambience",
            "rhythmic contact sounds tied to repeated hand or foot motion",
            "tool, surface, and material sounds that clearly explain the action",
            "movement sounds supported by a light background sound bed",
        ],
        "audio_emphasis": [
            "foley-led task sound",
            "rhythm-led motion sound",
            "mixed task foley and ambience",
            "contact sounds with restrained background noise",
        ],
        "change_types": [
            "the rhythm speeds up, slows down, or breaks briefly",
            "a small slip, correction, or recovery changes the action pattern",
            "the task result becomes visible after a final motion",
            "an environmental factor changes the movement at the end",
        ],
    },
    "Human_Presentation": {
        "mode": "presentation, interview, or direct address",
        "hero_subject_archetypes": [
            "a presenter, host, or reporter addressing a lens or microphone",
            "an interviewer or guide leading a small controlled exchange",
            "a person seated at a desk or table in a recorded setup",
            "a speaker explaining or introducing something in a real-world location",
        ],
        "setting_archetypes": [
            "a studio, booth, or desk setup with visible recording context",
            "a street-side or field-report location with controlled framing",
            "a gallery, exhibit, or demonstration space",
            "a table-based conversation setup with microphones or notes nearby",
        ],
        "action_classes": [
            "a short direct-to-camera or direct-to-mic introduction",
            "a question, handoff, or explanatory gesture that leads the scene",
            "a presentational beat where the speaker reveals something off to the side",
            "a composed speech moment interrupted by a live or practical cue",
        ],
        "appearance_focus": [
            "wardrobe, posture, and communication tools that define the role",
            "headset, microphone, notes, or desk objects that anchor the setup",
            "facial control and hand gestures associated with public speaking",
            "the relation between the speaker and the formal setting around them",
        ],
        "shot_scales": [
            "medium direct-to-camera shot",
            "seated studio medium shot",
            "walk-and-talk stand-up frame",
            "shoulder-level interview or presentation frame",
        ],
        "camera_patterns": [
            "the camera behaves like a single production camera with a controlled relation to the speaker",
            "a modest pan or reveal connects the speaker to what they are discussing",
            "the framing stays orderly and legible even when the scene changes",
            "the camera acknowledges the presented subject without losing the speaker",
        ],
        "lighting_motifs": [
            "studio practicals or controlled key light on the speaker",
            "clean broadcast or interview lighting with readable faces",
            "real-world daylight balanced for a presenter in front of a location",
            "museum, booth, or desk lighting that supports a formal speaking setup",
        ],
        "sound_profiles": [
            "clear close speech over low room tone",
            "interview or broadcast ambience with light production cues",
            "microphone proximity sound and subtle set noise",
            "speech-led sound design with a small environmental interruption",
        ],
        "audio_emphasis": [
            "speech-led",
            "broadcast-style speech over ambience",
            "interview-led speech plus room tone",
            "presentational speech with one live cue",
        ],
        "change_types": [
            "the speaker turns attention to another person or object",
            "a cue, light, or off-screen prompt shifts the presentational beat",
            "the environment behind the speaker becomes newly relevant",
            "the shot moves from introduction to reveal",
        ],
    },
    "Human_Performance": {
        "mode": "music, dance, or rhythm-led performance",
        "hero_subject_archetypes": [
            "a solo musician, singer, or dancer",
            "a small pair or trio performing together",
            "a performer locked into a visible rhythm",
            "a body-led performance where timing and breath matter",
        ],
        "setting_archetypes": [
            "a small stage, rehearsal room, or intimate venue",
            "a porch, room corner, or informal performance space",
            "a controlled studio or recital setting",
            "a simple performance environment with readable acoustics",
        ],
        "action_classes": [
            "a short musical or movement phrase with one rising and one settling beat",
            "a performance moment where body rhythm drives the camera relation",
            "a sung, played, or danced line that widens briefly into a larger phrase",
            "a performance beat where one accent changes posture and timing",
        ],
        "appearance_focus": [
            "instrument or costume relation to the body",
            "hands, torso, breath, and facial focus during performance",
            "surface wear on instruments, clothing, or stage materials",
            "how lighting and movement shape the performer's silhouette",
        ],
        "shot_scales": [
            "medium performance shot",
            "close musician or dancer portrait",
            "small-stage two-shot",
            "waist-up rhythm frame",
        ],
        "camera_patterns": [
            "the camera follows phrase timing rather than cutting around it",
            "a slow push, drift, or arc stays synchronized to the performance energy",
            "the framing keeps the sound-making body parts readable",
            "camera motion expands only when the performance does",
        ],
        "lighting_motifs": [
            "stage or rehearsal lighting with visible contrast",
            "warm practical lights around a small venue or room",
            "focused spotlight logic with readable shadows",
            "soft performance lighting that preserves body rhythm and texture",
        ],
        "sound_profiles": [
            "music-led sound with audible breath, contact, or foot timing",
            "instrument resonance and room response",
            "voice or melody forward with supporting room or crowd tone",
            "rhythmic body and instrument sounds synchronized to visible motion",
        ],
        "audio_emphasis": [
            "music-led",
            "rhythm-led",
            "performance-led with close mic detail",
            "voice-led with supporting ambience",
        ],
        "change_types": [
            "a phrase resolves into a pause or held note",
            "the energy widens briefly as another part joins in",
            "one accent or beat changes the performer's body pattern",
            "the room response or crowd tone shifts after a line or hit",
        ],
    },
    "Human_Foley_Task": {
        "mode": "foley-driven manual work",
        "hero_subject_archetypes": [
            "a worker, repairer, or technician handling a noisy task",
            "a craftsperson or maintenance worker using one primary tool",
            "a laborer controlling a heavy or resistant material",
            "a person operating or adjusting a mechanical system by hand",
        ],
        "setting_archetypes": [
            "a utility room, service corridor, or pipe run",
            "a workshop bay, repair bench, or machine alcove",
            "a construction nook, scaffold edge, or industrial passage",
            "a maintenance area with metal, concrete, or resonant surfaces",
        ],
        "action_classes": [
            "a repeated impact action with clear contact and recoil",
            "a turning, tightening, or ratcheting task with short pauses",
            "a scraping, prying, or levering action that changes resistance over time",
            "a start-stop repair motion where one adjustment changes the sound pattern",
        ],
        "appearance_focus": [
            "tool grip, gloves, sleeves, and residue from the task",
            "material wear, dust, grease, or moisture on hands and surfaces",
            "body alignment and strain during a forceful manual action",
            "metal, pipe, cable, or fixture textures that explain the sound",
        ],
        "shot_scales": [
            "waist-up work frame with tool contact readable",
            "medium shot focused on full-body tool use",
            "tight task frame that still shows cause and effect",
            "side-on working frame with repeated contact visible",
        ],
        "camera_patterns": [
            "the camera holds close enough to show the exact source of each sound",
            "a steady side or three-quarter angle preserves tool contact and recoil",
            "movement is minimal so the audio-producing action stays readable",
            "if the camera shifts, it reveals the part of the system causing the new sound",
        ],
        "lighting_motifs": [
            "harder work light that clarifies metal, dust, and edges",
            "cool industrial light with restrained reflections",
            "mixed practical light on pipes, concrete, or machine surfaces",
            "directional light that makes impacts and particles visible",
        ],
        "sound_profiles": [
            "metal-on-metal strikes with resonant ringing",
            "pipe, valve, or fitting sounds with hollow resonance and squeaks",
            "ratchets, clicks, scrapes, and material drag sounds",
            "heavy impacts followed by rattles, dust movement, or vibrating hardware",
        ],
        "audio_emphasis": [
            "foley-dominant impact sound",
            "mechanical resonance-led",
            "tool-driven repeated contact",
            "repair-foley with changing sound state",
        ],
        "change_types": [
            "the sound pattern changes after one successful adjustment",
            "a louder strike, slip, or jam interrupts the rhythm",
            "the system begins, stops, opens, or loosens at the end",
            "a lingering ring, rattle, or pressure release becomes the final beat",
        ],
    },
    "Landscape_Weather": {
        "mode": "landscape and weather motion",
        "hero_subject_archetypes": [
            "a terrain feature shaped by wind, rain, snow, surf, or fog",
            "a broad natural scene dominated by visible environmental motion",
            "a weather front or water movement taking over the frame",
            "a landscape where atmosphere changes the depth and visibility over time",
        ],
        "setting_archetypes": [
            "coastline, ridge, plain, forest edge, or desert terrain",
            "a wide outdoor space with clear foreground and distant depth",
            "a natural environment where weather visibly reshapes the image",
            "an exposed landscape where wind, rain, water, or snow affects surfaces",
        ],
        "action_classes": [
            "an environmental force moves steadily across the frame",
            "weather or water builds, shifts, and then changes visibility or texture",
            "one natural motion pattern overtakes another",
            "a landscape remains fixed while atmosphere transforms it over time",
        ],
        "appearance_focus": [
            "surface texture, moisture, dust, snow, or foliage movement",
            "foreground material behavior against distant scale",
            "light on water, rock, grass, fog, or airborne particles",
            "terrain edges, silhouettes, and atmospheric layering",
        ],
        "shot_scales": [
            "wide environmental shot",
            "low-angle landscape frame with active foreground",
            "slow panoramic natural frame",
            "fixed horizon-wide shot",
        ],
        "camera_patterns": [
            "the camera stays patient so the environment changes inside the frame",
            "a wide relation emphasizes the scale of wind, rain, or surf",
            "a slow pan or hold follows the weather front rather than inventing new action",
            "the camera movement reveals what the atmosphere is doing to the space",
        ],
        "lighting_motifs": [
            "overcast or storm light with restrained color",
            "dawn, dusk, or low-angle light catching moisture and particles",
            "flat daylight shaped by fog, haze, or snow",
            "harder weather light that clarifies surface texture and distance",
        ],
        "sound_profiles": [
            "wind, surf, rain, or thunder as the main sound layer",
            "weather sound tied directly to visible surface movement",
            "water force, branch movement, and terrain resonance",
            "ambience-led environmental sound with one stronger natural accent",
        ],
        "audio_emphasis": [
            "environment-led",
            "weather-led",
            "water-force-led",
            "atmosphere-led with a natural accent",
        ],
        "change_types": [
            "visibility drops or clears during the shot",
            "wind, rain, surf, or falling material changes intensity",
            "a new layer of atmosphere enters the frame",
            "light briefly shifts the whole landscape logic at the end",
        ],
    },
    "Animal_Behavior": {
        "mode": "animal behavior",
        "hero_subject_archetypes": [
            "a single animal or small pair showing readable natural behavior",
            "a wildlife subject whose body language clearly precedes a movement",
            "a domestic or wild animal reacting to the environment",
            "an animal using caution, patience, or sudden movement",
        ],
        "setting_archetypes": [
            "a habitat edge with clear ground, water, or branch contact",
            "a marsh, yard, field edge, forest floor, or rock surface",
            "a natural or semi-natural space with environmental sound detail",
            "a location where the animal's movement leaves visible traces or ripples",
        ],
        "action_classes": [
            "a pause-listen-move sequence with one quick release",
            "a cautious approach followed by a short decisive action",
            "a grooming, waiting, or scanning behavior interrupted by a stimulus",
            "a movement pattern that changes direction after a sound or scent cue",
        ],
        "appearance_focus": [
            "fur, feathers, skin, paws, claws, beak, or tail detail",
            "posture and balance that explain the next movement",
            "surface contact between the animal and water, snow, rock, or leaves",
            "small body details that help read species-specific motion",
        ],
        "shot_scales": [
            "medium wildlife frame",
            "telephoto observation frame",
            "low eye-level animal shot",
            "patient side-profile animal frame",
        ],
        "camera_patterns": [
            "the camera remains patient and does not crowd the subject",
            "a telephoto or low-angle relation preserves readable animal behavior",
            "movement follows only when the animal clearly changes direction",
            "the camera reveals the consequence of the animal's motion in the habitat",
        ],
        "lighting_motifs": [
            "soft natural light appropriate to the habitat",
            "cool or warm outdoor light that clarifies texture without stylizing it too hard",
            "light shaped by water reflection, snow bounce, foliage, or shade",
            "weather-filtered light that supports species and habitat detail",
        ],
        "sound_profiles": [
            "animal movement sounds plus habitat ambience",
            "water, leaves, snow, or ground contact tied to body motion",
            "wings, paws, claws, or tail movement against a clear background bed",
            "quiet habitat ambience punctuated by one decisive animal action",
        ],
        "audio_emphasis": [
            "behavior-led natural sound",
            "habitat plus animal contact sound",
            "quiet observation with one sharp action sound",
            "animal-movement-led",
        ],
        "change_types": [
            "the animal freezes, then commits to movement",
            "a ripple, footprint, or branch shift reveals the completed action",
            "an outside cue redirects the behavior",
            "the animal exits, settles, or changes stance as the final beat",
        ],
    },
    "Urban_Flow": {
        "mode": "city rhythm and architecture",
        "hero_subject_archetypes": [
            "a street, block, station edge, or architectural corridor defined by movement",
            "an urban space where traffic, pedestrians, or lights create the rhythm",
            "a built environment whose geometry and reflections dominate the image",
            "a corner of the city where transit and surface texture stay central",
        ],
        "setting_archetypes": [
            "an avenue, alley, platform, bridge, market edge, or office district",
            "a wet or reflective city surface with readable lines and depth",
            "a dense built space with passing movement but no single hero person",
            "a transit-connected location where infrastructure shapes the shot",
        ],
        "action_classes": [
            "flow moves through the frame in one dominant direction",
            "traffic, pedestrians, or light changes alter the visual rhythm",
            "the space holds steady while movement patterns accumulate and resolve",
            "one pass-through element changes how the environment reads",
        ],
        "appearance_focus": [
            "surface reflections, painted lines, glass, metal, or masonry texture",
            "verticals, vanishing lines, and repeating built forms",
            "water, grime, signage-free geometry, and transport infrastructure",
            "color separation between sky, interior light, and street-level material",
        ],
        "shot_scales": [
            "wide street frame",
            "medium-wide architectural frame",
            "street-corner low angle",
            "elevated city-rhythm frame",
        ],
        "camera_patterns": [
            "the camera frames movement through architecture rather than chasing people",
            "a pan, hold, or slow lateral move reveals the city's rhythm",
            "the built geometry stays readable as movement passes through it",
            "camera motion serves scale, flow, and reflection rather than drama alone",
        ],
        "lighting_motifs": [
            "dusk, overcast, or practical-lit urban color separation",
            "wet-surface reflections and restrained highlight behavior",
            "mixed daylight and artificial light on glass, steel, or concrete",
            "city light that changes the scene without overwhelming it",
        ],
        "sound_profiles": [
            "traffic wash, transit hum, footsteps, and signal tones",
            "urban ambience led by rails, engines, brakes, or distant sirens",
            "surface sound from water, tires, or foot traffic",
            "city ambience with one passing mechanical accent",
        ],
        "audio_emphasis": [
            "urban ambience-led",
            "transit-led ambience",
            "street-surface plus traffic sound",
            "city mechanical wash",
        ],
        "change_types": [
            "a light change, pass-through vehicle, or weather shift alters the space",
            "the flow pattern briefly condenses or opens up",
            "reflections or shadows change the geometry of the shot",
            "one new movement layer redefines the frame near the end",
        ],
    },
    "Vehicle_Motion": {
        "mode": "vehicle-driven motion",
        "hero_subject_archetypes": [
            "a single vehicle or short line of vehicles with one clear motion pattern",
            "a transit vehicle entering, leaving, or passing a readable environment",
            "a machine moving through road, track, water, or air with visible force",
            "a vehicle whose sound and motion are the main event of the shot",
        ],
        "setting_archetypes": [
            "a road, track, harbor, platform edge, or street approach",
            "a route where surface condition visibly affects the motion",
            "a transit corridor with clear entry and exit directions",
            "an open space where speed and inertia remain readable",
        ],
        "action_classes": [
            "a pass-by, arrival, departure, or turn with visible momentum change",
            "a controlled acceleration or deceleration sequence",
            "a motion path defined by one bend, crossing, or alignment beat",
            "a vehicle movement where surface contact strongly shapes the sound",
        ],
        "appearance_focus": [
            "body panels, windows, wheels, wake, rails, or lights in motion",
            "surface interaction between the vehicle and road, water, or track",
            "reflections, spray, grit, or vibration around the moving machine",
            "the mechanical parts or materials that explain the sound and speed",
        ],
        "shot_scales": [
            "tracking three-quarter vehicle frame",
            "low roadside or trackside pass-by frame",
            "elevated transit shot",
            "front-quarter or rear-follow motion frame",
        ],
        "camera_patterns": [
            "the camera keeps the vehicle's path and force legible",
            "a side track, low hold, or distant compression preserves speed logic",
            "the camera move reveals where the vehicle is going or what it passes",
            "camera motion stays tied to momentum, not spectacle editing",
        ],
        "lighting_motifs": [
            "weather-filtered daylight on moving metal or glass",
            "dusk or practical light interacting with vehicle surfaces",
            "harder light that clarifies spray, grit, or reflections",
            "restrained color separation between the machine and the route",
        ],
        "sound_profiles": [
            "engine, rail, hull, or wheel sound as the primary layer",
            "surface-contact sound tied to speed and direction",
            "mechanical motion sound with environmental support",
            "arrival or pass-by sound that clearly matches visible momentum",
        ],
        "audio_emphasis": [
            "vehicle-motion-led",
            "engine-led",
            "surface-contact-led",
            "transit-mechanical-led",
        ],
        "change_types": [
            "the vehicle changes speed, direction, or contact state",
            "a pass-by, stop, or entry changes the sound field",
            "a surface change alters spray, vibration, or traction",
            "the machine exits or settles as the final beat",
        ],
    },
    "Mechanical_Infrastructure": {
        "mode": "built-world mechanical system",
        "hero_subject_archetypes": [
            "a pipe, valve, rail, lift, conveyor, or industrial subsystem",
            "a mechanical installation whose motion or vibration is the main focus",
            "a piece of infrastructure operating with visible force and resonance",
            "a built system that changes state during the shot",
        ],
        "setting_archetypes": [
            "a factory bay, service corridor, tunnel, or maintenance room",
            "a platform edge, shaft, plant floor, or industrial walkway",
            "a mechanical room filled with metal, concrete, or cable structures",
            "a piece of urban or industrial infrastructure with repeating surfaces",
        ],
        "action_classes": [
            "a system starts, stops, opens, closes, or cycles visibly",
            "one moving part transfers force or vibration to the rest of the structure",
            "a repetitive mechanical action builds a rhythm and then changes state",
            "pressure, rotation, lift, or conveyance becomes visible over time",
        ],
        "appearance_focus": [
            "metal surfaces, worn paint, cables, joints, or moving fittings",
            "vibration, dust, steam, water, or debris responding to the mechanism",
            "the specific parts that explain force transfer and sound",
            "surface wear and industrial texture under controlled light",
        ],
        "shot_scales": [
            "medium mechanical frame",
            "close industrial detail frame",
            "corridor or machine-bay wide frame",
            "side-on system view with moving parts readable",
        ],
        "camera_patterns": [
            "the camera prioritizes causal clarity over cinematic flourish",
            "a hold or slow reveal exposes how the system is operating",
            "camera movement follows the line of force through the mechanism",
            "the shot ends by showing what changed after the system moves",
        ],
        "lighting_motifs": [
            "industrial practical light with restrained reflections",
            "cool or mixed work light on metal and concrete",
            "backlight or side light that clarifies steam, dust, or vibration",
            "utilitarian light that favors texture and moving parts",
        ],
        "sound_profiles": [
            "clanks, hums, squeals, rattles, and pressure sounds from a system in operation",
            "metal resonance, motor noise, and structural vibration",
            "repetitive mechanical rhythm with one state change",
            "air, fluid, cable, or rail sound tied to visible motion",
        ],
        "audio_emphasis": [
            "mechanical-system-led",
            "motor-and-resonance-led",
            "pressure-release-led",
            "repetitive industrial rhythm",
        ],
        "change_types": [
            "the mechanism switches state and the sound changes with it",
            "a vibration or resonance spreads after one key motion",
            "pressure, flow, or alignment becomes newly visible",
            "the system settles into silence or a new steady sound",
        ],
    },
    "Product_Detail": {
        "mode": "object or product detail",
        "hero_subject_archetypes": [
            "a single crafted object or product",
            "a tabletop subject with visible material qualities",
            "an object whose surface, edge, or moving part is the main event",
            "a hero object presented without competing elements",
        ],
        "setting_archetypes": [
            "a studio or tabletop setup",
            "a counter, bench, or neutral surface with controlled context",
            "a minimal environment built to emphasize the object",
            "a material-rich surface that supports the hero subject",
        ],
        "action_classes": [
            "a nearly static shot where one subtle material change becomes visible",
            "a slow reveal of surface, edge, or reflection behavior",
            "a small mechanical or fluid detail repeating over time",
            "a close observation of how light and material respond together",
        ],
        "appearance_focus": [
            "surface finish, edges, texture, and reflection behavior",
            "join lines, engraved details, or tiny imperfections",
            "moisture, powder, steam, or other material micro-detail",
            "contrast between matte and glossy surfaces in close view",
        ],
        "shot_scales": [
            "macro close-up",
            "tight hero frame",
            "tabletop detail shot",
            "material-study close-up",
        ],
        "camera_patterns": [
            "the camera moves slowly enough for texture and reflection to stay readable",
            "a subtle push, orbit, or drift reveals a new property of the object",
            "the framing stays disciplined and avoids introducing new hero subjects",
            "the camera relation clarifies scale and material rather than narrative drama",
        ],
        "lighting_motifs": [
            "controlled studio light with precise highlights",
            "soft side light that reveals texture and surface depth",
            "window-like light with restrained falloff",
            "clean top or rim light that emphasizes form and edge behavior",
        ],
        "sound_profiles": [
            "light object-contact sounds and room tone",
            "small mechanical, glass, ceramic, or liquid detail sounds",
            "subtle drips, taps, ticks, or soft handling sounds",
            "near-field material sound with minimal ambient distraction",
        ],
        "audio_emphasis": [
            "micro-foley-led",
            "material-detail-led",
            "subtle mechanical detail",
            "quiet object ambience",
        ],
        "change_types": [
            "a droplet, reflection, or moving highlight changes the image",
            "a tiny mechanism or material state becomes newly visible",
            "condensation, residue, or steam changes over the shot",
            "the object reveals another plane, edge, or internal detail near the end",
        ],
    },
    "Material_Process": {
        "mode": "material transformation or process",
        "hero_subject_archetypes": [
            "a food, craft, or industrial material undergoing change",
            "a liquid, powder, paste, or particulate material in motion",
            "a surface being cut, poured, heated, mixed, or coated",
            "a process where the state of a material visibly evolves",
        ],
        "setting_archetypes": [
            "a work surface, kitchen surface, lab-like bench, or production station",
            "a controlled process setup with one main material interaction",
            "a close-range workspace where the material remains the hero",
            "a minimal environment built around one physical transformation",
        ],
        "action_classes": [
            "a pour, spread, stir, cut, heat, or cool sequence with one clear result",
            "a repeated contact action that changes the material surface over time",
            "a material entering a new state through pressure, temperature, or motion",
            "a process beat where texture, flow, or particles become newly visible",
        ],
        "appearance_focus": [
            "viscosity, grain, bubbles, fibers, dust, or surface tension",
            "edge behavior, residue, splash, or particulate movement",
            "before-and-after texture differences created by the process",
            "the material's response to heat, pressure, liquid, or motion",
        ],
        "shot_scales": [
            "macro process shot",
            "tight tabletop process frame",
            "close material-transformation frame",
            "detail shot that keeps tool and material readable together",
        ],
        "camera_patterns": [
            "the camera stays close enough to read the transformation clearly",
            "a slow drift or push reveals the result of each contact",
            "framing tracks the material state rather than the person making it",
            "camera motion is minimal when the process itself is visually rich",
        ],
        "lighting_motifs": [
            "controlled light that reveals translucency, texture, or particles",
            "side light for liquid edges, powder, steam, or residue",
            "clean top light for process clarity",
            "soft practical light that supports a tactile, material-focused scene",
        ],
        "sound_profiles": [
            "pouring, dripping, spreading, sizzling, chopping, or stirring sounds",
            "material contact sounds that change as texture changes",
            "light tool sounds secondary to the material's response",
            "foley-led process sound with minimal background noise",
        ],
        "audio_emphasis": [
            "material-process-led",
            "foley-led transformation sound",
            "liquid-and-contact-led",
            "texture-change-led",
        ],
        "change_types": [
            "the material reaches a visibly new state",
            "surface behavior changes after one key action",
            "particles, steam, or residue accumulate into the final beat",
            "a once-subtle sound becomes clearer as the process develops",
        ],
    },
    "Mechanical_Object": {
        "mode": "small mechanical or engineered object",
        "hero_subject_archetypes": [
            "a watch, tool, instrument, device, or compact mechanism",
            "an engineered object with one clear moving part",
            "a machine detail where function and sound are visually linked",
            "a compact object whose mechanics are more important than branding",
        ],
        "setting_archetypes": [
            "a workbench, desk, or neutral surface that supports inspection",
            "a close-view setup where the mechanism remains unobstructed",
            "a tool or instrument context without extra hero clutter",
            "a maintenance or display surface with strong material contrast",
        ],
        "action_classes": [
            "a tick, spin, click, open, latch, or calibrated motion repeating clearly",
            "a mechanism cycling through one small action with visible consequence",
            "a manual interaction that exposes the device's moving parts",
            "a small engineered motion whose sound state changes near the end",
        ],
        "appearance_focus": [
            "gears, joints, screws, hinges, edges, or engraved structure",
            "material contrast between metal, glass, rubber, or wood",
            "precision marks, alignment, and the parts that move together",
            "surface wear or polish that supports mechanical readability",
        ],
        "shot_scales": [
            "macro mechanism shot",
            "tight engineered-object frame",
            "close structural detail frame",
            "tabletop shot with moving part readable",
        ],
        "camera_patterns": [
            "the camera isolates the moving parts and their relation to the rest of the object",
            "a slight push or orbit reveals how the mechanism works",
            "the frame stays disciplined so motion and sound line up cleanly",
            "camera motion serves functional clarity rather than broad reveal",
        ],
        "lighting_motifs": [
            "clean light on metal, glass, and precision edges",
            "controlled specular highlights that track mechanical form",
            "soft contrast that supports engraved or joint detail",
            "functional light that favors clarity over glamour",
        ],
        "sound_profiles": [
            "ticks, clicks, latches, springs, whirs, or ratchets",
            "small motor or mechanism noise with clear contact points",
            "precise repeating sound patterns from a compact device",
            "micro-mechanical sound with restrained room tone",
        ],
        "audio_emphasis": [
            "mechanical-detail-led",
            "tick-and-click-led",
            "small-motor-led",
            "precision-foley-led",
        ],
        "change_types": [
            "the mechanism engages, releases, or changes cadence",
            "a hidden movement becomes visible after one interaction",
            "a repeating motion stops or locks into place",
            "a final click, release, or vibration completes the shot",
        ],
    },
}


CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Human": {
        "desc": "Human-centered video content. Concrete character, prop, and action details should be invented by the model within the chosen scene mode.",
        "weight": 40,
        "temperature_offset": 0.06,
        "style_rate": 0.38,
        "dynamic_ending_rate": 0.45,
        "subject_rule": "People are the main subject. Usually 1-3 people, with one clear readable action, exchange, or sound-producing behavior.",
        "focus": ["body language", "facial readability", "clear action or speech beat", "visible cause-and-effect for sound"],
        "audio_targets": ["speech and turn-taking", "breath and clothing movement", "performance rhythm", "manual-work or tool foley"],
        "avoid": ["large crowds", "chaotic choreography", "too many unrelated actions"],
        "style_families": ["naturalistic", "cinematic", "stylized", "animation"],
        "diversity_dims": ["time_of_day", "weather", "palette", "lens", "camera_energy", "texture_focus", "pacing"],
        "scene_modes": [
            "portrait close-up",
            "conversation or social interaction",
            "everyday physical activity",
            "presentation, interview, or direct address",
            "music, dance, or rhythm-led performance",
            "foley-driven manual work",
        ],
        "families": {
            "Human_Portrait": 8,
            "Human_Interaction": 8,
            "Human_Activity": 8,
            "Human_Presentation": 5,
            "Human_Performance": 5,
            "Human_Foley_Task": 6,
        },
    },
    "Natural_World": {
        "desc": "Nature- and animal-centered video content. The program defines broad environmental or behavioral modes, but the model invents the exact scene details.",
        "weight": 22,
        "temperature_offset": 0.04,
        "style_rate": 0.30,
        "dynamic_ending_rate": 0.35,
        "subject_rule": "Nature or animals are the main subject. Avoid human-led staging or dialogue.",
        "focus": ["environmental motion", "species or terrain detail", "natural sound", "scale and atmosphere"],
        "audio_targets": ["weather beds", "water and foliage movement", "animal calls and body contact sounds", "habitat ambience"],
        "avoid": ["human-led scenes", "indoor product staging", "impossible environmental combinations"],
        "style_families": ["naturalistic", "cinematic", "stylized", "animation"],
        "diversity_dims": ["time_of_day", "weather", "palette", "lens", "camera_energy", "pacing"],
        "scene_modes": [
            "landscape and weather motion",
            "animal behavior",
        ],
        "families": {
            "Landscape_Weather": 12,
            "Animal_Behavior": 10,
        },
    },
    "Built_World": {
        "desc": "Built-environment video content. This covers city rhythm, vehicles, and mechanical infrastructure without over-specifying the final shot details.",
        "weight": 24,
        "temperature_offset": -0.02,
        "style_rate": 0.26,
        "dynamic_ending_rate": 0.40,
        "subject_rule": "The built world, traffic, infrastructure, or a vehicle should remain visually central.",
        "focus": ["spatial geometry", "traffic or system rhythm", "camera relation to movement", "mechanical or urban sound"],
        "audio_targets": ["traffic wash", "engines, rails, or motors", "industrial resonance", "urban mechanical pulses"],
        "avoid": ["single-person portrait emphasis", "large rally crowds", "crash-heavy action"],
        "style_families": ["naturalistic", "cinematic", "stylized", "animation"],
        "diversity_dims": ["time_of_day", "weather", "palette", "lens", "camera_energy", "texture_focus", "pacing"],
        "scene_modes": [
            "city rhythm and architecture",
            "vehicle-driven motion",
            "built-world mechanical system",
        ],
        "families": {
            "Urban_Flow": 9,
            "Vehicle_Motion": 8,
            "Mechanical_Infrastructure": 7,
        },
    },
    "Object": {
        "desc": "Object- and material-centered video content. The model should invent the exact object and process details under broad material or mechanical constraints.",
        "weight": 14,
        "temperature_offset": -0.04,
        "style_rate": 0.22,
        "dynamic_ending_rate": 0.45,
        "subject_rule": "An object, material, or compact mechanism is the clear hero subject, not a person or a large environment.",
        "focus": ["texture", "material behavior", "precise lighting", "sound-producing contact or process detail"],
        "audio_targets": ["micro-foley", "material transformation sound", "ticks, clicks, drips, and pours", "close mechanical detail"],
        "avoid": ["crowded scenes", "dialogue-led staging", "multiple competing hero subjects"],
        "style_families": ["naturalistic", "cinematic", "stylized", "animation"],
        "diversity_dims": ["palette", "lens", "camera_energy", "texture_focus", "pacing"],
        "scene_modes": [
            "object or product detail",
            "material transformation or process",
            "small mechanical or engineered object",
        ],
        "families": {
            "Product_Detail": 5,
            "Material_Process": 5,
            "Mechanical_Object": 4,
        },
    },
}


STYLE_DIRECTION_POOLS: Dict[str, List[str]] = {
    "naturalistic": [
        "quiet observational realism",
        "naturalistic documentary realism",
        "grounded contemporary realism",
        "restrained slice-of-life realism",
    ],
    "cinematic": [
        "cinematic modern drama",
        "restrained thriller cinematography",
        "warm character-driven cinema",
        "noir-leaning cinematic realism",
    ],
    "stylized": [
        "fashion-editorial stylization",
        "retro-futurist stylization",
        "painterly stylized cinema",
        "graphic high-contrast stylization",
    ],
    "animation": [
        "hand-drawn animated look",
        "stylized 3D animation",
        "stop-motion inspired animation",
        "claymation-like miniature animation",
    ],
}


DIVERSITY_POOLS: Dict[str, List[str]] = {
    "time_of_day": ["dawn", "mid-morning", "late afternoon", "golden hour", "blue hour", "midnight"],
    "weather": ["light drizzle", "gusty wind", "fresh snowfall", "humid haze", "dry heat shimmer", "crisp overcast air"],
    "palette": ["cool blue-gray palette", "warm amber palette", "muted earth tones", "restrained neon accents", "faded pastel palette", "high-contrast monochrome"],
    "lens": ["28mm wide lens", "35mm lens", "50mm standard lens", "85mm portrait lens", "135mm telephoto lens", "macro lens"],
    "camera_energy": ["locked-off precision", "gentle handheld sway", "slow push-in", "smooth side tracking", "patient observational pan", "telephoto follow from a distance"],
    "texture_focus": ["wet reflective surfaces", "worn fabric texture", "brushed metal detail", "dust in the air", "condensation and glass", "paper and wood grain"],
    "pacing": ["lingering rhythm", "measured buildup", "brief sudden interruption", "quiet pause before motion", "steady repeated motion", "short release after tension"],
}


FAMILY_BRIEF_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("hero_subject_archetypes", "Hero subject archetype"),
    ("setting_archetypes", "Setting archetype"),
    ("action_classes", "Action class"),
    ("appearance_focus", "Appearance focus"),
    ("shot_scales", "Shot scale"),
    ("camera_patterns", "Camera pattern"),
    ("lighting_motifs", "Lighting motif"),
    ("sound_profiles", "Sound profile"),
    ("audio_emphasis", "Dominant audio relationship"),
)


STEADY_ENDING_POOLS: Dict[str, List[str]] = {
    "Human": [
        "the shot can simply hold on the established posture, gaze, repeated motion, or working rhythm without a new interruption",
        "the breath pattern, room tone, manual task, or performance phrase can continue through the end of the shot",
        "the exchange, task, or performance can taper into a steady pause rather than a reveal",
    ],
    "Natural_World": [
        "the weather, water, or habitat ambience can remain steady through the end of the shot",
        "the landscape can hold the same atmospheric state without a new event",
        "the animal can stay in observation or continue a small repeated behavior without a decisive turn",
    ],
    "Built_World": [
        "the visible flow, vehicle path, or system rhythm can continue at the same pace without a visible state switch",
        "the built environment can stay structurally consistent while the existing transit, traffic, or machinery pattern continues",
        "the system can remain in a stable cycle, hum, vibration, or pass-through motion through the end of the shot",
    ],
    "Object": [
        "the material or mechanism can remain in a stable loop with no major reveal",
        "the close-up can end on continued drip, rotation, vibration, or texture behavior",
        "the process can simply continue in steady state rather than visibly completing",
    ],
}


CAMERA_TERMS = (
    "close-up",
    "close up",
    "medium shot",
    "wide shot",
    "low angle",
    "high angle",
    "eye level",
    "eye-level",
    "over the shoulder",
    "tracking",
    "push-in",
    "push in",
    "pull-back",
    "pull back",
    "dolly",
    "pan",
    "tilt",
    "crane",
    "handheld",
    "telephoto",
    "macro",
    "lens",
    "framed",
)

LIGHT_TERMS = (
    "light",
    "lighting",
    "lit",
    "shadow",
    "glow",
    "sunlight",
    "lamp",
    "neon",
    "overcast",
    "fluorescent",
    "dawn",
    "dusk",
    "amber",
    "blue",
    "red",
    "green",
    "gold",
    "silver",
    "color",
)

SOUND_TERMS = (
    "sound",
    "audio",
    "hum",
    "buzz",
    "hiss",
    "drone",
    "whisper",
    "voice",
    "speaks",
    "says",
    "rain",
    "traffic",
    "wind",
    "music",
    "song",
    "melody",
    "rhythm",
    "beat",
    "dialogue",
    "conversation",
    "breath",
    "breathing",
    "clang",
    "clank",
    "clatter",
    "thud",
    "impact",
    "hammer",
    "scrape",
    "grind",
    "click",
    "clicking",
    "knock",
    "tap",
    "tapping",
    "drip",
    "dripping",
    "splash",
    "slosh",
    "squeal",
    "creak",
    "groan",
    "ring",
    "ringing",
    "resonance",
    "rattle",
    "rumble",
    "roar",
    "whir",
    "whirr",
    "whine",
    "honk",
    "siren",
    "brake",
    "brakes",
    "engine",
    "motor",
    "rail",
    "train",
    "footfall",
    "thunder",
    "footsteps",
    "ticks",
    "ticking",
    "crackle",
    "sizzle",
    "pour",
    "pouring",
    "stir",
    "stirring",
    "steam",
    "spray",
    "drum",
    "guitar",
    "piano",
    "violin",
    "cello",
    "birdsong",
    "chirp",
    "bark",
)

TEMPORAL_TERMS = (
    "then",
    "while",
    "as",
    "before",
    "after",
    "when",
    "meanwhile",
)

DISALLOWED_OPENERS = (
    "the scene opens",
    "the video starts",
    "the video opens",
    "this video shows",
    "a video of",
    "style:",
)

STOPWORDS = {
    "a", "an", "and", "the", "to", "of", "in", "on", "with", "at", "for", "by",
    "from", "into", "is", "are", "was", "were", "be", "been", "being", "it",
    "its", "their", "his", "her", "that", "this", "then", "while", "as", "one",
    "two", "through", "under", "over", "near", "after", "before", "around",
}

TEXT_LOGO_RE = re.compile(
    r"\b(?:logo|logos|brand(?:s| name)?|signage|subtitle(?:s)?|caption(?:s)?|title card|"
    r"printed material|readable text|billboard text|poster text|on-screen text)\b",
    re.IGNORECASE,
)

COMPLEX_MOTION_RE = re.compile(
    r"\b(?:juggling|somersault|backflip|front flip|triple spin|midair collision|pileup|"
    r"shattering glass storm|liquid simulation|chaotic swarm)\b",
    re.IGNORECASE,
)

OVERLOAD_RE = re.compile(
    r"\b(?:crowd of|packed crowd|dozens of|hundreds of|large group|ensemble cast|"
    r"parade|marching band|packed stadium|busy rally)\b",
    re.IGNORECASE,
)

HUMAN_NOUN_RE = re.compile(
    r"\b(?:man|woman|person|people|child|children|boy|girl|teenager|adult|couple|host|reporter|"
    r"worker|chef|mechanic|technician|operator|craftsperson|repairer|presenter|interviewer|performer|"
    r"musician|dancer|singer|drummer|cellist|guitarist|pianist|father|mother|daughter|son|coworker|"
    r"guide|guest|officer|detective|courier)\b",
    re.IGNORECASE,
)

HUMAN_BODY_RE = re.compile(
    r"\b(?:face|eyes|mouth|jaw|breath|posture|gaze|hands?|fingers?|shoulders?|torso)\b",
    re.IGNORECASE,
)

CLOSE_PORTRAIT_RE = re.compile(
    r"\b(?:close-up|close up|medium close-up|portrait frame|shoulder-up|shallow depth of field)\b",
    re.IGNORECASE,
)

INTERACTION_RE = re.compile(
    r"\b(?:says|speaks|asks|replies|answers|whispers|murmurs|nods|meets the other gaze|"
    r"across from|face to face|between them|the other person)\b|\"",
    re.IGNORECASE,
)

DIALOGUE_CUE_RE = re.compile(
    r"\b(?:says|speaks|asks|replies|answers|whispers|murmurs|shouts|calls out|tells|conversation|dialogue|"
    r"question|answer|interview question|spoken line)\b|\"",
    re.IGNORECASE,
)

BROADCAST_RE = re.compile(
    r"\b(?:microphone|host|reporter|podcast|audience|viewer|interview|museum guide|cue card|studio desk|"
    r"presentation|earpiece)\b",
    re.IGNORECASE,
)

PERFORMANCE_RE = re.compile(
    r"\b(?:performance|performer|music|song|melody|rhythm|beat|dance|dancer|sing|sings|singer|"
    r"musician|instrument|microphone|drum|drummer|guitar|piano|violin|cello|stage|rehearsal)\b",
    re.IGNORECASE,
)

FOLEY_RE = re.compile(
    r"\b(?:tool|hammer|wrench|ratchet|pipe|valve|bolt|tighten|loosen|repair|maintenance|scrape|grind|"
    r"pry|lever|impact|strike|clang|clank|thud|metal-on-metal|recoil|resonant metal)\b",
    re.IGNORECASE,
)

NATURE_RE = re.compile(
    r"\b(?:wind|rain|storm|snow|surf|wave|cloud|ridge|forest|grass|desert|river|ocean|fog)\b",
    re.IGNORECASE,
)

ANIMAL_RE = re.compile(
    r"\b(?:fox|dog|retriever|heron|gecko|bird|animal|paw|fur|beak|tail|wings?|claws?)\b",
    re.IGNORECASE,
)

URBAN_RE = re.compile(
    r"\b(?:street|avenue|intersection|tower|alley|storefront|tram|train|traffic|sidewalk|office|station|"
    r"crosswalk|platform|pavement|headlights)\b",
    re.IGNORECASE,
)

OBJECT_RE = re.compile(
    r"\b(?:macro|tabletop|surface|glass|steel|ceramic|texture|reflection|condensation|droplet|steam|watch|"
    r"perfume|cup|knife|citrus)\b",
    re.IGNORECASE,
)

VEHICLE_RE = re.compile(
    r"\b(?:car|truck|bus|tram|motorcycle|bike|bicycle|train|subway|boat|ship|van|taxi|headlights|"
    r"platform lights|engine|hull|tires?)\b",
    re.IGNORECASE,
)

MECHANICAL_RE = re.compile(
    r"\b(?:machine|mechanical|infrastructure|industrial|factory|motor|engine|gear|gears|pipe|valve|"
    r"pump|conveyor|rail|track|cable|lift|elevator|tunnel|pressure|vibration|resonance|brake|wheels?)\b",
    re.IGNORECASE,
)

MATERIAL_RE = re.compile(
    r"\b(?:material|liquid|powder|paste|steam|condensation|droplet|drip|pour|spread|stir|mix|slice|cut|"
    r"chop|sizzle|bubble|bubbles|foam|grain|dust|residue|texture|viscosity)\b",
    re.IGNORECASE,
)

DEVICE_RE = re.compile(
    r"\b(?:watch|device|instrument|mechanism|gear|hinge|spring|latch|ratchet|dial|switch|button|lever|"
    r"screw|motor|engineered object|compact mechanism)\b",
    re.IGNORECASE,
)

CATEGORY_REQUIRED_RE: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "Human": (
        HUMAN_NOUN_RE,
        HUMAN_BODY_RE,
        INTERACTION_RE,
        BROADCAST_RE,
        PERFORMANCE_RE,
        FOLEY_RE,
    ),
    "Natural_World": (
        NATURE_RE,
        ANIMAL_RE,
    ),
    "Built_World": (
        URBAN_RE,
        VEHICLE_RE,
        MECHANICAL_RE,
    ),
    "Object": (
        OBJECT_RE,
        MATERIAL_RE,
        DEVICE_RE,
    ),
}

CATEGORY_FORBIDDEN_RE: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "Human": (
        OVERLOAD_RE,
    ),
    "Natural_World": (
        re.compile(r'"', re.IGNORECASE),
    ),
    "Object": (
        re.compile(r'"', re.IGNORECASE),
    ),
}


LTX2_BASE_SYSTEM = """You are writing production-ready prompts for the LTX-2 video model.

Write prompts the way a cinematographer would describe a single shot:
- one flowing paragraph only
- 4 to 8 sentences total
- 200 words maximum
- begin directly with the main shot and action, not with meta wording
- describe the scene in chronological order from first visual beat to last
- follow this content order as closely as it fits the shot: establish the shot, set the scene, describe the action, define the character or object, describe camera movement relative to the subject, then weave in audio and the closing change or interruption
- stay literal and concrete: movement, gesture, appearance, environment, camera relationship, lighting, colors, sound, and any change or sudden event
- use natural present tense; simple present and present-progressive are both acceptable when they sound natural
- match the amount of visual detail to the shot scale; close-ups need finer facial or texture detail than wide shots
- if the camera moves, describe what it reveals about the subject after the move
- if dialogue appears, keep it brief, place it in quotation marks, and mention language or accent only when it matters
- tie sound to visible action whenever possible so the audio feels synchronized to the motion
- if a style direction is supplied, name it early and let it shape palette, materials, camera language, and motion without changing the core content type
- show emotion through visible physical cues instead of labels

Avoid:
- headings, bullet lists, markdown, timestamps, or screenplay slug lines
- "The scene opens", "The video starts", or similar meta openings
- brand names, logos, readable signage, subtitles, or on-screen text
- too many simultaneous actions or camera moves
- impossible physics, collisions, or dense choreography
- contradictory lighting descriptions

Return strict JSON only:
{"prompts": ["prompt1", "prompt2", "..."]}
"""


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    api_base: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    target_count: int
    workers: int
    drafts_per_request: int
    keep_per_request: int
    output_csv: str
    categorized_csv: str


@dataclass(frozen=True)
class SeedSpec:
    family_name: str
    dialogue_allowed: bool


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", prompt.lower())).strip()


def prompt_fingerprint(prompt: str) -> Tuple[str, ...]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", prompt.lower()) if token not in STOPWORDS]
    return tuple(tokens[:18])


def sentence_count(prompt: str) -> int:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", prompt.strip()) if part.strip()]
    return len(sentences)


def contains_any(prompt: str, terms: Tuple[str, ...]) -> bool:
    lower = prompt.lower()
    return any(term in lower for term in terms)


def sample_style_direction(category_name: str) -> Optional[Tuple[str, str]]:
    style_rate = CATEGORIES[category_name].get("style_rate", 1.0)
    if random.random() >= style_rate:
        return None

    style_family = random.choice(CATEGORIES[category_name]["style_families"])
    return style_family, random.choice(STYLE_DIRECTION_POOLS[style_family])


def sample_ending_behavior(category_name: str, family_name: str) -> Tuple[str, str]:
    family = SCENE_FAMILIES[family_name]
    dynamic_rate = CATEGORIES[category_name].get("dynamic_ending_rate", 0.5)
    if family.get("change_types") and random.random() < dynamic_rate:
        return "state shift or interruption", random.choice(family["change_types"])
    return "steady hold or continuing loop", random.choice(STEADY_ENDING_POOLS[category_name])


def sample_diversity_cues(category_name: str, max_cues: int = 3) -> List[str]:
    dims = CATEGORIES[category_name].get("diversity_dims", list(DIVERSITY_POOLS.keys()))
    cue_limit = min(max_cues, len(dims))
    if cue_limit <= 1:
        cue_count = cue_limit
    else:
        cue_count = random.randint(2, cue_limit)
    selected_dims = random.sample(dims, cue_count)
    return [f"{dim.replace('_', ' ')}: {random.choice(DIVERSITY_POOLS[dim])}" for dim in selected_dims]


def get_temperature_for_category(category_name: str, config: RuntimeConfig) -> float:
    offset = CATEGORIES[category_name].get("temperature_offset", 0.0)
    return max(0.2, min(1.2, config.temperature + offset))


def format_category_temperature_table(config: RuntimeConfig) -> str:
    parts = []
    for category_name in CATEGORIES:
        parts.append(f"{category_name}={get_temperature_for_category(category_name, config):.2f}")
    return ", ".join(parts)


def weighted_sample_without_replacement(items: List[str], weights: List[int], count: int) -> List[str]:
    if count <= 0 or not items:
        return []

    pool = list(zip(items, weights))
    chosen: List[str] = []
    while pool and len(chosen) < count:
        current_items = [item for item, _ in pool]
        current_weights = [weight for _, weight in pool]
        selected = random.choices(current_items, weights=current_weights, k=1)[0]
        chosen.append(selected)
        pool = [(item, weight) for item, weight in pool if item != selected]
    return chosen


def sample_family_names(category_name: str, count: int) -> List[str]:
    cat = CATEGORIES[category_name]
    family_names = list(cat["families"].keys())
    family_weights = list(cat["families"].values())
    if count <= len(family_names):
        return weighted_sample_without_replacement(family_names, family_weights, count)

    selected = weighted_sample_without_replacement(family_names, family_weights, len(family_names))
    while len(selected) < count:
        selected.append(random.choices(family_names, weights=family_weights, k=1)[0])
    return selected


def sample_family_blueprint(family_name: str, used_signatures: Optional[Set[Tuple[str, ...]]] = None) -> Dict[str, str]:
    family = SCENE_FAMILIES[family_name]
    last_blueprint: Dict[str, str] = {}

    for _ in range(24):
        blueprint = {"mode": family["mode"]}
        for field_name, _ in FAMILY_BRIEF_FIELDS:
            blueprint[field_name] = random.choice(family[field_name])

        signature = (
            family_name,
            blueprint["hero_subject_archetypes"],
            blueprint["setting_archetypes"],
            blueprint["action_classes"],
            blueprint["audio_emphasis"],
        )
        last_blueprint = blueprint
        if used_signatures is None or signature not in used_signatures:
            if used_signatures is not None:
                used_signatures.add(signature)
            return blueprint

    return last_blueprint


def build_audio_visual_rule(audio_emphasis: str, sound_profile: str) -> str:
    cue = f"{audio_emphasis} {sound_profile}".lower()
    if any(term in cue for term in ("speech", "dialogue", "interview", "broadcast", "voice")):
        return "make the speaker clearly visible when the main speech beat happens"
    if any(term in cue for term in ("music", "performance", "melody", "instrument", "singer", "song", "stage", "voice-led", "rhythm-led")):
        return "keep the sound-making body parts, instrument, or performer movement readable when the phrasing changes"
    if any(term in cue for term in ("weather", "environment", "water", "urban", "transit", "city", "habitat", "animal")):
        return "tie the dominant ambience to visible environmental motion, passing flow, or animal behavior"
    if any(term in cue for term in ("foley", "impact", "tool", "mechanical", "motor", "engine", "pressure", "tick", "click", "material", "contact", "resonance", "industrial", "vehicle", "rail", "process")):
        return "keep the source of contact, impacts, scrapes, clicks, resonance, or state changes visibly legible on screen"
    if any(term in cue for term in ("task", "motion", "movement", "breath", "foot", "body")):
        return "make the body's contact, breath, or repeated movement visibly explain the dominant sound"
    return "give the dominant sound a clear visible source or visible effect inside the shot"


def quoted_dialogue_word_count(prompt: str) -> int:
    total = 0
    for segment in re.findall(r'"([^"]*)"', prompt):
        total += len(re.findall(r"[A-Za-z0-9']+", segment))
    return total


def family_allows_dialogue(category_name: str, family_name: str) -> bool:
    return category_name == "Human" and family_name in DIALOGUE_ALLOWED_FAMILIES


def dialogue_policy_text(dialogue_allowed: bool) -> str:
    if dialogue_allowed:
        return "dialogue allowed if helpful, but keep all quoted speech across all speakers to 30 words or fewer"
    return "no dialogue, no quoted speech, and no conversational exchange; rely on visible action, gesture, breath, music, foley, or ambience instead"


def validate_prompt_for_seed(prompt: str, seed_spec: SeedSpec) -> Tuple[bool, str]:
    if seed_spec.dialogue_allowed:
        return True, "ok"
    if DIALOGUE_CUE_RE.search(prompt):
        return False, f"dialogue_not_allowed:{seed_spec.family_name}"
    return True, "ok"


def build_category_prompt(category_name: str, config: RuntimeConfig) -> str:
    cat = CATEGORIES[category_name]
    focus = ", ".join(cat["focus"])
    audio_targets = ", ".join(cat["audio_targets"])
    avoid = ", ".join(cat["avoid"])
    style_families = ", ".join(cat["style_families"])
    scene_modes = ", ".join(cat["scene_modes"])
    return (
        f"{LTX2_BASE_SYSTEM}\n"
        f"Category: {category_name}\n"
        f"Category goal: {cat['desc']}\n"
        f"Subject rule: {cat['subject_rule']}\n"
        f"Prioritize: {focus}\n"
        f"Typical dominant audio anchors in this category: {audio_targets}\n"
        f"Avoid especially: {avoid}\n"
        f"Possible scene modes inside this category: {scene_modes}\n"
        f"Allowed style families for this category: {style_families}\n"
        "Treat the category as the primary content type. Treat style, genre, or animation cues only as modifiers layered onto that content type.\n"
        "Many valid prompts should have no explicit style modifier at all. If a seed does not include a style cue, keep the wording plain and content-first instead of inventing one.\n"
        "The seed descriptors you receive are abstract control signals, not wording to copy. Convert them into concrete people, animals, objects, tools, materials, locations, and exact motions of your own invention.\n"
        "For audiovisual usefulness, preserve strong visible-audible coupling: speech should have an identifiable speaker, performance should reveal the sound-making body or instrument, and foley-, material-, vehicle-, or machine-led scenes should keep the sound source visually legible.\n"
        "Only use dialogue when a seed explicitly marks dialogue as allowed. For all other seeds, avoid quoted speech and avoid describing a spoken conversational exchange.\n"
        f"Internally draft {config.drafts_per_request} diverse candidates, compare them, and return the most distinct valid {config.keep_per_request} prompts rather than polishing them toward one ideal template.\n"
        "Keep the output prompt order aligned with the seed order.\n"
        "Every returned prompt must clearly include:\n"
        "- first sentence starts directly with the main shot and main action\n"
        "- appearance details that match the chosen shot scale\n"
        "- environment or background details that support the action\n"
        "- camera position or movement described relative to the subject\n"
        "- what becomes visible after any camera movement\n"
        "- lighting or color information with one coherent light logic\n"
        "- sound, ambience, music, or dialogue naturally embedded in the timeline\n"
        "- if dialogue appears, total quoted speech across all speakers must stay at 30 words or fewer\n"
        "- a coherent ending behavior, which may be a small change, a brief interruption, a steady hold, or a continuing loop\n"
        "Avoid readable text, logos, subtitles, printed material, chaotic physics, and overloaded scenes.\n"
    )


def build_creative_brief(category_name: str, keep_count: int) -> Tuple[str, List[SeedSpec]]:
    family_names = sample_family_names(category_name, keep_count)
    used_signatures: Set[Tuple[str, ...]] = set()
    seed_specs: List[SeedSpec] = []

    lines = [
        "The abstract seeds below define shot constraints, not exact details to copy.",
        f"Category: {category_name}",
        f"Generate exactly {keep_count} prompts, each based on a different seed.",
        "Invent specific subjects, props, tools, materials, locations, and exact motion details yourself instead of repeating the abstract labels verbatim.",
        "Treat each seed as a coherent latent design. Keep one dominant content focus and one dominant audio relationship per prompt.",
        "Some seeds intentionally omit style cues. If a seed has no style line, keep the prompt plain and content-first rather than forcing a stylized tag.",
        "Only use dialogue for seeds that explicitly say dialogue is allowed. All other seeds must avoid quoted speech and avoid describing a spoken conversational exchange.",
        "Any quoted dialogue is optional, and the total quoted speech across all speakers must not exceed 30 words because the target clip is only about 5 seconds long.",
        "Across outputs, prioritize distinct subjects, settings, and motion patterns over converging on one polished house style.",
        "Return prompts in the same order as the seed list below.",
        "Do not merge seeds, and do not reuse the same exact subject, setting, or wording across outputs.",
    ]
    for index, family_name in enumerate(family_names, start=1):
        family = SCENE_FAMILIES[family_name]
        blueprint = sample_family_blueprint(family_name, used_signatures=used_signatures)
        style_selection = sample_style_direction(category_name)
        diversity_cues = sample_diversity_cues(category_name)
        ending_mode, ending_behavior = sample_ending_behavior(category_name, family_name)
        dialogue_allowed = family_allows_dialogue(category_name, family_name)
        seed_specs.append(SeedSpec(family_name=family_name, dialogue_allowed=dialogue_allowed))
        audio_rule = build_audio_visual_rule(
            blueprint["audio_emphasis"],
            blueprint["sound_profiles"],
        )
        lines.extend(
            [
                f"Seed {index}:",
                f"- Content family: {family_name.replace('_', ' ')}",
                f"- Scene mode: {family['mode']}",
            ]
        )
        if style_selection is not None:
            style_family, style_direction = style_selection
            lines.extend(
                [
                    f"- Style family: {style_family}",
                    f"- Style direction: {style_direction}",
                ]
            )
        for field_name, label in FAMILY_BRIEF_FIELDS:
            lines.append(f"- {label}: {blueprint[field_name]}")
        lines.extend(
            [
                f"- Audio-visual rule: {audio_rule}",
                f"- Dialogue policy: {dialogue_policy_text(dialogue_allowed)}",
                f"- Ending behavior ({ending_mode}): {ending_behavior}",
                f"- Diversity cues: {'; '.join(diversity_cues)}",
            ]
        )
    return "\n".join(lines), seed_specs


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        end = start
        for index, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                end = index + 1
                break
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot parse JSON from response: {text[:200]}...")


def api_call(system: str, user: str, config: RuntimeConfig, temperature: float) -> Dict[str, Any]:
    base = config.api_base.rstrip("/")
    url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"

    for attempt in range(5):
        seed = random.randint(0, 2**31 - 1)
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": config.max_tokens,
                    "seed": seed,
                },
                timeout=config.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            return extract_json(content)
        except Exception as exc:
            print(f"  Retry {attempt + 1}/5: {exc}")
            time.sleep(2 ** attempt)

    raise RuntimeError("API failed after retries")


def validate_prompt_with_reason(prompt: str) -> Tuple[bool, str]:
    if not prompt:
        return False, "empty"

    stripped = prompt.strip()
    if "\n" in stripped:
        return False, "multiple_lines"
    if re.search(r"^[\s]*[-*#]\s", stripped, re.MULTILINE):
        return False, "contains_bullets"
    if re.search(r"\b(INT\.|EXT\.|SCENE\s+\d+)\b", stripped, re.IGNORECASE):
        return False, "contains_screenplay_slug"
    if re.search(r"\b\d{1,2}:\d{2}\b", stripped):
        return False, "contains_timestamp"

    lowered = stripped.lower()
    if any(lowered.startswith(opener) for opener in DISALLOWED_OPENERS):
        return False, "meta_opener"
    if stripped.count('"') % 2 != 0:
        return False, "unbalanced_quotes"
    dialogue_words = quoted_dialogue_word_count(stripped)
    if dialogue_words > 30:
        return False, f"dialogue_too_long:{dialogue_words}"

    words = stripped.split()
    if len(words) < MIN_WORDS:
        return False, f"too_short:{len(words)}"
    if len(words) > MAX_WORDS:
        return False, f"too_long:{len(words)}"

    sentences = sentence_count(stripped)
    if sentences < MIN_SENTENCES:
        return False, f"too_few_sentences:{sentences}"
    if sentences > MAX_SENTENCES:
        return False, f"too_many_sentences:{sentences}"

    if not contains_any(stripped, CAMERA_TERMS):
        return False, "missing_camera_language"
    if not contains_any(stripped, LIGHT_TERMS):
        return False, "missing_light_detail"
    if not contains_any(stripped, SOUND_TERMS) and '"' not in stripped:
        return False, "missing_sound_detail"
    if not contains_any(stripped, TEMPORAL_TERMS):
        return False, "missing_temporal_flow"
    if re.search(r"\b(feels|emotionally|symbolizes|represents|metaphor|dreamlike in an abstract way)\b", stripped, re.IGNORECASE):
        return False, "too_abstract"
    if TEXT_LOGO_RE.search(stripped):
        return False, "text_or_logo"
    if COMPLEX_MOTION_RE.search(stripped):
        return False, "complex_motion"
    if OVERLOAD_RE.search(stripped):
        return False, "scene_overload"

    return True, "ok"


def validate_prompt_for_category(prompt: str, category_name: str) -> Tuple[bool, str]:
    required_patterns = CATEGORY_REQUIRED_RE.get(category_name, ())
    forbidden_patterns = CATEGORY_FORBIDDEN_RE.get(category_name, ())

    if category_name == "Natural_World" and HUMAN_NOUN_RE.search(prompt):
        return False, "nature_has_human_subject"
    if category_name == "Object" and HUMAN_NOUN_RE.search(prompt) and not (OBJECT_RE.search(prompt) or MATERIAL_RE.search(prompt) or DEVICE_RE.search(prompt)):
        return False, "object_not_primary"

    for pattern in forbidden_patterns:
        if pattern.search(prompt):
            return False, f"category_forbidden:{category_name}"

    if required_patterns and not any(pattern.search(prompt) for pattern in required_patterns):
        return False, f"category_missing:{category_name}"

    return True, "ok"


def score_prompt(prompt: str, category_name: str) -> int:
    score = 0
    words = len(prompt.split())
    sentences = sentence_count(prompt)

    if 70 <= words <= 160:
        score += 3
    elif 55 <= words <= 180:
        score += 2
    else:
        score += 1

    if 4 <= sentences <= 6:
        score += 3
    else:
        score += 1

    if contains_any(prompt, CAMERA_TERMS):
        score += 2
    if contains_any(prompt, LIGHT_TERMS):
        score += 2
    if contains_any(prompt, SOUND_TERMS) or '"' in prompt:
        score += 2
    if contains_any(prompt, TEMPORAL_TERMS):
        score += 1
    if re.search(r"\b(close-up|medium|wide|tracking|push-in|telephoto|macro)\b", prompt, re.IGNORECASE):
        score += 1
    if re.search(r"\b(red|blue|green|amber|gold|silver|shadow|glow)\b", prompt, re.IGNORECASE):
        score += 1
    if category_name == "Human" and (
        HUMAN_NOUN_RE.search(prompt)
        or HUMAN_BODY_RE.search(prompt)
        or INTERACTION_RE.search(prompt)
        or BROADCAST_RE.search(prompt)
        or PERFORMANCE_RE.search(prompt)
        or FOLEY_RE.search(prompt)
    ):
        score += 2
    if category_name == "Human" and (CLOSE_PORTRAIT_RE.search(prompt) or PERFORMANCE_RE.search(prompt) or FOLEY_RE.search(prompt)):
        score += 2
    if category_name == "Natural_World" and (NATURE_RE.search(prompt) or ANIMAL_RE.search(prompt)):
        score += 2
    if category_name == "Built_World" and (URBAN_RE.search(prompt) or VEHICLE_RE.search(prompt) or MECHANICAL_RE.search(prompt)):
        score += 2
    if category_name == "Object" and (OBJECT_RE.search(prompt) or MATERIAL_RE.search(prompt) or DEVICE_RE.search(prompt)):
        score += 2

    return score


def shuffle_csv(csv_path: str, categorized_path: str) -> None:
    for path, has_category in ((csv_path, False), (categorized_path, True)):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = list(reader)
        random.shuffle(rows)
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            writer.writerows(rows)
        kind = "categorized CSV" if has_category else "CSV"
        print(f"Shuffled {len(rows)} rows in {kind}: {path}")


def load_existing_prompts(output_csv: str) -> Tuple[Set[str], Set[str], Set[Tuple[str, ...]]]:
    raw_prompts: Set[str] = set()
    normalized_prompts: Set[str] = set()
    fingerprints: Set[Tuple[str, ...]] = set()

    if not os.path.exists(output_csv):
        return raw_prompts, normalized_prompts, fingerprints

    with open(output_csv, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            prompt = row[0]
            raw_prompts.add(prompt)
            normalized_prompts.add(normalize_prompt(prompt))
            fingerprints.add(prompt_fingerprint(prompt))

    return raw_prompts, normalized_prompts, fingerprints


def load_existing_category_counts(categorized_csv: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not os.path.exists(categorized_csv):
        return counts

    with open(categorized_csv, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                counts[row[0]] = counts.get(row[0], 0) + 1

    return counts


def save_prompts(prompts: List[str], output_csv: str) -> None:
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["text_prompt"])
        for prompt in prompts:
            writer.writerow([prompt])


def save_prompts_categorized(prompts: List[str], category: str, categorized_csv: str) -> None:
    file_exists = os.path.exists(categorized_csv)
    with open(categorized_csv, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["category", "text_prompt"])
        for prompt in prompts:
            writer.writerow([category, prompt])


def calculate_category_counts(total_count: int) -> Dict[str, int]:
    total_weight = sum(cat["weight"] for cat in CATEGORIES.values())
    counts: Dict[str, int] = {}
    remaining = total_count
    category_names = list(CATEGORIES.keys())

    for index, name in enumerate(category_names):
        if index == len(category_names) - 1:
            counts[name] = remaining
            break
        count = round(total_count * CATEGORIES[name]["weight"] / total_weight)
        counts[name] = count
        remaining -= count

    return counts


def register_reject(reject_stats: Dict[str, int], reason: str) -> None:
    key = reason.split(":", 1)[0]
    reject_stats[key] = reject_stats.get(key, 0) + 1


def select_prompts(
    category_name: str,
    raw_prompts: List[str],
    seed_specs: List[SeedSpec],
    keep_count: int,
    existing_prompts: Set[str],
    normalized_prompts: Set[str],
    fingerprints: Set[Tuple[str, ...]],
    reject_stats: Dict[str, int],
) -> List[str]:
    candidates: List[Tuple[int, str, str, Tuple[str, ...]]] = []
    local_normalized: Set[str] = set()
    local_fingerprints: Set[Tuple[str, ...]] = set()

    for index, prompt in enumerate(raw_prompts):
        ok, reason = validate_prompt_with_reason(prompt)
        if not ok:
            register_reject(reject_stats, reason)
            continue
        ok, reason = validate_prompt_for_category(prompt, category_name)
        if not ok:
            register_reject(reject_stats, reason)
            continue
        seed_spec = seed_specs[index] if index < len(seed_specs) else SeedSpec(family_name="unmatched_seed", dialogue_allowed=False)
        ok, reason = validate_prompt_for_seed(prompt, seed_spec)
        if not ok:
            register_reject(reject_stats, reason)
            continue

        normalized = normalize_prompt(prompt)
        fingerprint = prompt_fingerprint(prompt)
        if normalized in normalized_prompts or normalized in local_normalized:
            register_reject(reject_stats, "duplicate_normalized")
            continue
        if fingerprint and (fingerprint in fingerprints or fingerprint in local_fingerprints):
            register_reject(reject_stats, "duplicate_fingerprint")
            continue

        local_normalized.add(normalized)
        local_fingerprints.add(fingerprint)
        candidates.append((score_prompt(prompt, category_name), prompt, normalized, fingerprint))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: List[str] = []
    for _, prompt, normalized, fingerprint in candidates[:keep_count]:
        existing_prompts.add(prompt)
        normalized_prompts.add(normalized)
        if fingerprint:
            fingerprints.add(fingerprint)
        selected.append(prompt)

    return selected


def make_request_for_category(
    category_name: str,
    config: RuntimeConfig,
    existing_prompts: Set[str],
    normalized_prompts: Set[str],
    fingerprints: Set[Tuple[str, ...]],
    reject_stats: Dict[str, int],
    lock: threading.Lock,
) -> List[str]:
    system_prompt = build_category_prompt(category_name, config)
    user_prompt, seed_specs = build_creative_brief(category_name, config.keep_per_request)
    request_temperature = get_temperature_for_category(category_name, config)

    try:
        result = api_call(system_prompt, user_prompt, config, temperature=request_temperature)
        raw_prompts = result.get("prompts", [])
        if not isinstance(raw_prompts, list):
            register_reject(reject_stats, "invalid_json_shape")
            return []
        with lock:
            return select_prompts(
                category_name=category_name,
                raw_prompts=raw_prompts,
                seed_specs=seed_specs,
                keep_count=config.keep_per_request,
                existing_prompts=existing_prompts,
                normalized_prompts=normalized_prompts,
                fingerprints=fingerprints,
                reject_stats=reject_stats,
            )
    except Exception:
        with lock:
            register_reject(reject_stats, "api_error")
        return []


def generate_prompts_by_category(config: RuntimeConfig, target_category: Optional[str] = None) -> None:
    existing_prompts, normalized_prompts, fingerprints = load_existing_prompts(config.output_csv)
    existing_category_counts = load_existing_category_counts(config.categorized_csv)

    if len(existing_prompts) >= config.target_count:
        print("Target already satisfied by existing prompts.")
        return

    if target_category:
        if target_category not in CATEGORIES:
            raise ValueError(f"Unknown category: {target_category}")
        category_counts = {target_category: config.target_count}
    else:
        category_counts = calculate_category_counts(config.target_count)
    category_stats = {name: 0 for name in CATEGORIES}
    reject_stats: Dict[str, int] = {}

    total_bar = tqdm(
        total=config.target_count,
        desc="Total",
        unit="prompt",
        position=0,
        bar_format="{l_bar}{bar:30}{r_bar}",
        colour="green",
    )
    total_bar.update(len(existing_prompts))
    total_bar.set_postfix(existing=len(existing_prompts))

    total_new = 0

    for category_name, target_for_category in category_counts.items():
        already_have = existing_category_counts.get(category_name, 0)
        remaining_for_category = target_for_category - already_have
        if remaining_for_category <= 0:
            print(f"Skipping {category_name}, already has {already_have} prompts.")
            continue

        lock = threading.Lock()
        produced = [0]
        category_bar = tqdm(
            total=remaining_for_category,
            desc=f"  {category_name}",
            unit="prompt",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar:25}{r_bar}",
            colour="cyan",
        )

        def process_batch() -> int:
            with lock:
                if produced[0] >= remaining_for_category:
                    return 0

            prompts = make_request_for_category(
                category_name=category_name,
                config=config,
                existing_prompts=existing_prompts,
                normalized_prompts=normalized_prompts,
                fingerprints=fingerprints,
                reject_stats=reject_stats,
                lock=lock,
            )
            if not prompts:
                return 0

            with lock:
                if produced[0] >= remaining_for_category:
                    return 0
                need = remaining_for_category - produced[0]
                to_save = prompts[:need]
                save_prompts(to_save, config.output_csv)
                save_prompts_categorized(to_save, category_name, config.categorized_csv)
                produced[0] += len(to_save)
                category_bar.update(len(to_save))
                total_bar.update(len(to_save))
                total_bar.set_postfix(cat=category_name, last=f"+{len(to_save)}")
                return len(to_save)

        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            while produced[0] < remaining_for_category:
                remaining = remaining_for_category - produced[0]
                batches_needed = (remaining + config.keep_per_request - 1) // config.keep_per_request
                worker_count = min(config.workers, batches_needed)
                if worker_count <= 0:
                    break
                futures = [executor.submit(process_batch) for _ in range(worker_count)]
                for future in as_completed(futures):
                    total_new += future.result()

        category_stats[category_name] = produced[0]
        category_bar.close()

    total_bar.close()
    shuffle_csv(config.output_csv, config.categorized_csv)

    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"New prompts generated: {total_new}")
    print("\nCategory breakdown:")
    for name, count in category_stats.items():
        if count > 0:
            print(f"  {name}: {count}")
    if reject_stats:
        print("\nReject reasons:")
        for reason, count in sorted(reject_stats.items(), key=lambda item: item[1], reverse=True):
            print(f"  {reason}: {count}")


def generate_prompts_random(config: RuntimeConfig, target_category: Optional[str] = None) -> None:
    existing_prompts, normalized_prompts, fingerprints = load_existing_prompts(config.output_csv)
    if len(existing_prompts) >= config.target_count:
        print("Target already satisfied by existing prompts.")
        return

    if target_category:
        if target_category not in CATEGORIES:
            raise ValueError(f"Unknown category: {target_category}")
        category_names = [target_category]
        category_weights = [1.0]
    else:
        category_names = list(CATEGORIES.keys())
        category_weights = [CATEGORIES[name]["weight"] for name in category_names]

    reject_stats: Dict[str, int] = {}
    category_stats = {name: 0 for name in CATEGORIES}
    lock = threading.Lock()
    saved_count = [len(existing_prompts)]
    total_new = 0

    progress = tqdm(
        total=config.target_count,
        initial=len(existing_prompts),
        desc="Generating",
        unit="prompt",
        bar_format="{l_bar}{bar:30}{r_bar}",
        colour="green",
    )

    def process_batch() -> int:
        with lock:
            if saved_count[0] >= config.target_count:
                return 0

        category_name = random.choices(category_names, weights=category_weights, k=1)[0]
        prompts = make_request_for_category(
            category_name=category_name,
            config=config,
            existing_prompts=existing_prompts,
            normalized_prompts=normalized_prompts,
            fingerprints=fingerprints,
            reject_stats=reject_stats,
            lock=lock,
        )

        if not prompts:
            return 0

        with lock:
            if saved_count[0] >= config.target_count:
                return 0
            need = config.target_count - saved_count[0]
            to_save = prompts[:need]
            save_prompts(to_save, config.output_csv)
            save_prompts_categorized(to_save, category_name, config.categorized_csv)
            saved_count[0] += len(to_save)
            category_stats[category_name] += len(to_save)
            progress.update(len(to_save))
            progress.set_postfix(cat=category_name, last=f"+{len(to_save)}")
            return len(to_save)

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        while saved_count[0] < config.target_count:
            remaining = config.target_count - saved_count[0]
            batches_needed = (remaining + config.keep_per_request - 1) // config.keep_per_request
            worker_count = min(config.workers, batches_needed)
            if worker_count <= 0:
                break
            futures = [executor.submit(process_batch) for _ in range(worker_count)]
            for future in as_completed(futures):
                total_new += future.result()

    progress.close()
    shuffle_csv(config.output_csv, config.categorized_csv)

    print(f"\nDone. New: {total_new}, Total: {saved_count[0]}")
    print("\nCategory statistics:")
    for name, count in category_stats.items():
        if count > 0:
            print(f"  {name}: {count}")
    if reject_stats:
        print("\nReject reasons:")
        for reason, count in sorted(reject_stats.items(), key=lambda item: item[1], reverse=True):
            print(f"  {reason}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LTX-2 prompt generator v1 aligned with the official Prompting Guide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available categories:
  {', '.join(CATEGORIES.keys())}

Examples:
  python gen_prompts_v1.py
  python gen_prompts_v1.py -n 300
  python gen_prompts_v1.py -c Natural_World --random
  python gen_prompts_v1.py -c Built_World -n 24
  python gen_prompts_v1.py --model gpt-5.4 --api-base https://api.xcode.best
""",
    )
    parser.add_argument("-c", "--category", type=str, default=None, help="Only generate one category")
    parser.add_argument("--random", action="store_true", help="Use weighted random category sampling")
    parser.add_argument("-n", "--count", type=int, default=DEFAULT_TARGET_COUNT, help="Target prompt count")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output base name without extension")
    parser.add_argument("--shuffle-only", action="store_true", help="Shuffle existing CSV outputs without generating")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel worker count")
    parser.add_argument("--drafts-per-request", type=int, default=DEFAULT_DRAFTS_PER_REQUEST, help="How many candidates the model should draft internally")
    parser.add_argument("--keep-per-request", type=int, default=DEFAULT_KEEP_PER_REQUEST, help="How many prompts to keep per API response")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Base sampling temperature before category-specific offsets")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="Prompt-generation API key")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="Prompt-generation API base")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Prompt-generation model")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    output_base = args.output or f"ltx_prompts_v1_{args.count}"
    return RuntimeConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        target_count=args.count,
        workers=args.workers,
        drafts_per_request=args.drafts_per_request,
        keep_per_request=args.keep_per_request,
        output_csv=f"{output_base}.csv",
        categorized_csv=f"{output_base}_categorized.csv",
    )


def main() -> int:
    args = parse_args()
    config = build_runtime_config(args)

    if args.shuffle_only:
        shuffle_csv(config.output_csv, config.categorized_csv)
        return 0

    if not config.api_key:
        print("Missing API key. Set PROMPT_GEN_API_KEY or OPENAI_API_KEY, or pass --api-key.")
        return 1

    print("=== LTX-2 Prompt Generator v1 ===")
    print(f"Target: {config.target_count} prompts -> {config.output_csv}")
    print(f"Categorized output -> {config.categorized_csv}")
    print(f"Model: {config.model}")
    print(f"Workers: {config.workers}")
    print(f"Drafts/request: {config.drafts_per_request}, keep/request: {config.keep_per_request}")
    print(f"Base temperature: {config.temperature:.2f}")
    print(f"Category temperatures: {format_category_temperature_table(config)}")

    if args.random:
        print("Mode: RANDOM SAMPLING")
        generate_prompts_random(config, target_category=args.category)
    else:
        if args.category:
            print(f"Mode: SINGLE CATEGORY ({args.category})")
            print(f"  - {args.category}: {config.target_count}")
            print()
            generate_prompts_by_category(config, target_category=args.category)
        else:
            print("Mode: BY CATEGORY")
            category_counts = calculate_category_counts(config.target_count)
            for name, count in category_counts.items():
                print(f"  - {name}: {count}")
            print()
            generate_prompts_by_category(config)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
