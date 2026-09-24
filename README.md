# SignLink

Video calls with real-time ASL fingerspelling to text.

Built in 36-ish hours at BoilerMake X (Purdue), January 2023.

## What it does

SignLink is a video conferencing app built for calls between a deaf or hard-of-hearing
person and someone who doesn't know ASL. One participant fingerspells; the other side of
the call sees the letters as text overlaid on the video, no interpreter required.

The translation is letter-by-letter, not sentence-level. It reads hand shape from a
fixed set of geometric rules, not a trained model. We're calling that out directly
instead of dressing it up: it's a foundational piece, not a finished translator.

## How it works

- WebRTC video call, negotiated with PeerJS — one peer creates a room, shares the room
  ID, the other joins.
- MediaPipe Hands runs on the remote video stream in the browser and returns 21 hand
  landmarks per frame.
- Landmarks get re-centered on the wrist (landmark 0), then vector algebra and
  trigonometry turn triples of landmarks into joint angles, summed per finger (thumb,
  index, middle, ring, pinky).
- Those five angle sums (plus an index-to-middle fingertip distance, for telling V from
  U and K from H) get checked against a hand-coded if/else tree in `app.html` that maps
  shapes to letters.
- The matched letter is drawn straight onto a canvas over the remote video feed.

Tech: HTML5/CSS/JS, WebRTC, PeerJS, MediaPipe, OpenCV.

## Try it locally

Serve the repository from its root (the camera API needs a secure context; `localhost`
counts) and open the landing page:

```sh
python3 -m http.server 8000
```

Then visit <http://localhost:8000/>. Open the app in two browser windows to try the
room flow. Allow camera and microphone access. The app loads MediaPipe and PeerJS from
public CDNs, so those services and an internet connection are required; this is a
hackathon prototype rather than a hosted calling service.

## Prototype

The real matcher is a hand-coded threshold tree tuned by eye during the hackathon —
there's no dataset or accuracy number behind it, so there's nothing to chart. Instead
`prototype/asl_prototype.py` draws the recognition pipeline itself: camera frame ->
MediaPipe's 21 hand landmarks -> joint angles via vector algebra -> match against the
hand-coded letter table -> letter -> caption in the call. The landmark topology and
the angle formula are real (the same `angle()` math as `app.html`), applied to one
stated toy pose (a static "L" handshape) to show the mechanism, not a measurement.
Static poses only — motion letters J and Z need a trajectory across frames and are
out of scope for this pipeline.

Regenerate it locally (Python 3 with Matplotlib and NumPy):

```
python3 -m pip install matplotlib numpy
python3 prototype/asl_prototype.py
```

![SignLink recognition pipeline: camera frame to on-screen letter](https://vircgxpcwyvniemqmdyi.supabase.co/storage/v1/object/public/media/writing/SignLink/recognition_flow.png)

## Team

Arya Garg, [Reeve Prinson Fernandes](https://github.com/ReeveFernandes), [Arleen Monteiro](https://www.linkedin.com/in/arleen-monteiro-665481258/).

This repo is a copy of the team's original repo, kept with its original commit
history: Reeve built the video-calling and MediaPipe gesture-matching pipeline;
Arleen built the landing page and styling. I was the third person on the team —
my contribution isn't visible in this repo's git history, so I'm crediting the code
to the people who actually wrote it rather than claiming lines that aren't mine.
Originally hacked together in [ReeveFernandes/SignLink](https://github.com/ReeveFernandes/SignLink).

## Links

- Devpost project: https://devpost.com/software/signlink
- Writeup: https://aryagarg23.com/writing/signlink
- Site: https://aryagarg23.com
- Devpost profile: https://devpost.com/Aryagarg23

## More hackathon builds

- [Gyrus](https://github.com/Aryagarg23/Gyrus) — agentic browser that supports curiosity instead of replacing it (WeaveHacks 2025)
- [WhiteBox](https://github.com/Aryagarg23/WhiteBox) — traceable GraphRAG over medical literature (Future of Data 2024, 1st place)
- [G-Code-Assembler](https://github.com/Aryagarg23/G-Code-Assembler) — G-code assembly + STL visualization (MakeUC 2024, Kinetic Vision winner)
- [Terminally-Addicted](https://github.com/Aryagarg23/Terminally-Addicted) — Spotify, GitHub, GPT and YouTube without leaving the terminal (HackOHI/O 2024)
- [Memento](https://github.com/Aryagarg23/Memento) — digital memory journal for Alzheimer's patients and caregivers (RevolutionUC 2024, 3rd overall)
- [Buycott](https://github.com/Aryagarg23/Buycott) — barcode scan -> parent company -> NLP stance on social issues (MakeUC 2023, 1st overall)
- [Kuka Arm Viz](https://github.com/Aryagarg23/Visualizing-Kuka-7-Node-Robot-Arm) — interactive 7-DOF robot arm in WebGL with inverse kinematics (RevolutionUC 2023)
- [Hi-Five](https://github.com/Aryagarg23/Hi-Five) — anonymous friend-matching on OCEAN personality vectors (SASEhack 2024)
- [Friction](https://github.com/Aryagarg23/Friction) — speculative OS + hardware that protects flow state with physical friction (Fig Build 2026)
