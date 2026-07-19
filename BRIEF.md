# BRIEF — SignLink (repo: Aryagarg23/SignLink)
Slug: signlink. Hackathon: BoilerMake X (Purdue), January 2023. Prizes: none listed.
Team: Arya Garg, Reeve Prinson Fernandes (backend JS, PeerJS, MediaPipe gesture recognition), Arleen Monteiro. Original team repo: https://github.com/ReeveFernandes/SignLink
What: video conferencing for the deaf/hard-of-hearing — real-time ASL fingerspelling to text during calls. MediaPipe hand landmarks → 20 nodes per hand → vector algebra/trig on joint angles → hand-coded matcher against an ASL letter database. WebRTC + PeerJS calls; letter-by-letter, honestly framed as foundational.
Tech: HTML5/CSS/JS, WebRTC, PeerJS, MediaPipe, OpenCV.
Devpost: https://devpost.com/software/signlink
Prototype idea: synthetic joint-angle vectors for ~6 ASL letters (A,B,C,L,V,Y) with noise, nearest-neighbor matcher, chart 1: heatmap of mean joint angle per letter (letters x joints), chart 2: classification accuracy per letter bar at a given noise level.
