# face-swapper
Realtime face swapper. 
It uses your camera or a video file as input and swaps the two faces it finds, like those popular smartphone apps.
It's mostly a rewrite of this C++ project to Scala: https://github.com/hrastnik/FaceSwap

Here's what's different:
- handles camera **or** video file input
- uses OpenCV's face landmark API to remove dependency on dlib which is cumbersome on JVM
- rewritten/simplified face detection and tracking
- optimized blitting and LUT application by using OpenCV API - doing it in Scala was sluggish
- other smaller tweaks

### TODO
- debug/solve the face jitteriness
- cleanup and rewrite the code to make it more Scala idiomatic