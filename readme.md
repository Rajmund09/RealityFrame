<div align="center">
  <h1>🌌 RealityFrame</h1>
  <p><strong>A Real-Time Advanced Computer Vision & Augmented Reality System</strong></p>
  <p>
    RealityFrame combines cutting-edge AI technologies, gesture recognition, background reconstruction, virtual camera support, and interactive AR capabilities to create an immersive, privacy-focused visual effects engine built purely in Python.
  </p>
</div>

---

## ✨ Key Features

### ✋ Advanced Gesture Control & Invisibility Modes
Take control of your presence with dynamic invisibility and privacy modes, all triggered seamlessly via intuitive hand gestures (powered by MediaPipe):
- **Portal Mode:** Create a floating window into reality while keeping the rest obscured.
- **Full Invisibility:** Seamlessly blend into the reconstructed background for complete stealth.
- **Blur Mode:** Maintain a professional aesthetic by applying an adaptive blur to the environment.
- **Toggle Mechanism:** Switch modes effortlessly using a **Two-Hand Pinch Gesture**.

### 🖌️ Interactive 3D AR Canvas
Transform your space into a canvas. RealityFrame offers a powerful Augmented Reality drawing experience:
- **Draw in Thin Air:** Use your index finger to draw dynamic strokes in 3D space.
- **Body Anchoring:** Press `N` to anchor your drawing to a body part (e.g., attach a drawn halo to your head). As you move closer, further, or shift around, the drawing scales and follows you naturally!
- **Dynamic Brush Controls:** Change brush size, cycle colors, undo strokes, or clear the canvas.
- **Lift Pen:** Simply make a **Fist Gesture** to stop drawing and start a new stroke.

### 🔍 Dynamic Zoom & Focus Area
- **One-Hand Pinch Zoom:** Smoothly zoom in and out continuously without touching the keyboard.
- **Custom Focus Window (Peace Gesture):** Use the 'Peace' (V) hand gesture to specify exact regions of the screen you want to remain crystal clear.

### 🎥 Virtual Camera Output
Take your AR and privacy enhancements into your daily workflow. Stream your RealityFrame output directly into **Zoom, Microsoft Teams, Google Meet**, or any software supporting virtual webcams.

### 🖼️ Real-Time Background Reconstruction & Replacement
- **Adaptive Reconstruction:** The system constantly learns and reconstructs the real background even while you stand in front of it.
- **Custom Backgrounds:** Replace your surroundings instantly with any image or video file.

### 🎯 AR Target Tracking & Overlays
- **Marker Tracking:** Detect custom AR markers in real-time.
- **Dynamic Overlays:** Render images and visual elements anchored perfectly to tracked targets.

---

## 🎮 Controls

### 🖐️ Gesture Commands
| Action | Gesture |
| :--- | :--- |
| **Cycle Invisibility Mode** | Two-Hand Pinch |
| **Zoom In / Out** | One-Hand Pinch (Continuous) |
| **Start Focus Selection** | Peace (V) Gesture |
| **Draw on AR Canvas** | Pointing Finger (While in Draw Mode) |
| **Stop Drawing (Pen Up)** | Fist Gesture (While in Draw Mode) |

### ⌨️ Keyboard Shortcuts
| Key | Action |
| :---: | :--- |
| **`D`** | Toggle AR Draw Mode |
| **`C`** | Toggle Virtual Camera / Cycle Brush Color (Draw Mode) |
| **`W`** | Cycle Brush Size (Draw Mode) |
| **`N`** | Anchor Drawing to Body (Draw Mode) |
| **`U`** | Undo Last Stroke (Draw Mode) |
| **`X`** | Clear Canvas (Draw Mode) |
| **`F`** | Toggle Focus Window |
| **`R`** | Reset Focus Selection |
| **`B`** | Recapture/Reset Background |
| **`A`** | Toggle AR Overlays |
| **`V`** | Toggle Tracking Debug Frame |
| **`I`** | Load Custom Background (Image/Video) |
| **`Q`** | Quit Application |

---

## 🛠️ Project Structure

```text
RealityFrame/
├── ar/                 # Augmented Reality tracking and overlay logic
├── assets/             # Images, markers, and visual assets
├── core/               # Core engine: AR Canvas, Backgrounds, Virtual Cam, Zoom
├── graphics/           # Rendering utilities
├── tools/              # Helper scripts (e.g., marker generation)
├── vision/             # AI components: Gestures, Hand/Pose Tracking, Portal Detection
├── main.py             # Main execution script
└── requirements.txt    # Python dependencies
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rajmund09/RealityFrame.git
   cd RealityFrame
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For Virtual Camera support, you must have OBS Virtual Camera or a compatible virtual webcam driver installed on your OS).*

3. **Run RealityFrame:**
   ```bash
   python main.py
   ```

---

## 💡 Practical Use Cases
- **Content Creation:** Record videos with zero-latency visual effects, body-anchored AR drawings, and dynamic background removal without needing a green screen.
- **Remote Work & Meetings:** Keep your workspace professional by blurring out distractions or loading custom video backgrounds.
- **Interactive Presentations:** Walk through concepts while drawing annotations mid-air, utilizing the AR canvas to captivate your audience.

---

## 🔮 Future Roadmap
- [ ] Advanced 3D AR object placement (beyond 2D canvas strokes).
- [ ] Multi-user tracking and interaction.
- [ ] Object-specific invisibility modes.
- [ ] Optimized segmentation edge-smoothing.

*Built as an exploration of real-time computer vision, human-computer interaction, privacy systems, and augmented reality.*
