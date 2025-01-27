# Object Detection and Tracking

## Overview
This project implements a real-time object detection and tracking system using deep learning models, specifically YOLOv5 and OpenCV. The system captures video from the webcam, detects objects in the video frames, and tracks them using specified algorithms.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Notes](#notes)
- [Authors](#authors)
- [License](#license)

## Features
- **Real-Time Object Detection**: Utilizes YOLOv5 for fast and accurate object detection.
- **Object Tracking**: Implements CSRT object tracking to follow detected objects in video feeds.
- **Customizable**: Easily modifiable to include different models or video sources.

## Requirements
- Python 3.x
- PyTorch (1.7 or higher)
- torchvision
- numpy
- opencv-python
- pandas

### Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd Object-Detection-Tracking
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run Object Detection**:
   - This script captures video from the webcam and detects objects using the YOLOv5 model.
   ```bash
   python object_detection.py
   ```

2. **Run Object Tracking**:
   - This script uses detected objects from the YOLOv5 model and tracks them using OpenCV’s CSRT tracker.
   ```bash
   python tracker.py
   ```

3. **Exit the Application**: 
   - Press the 'q' key to exit the video stream.

## Running the Project
### Prerequisites
Ensure you have a webcam connected or provide a video file path to the script. The YOLOv5 model will automatically download its weights when the code is run for the first time.

### Example
1. Start the webcam and run the object detection:
   - Open a terminal window and execute the object detection script:
   ```bash
   python object_detection.py
   ```

2. Open another terminal window and run the tracking script:
   ```bash
   python tracker.py
   ```

## Project Structure
The project directory typically looks like this:
```
Object-Detection-Tracking/
├── object_detection.py       # Script for real-time object detection using YOLOv5
├── tracker.py                # Script for tracking detected objects
├── requirements.txt          # List of required Python packages
└── README.md                 # Project documentation
```

## Notes
- You can customize the model used in the detection script by modifying the line where the model is loaded. You can choose between different YOLOv5 model sizes (YOLOv5s, YOLOv5m, etc.) based on your performance needs.
- Ensure that your Python environment has access to a compatible GPU for faster inference.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author
- Zunaid Hasasn</br>
- hasan15-6033@diu.edu.bd
