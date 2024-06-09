# Hand Gesture Recognition Based Game Controller

This project utilizes computer vision and machine learning techniques to create a game controller based on hand gesture recognition. It leverages libraries such as OpenCV, MediaPipe, NumPy, and PyAutoGUI to interpret hand gestures as inputs for controlling games or applications.

## Getting Started

### Pre-requisites

Ensure you have Python 3.12 installed on your system. This project also requires a webcam for hand gesture recognition.

### Installation

1. Clone the repository to your local machine:

```sh
git clone https://github.com/Jay-sanjay/Gemini-Api-Based-SQL-Query-Retriving-Web-App
```
2. Navigate to the project directory:

```sh
cd /path/to/Gemini-Api-Based-SQL-Query-Retriving-Web-App
```

3. Create a virtual environment:

```sh
python3 -m venv venv
```

4. Activate the virtual environment:

```sh
source venv/bin/activate
```

5. Install the required packages:

```sh
pip install -r requirements.txt
```

6. Run the application:

```sh
python app.py
```

7. If you are on a Linux system then you might need to run the following command to run the program successfully:

```sh
sudo apt-get install python3-tk python3-dev
```


## How It Works
The project uses MediaPipe for hand tracking and gesture recognition. OpenCV is used for image processing, and PyAutoGUI simulates keyboard and mouse inputs based on recognized gestures. The recognition logic is defined in asphalt.py, which can be modified to add new gestures or change the control scheme.

## Contributing
Contributions to improve the project are welcome. Please follow the standard fork and pull request workflow.

