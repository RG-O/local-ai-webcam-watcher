# Local AI Webcam Monitor

A local, self-hosted Python application that watches a camera through its RTSP stream,
periodically captures screenshots, sends rolling batches of images to a
local Ollama vision model, and can send an ntfy notification when the AI
response begins with a configured trigger word.

The application includes a system tray icon and a LAN-accessible web
interface for configuring the monitor, viewing the latest camera image,
starting or stopping monitoring, and reviewing previous AI responses.

## Features

* Connects to RTSP cameras and network video streams
* Captures screenshots at a configurable interval
* Sends rolling batches of screenshots to a local Ollama vision model
* Lets you choose from locally installed Ollama models
* Uses a fully customizable AI prompt
* Detects alerts using a configurable first-word trigger
* Sends ntfy notifications when the trigger word is returned
* Optionally attaches the latest camera screenshot to notifications
* Stores AI response history locally
* Optional automatic monitoring when the application starts
* Web interface can be accessed by other devices on the local network
* Persistent settings and response history

## Requirements

* Python 3
* An RTSP-compatible camera or video stream
* [Ollama](https://ollama.com/) installed and running locally
* An Ollama vision model capable of accepting images
* Optional: an [ntfy](https://ntfy.sh/) topic for phone/desktop
notifications

The default model is:

``` text
qwen2.5vl:7b
```

You can use another locally installed Ollama vision model from the web
interface.

## Installation

### 1\. Install Python

Install a current version of Python 3 if you do not already have it.

During installation on Windows, it is helpful to enable **Add Python to
PATH**.

### 2\. Install Ollama

Install Ollama from:

https://ollama.com/

Then download a vision model. For example:

``` bash
ollama pull qwen2.5vl:7b
```

Make sure Ollama is running before starting Local AI Webcam Monitor.

### 3\. Download the Script

Download or clone this repository and place the Python script in a
folder where it can create its settings and history files.

### 4\. Install Python Dependencies

From a terminal or command prompt in the project folder, run:

``` bash
pip install opencv-python flask ollama requests pystray pillow
```

## Running

Run the application with:

``` bash
python rtsp\_ollama\_monitor\_live\_ai\_startup\_option.py
```

The exact filename can be changed if desired.

Once running, the application will display a system tray icon and start
the local web interface.

By default, the web interface is available at:

``` text
http://localhost:5000
```

The console also displays the LAN address you can use to open the
interface from another device on the same network.

For example:

``` text
http://192.168.1.50:5000
```

## Initial Setup

Open the web interface and configure the following settings.

### RTSP URL

Enter the RTSP URL provided by your camera.

A typical URL may look similar to:

``` text
rtsp://username:password@192.168.1.100:554/stream1
```

The exact format depends on your camera manufacturer.

### Screenshot Interval

Controls how often a new screenshot is captured from the RTSP stream.

### Rolling Batch Size

Controls how many recent screenshots are sent to Ollama in each
analysis.

For example, with a batch size of `4`, Ollama receives the four most
recent screenshots.

### Ollama Model

Select the local Ollama model used to analyze the images.

The dropdown is populated using the models currently installed in
Ollama.

### AI Prompt

This is the instruction sent to Ollama along with the screenshot batch.

The default prompt asks the model to begin its response with either:

``` text
ALERT
```

or:

``` text
NO
```

You can completely customize this prompt for your use case.

For example, the application could be configured to watch for:

* People
* Vehicles
* Package deliveries
* Animals
* Open doors or gates
* Specific activity in an area
* Other conditions a vision model can reasonably identify

### Trigger Word

The application examines the **first word** of the Ollama response.

If it matches the configured trigger word, the response is considered
triggered.

For example:

``` text
Trigger Word: ALERT
```

An AI response beginning with:

``` text
ALERT A person is standing near the front door.
```

will trigger a notification.

A response beginning with:

``` text
NO No people are visible in the monitored area.
```

will not.

Trigger-word matching is case-insensitive.

## ntfy Notifications

[ntfy](https://ntfy.sh/) can be used to send notifications from the
computer running this program to a phone or other device.

Configure:

* **ntfy Server** --- defaults to `https://ntfy.sh`
* **ntfy Topic** --- your ntfy topic
* **ntfy Notification Title** --- title shown on notifications
* **Attach latest screenshot** --- optionally includes the most recent
camera image

Install the ntfy app on your phone and subscribe to the same topic to
receive notifications.

### Topic Privacy

Public ntfy topics can potentially be discovered or accessed by someone
who knows the topic name. If using the public ntfy service, choose a
long, difficult-to-guess topic name rather than something simple.

For cameras or other privacy-sensitive uses, consider running your own
ntfy server or using ntfy authentication/access controls.

## Web Interface

The web interface provides several sections.

### Status

Shows live information including:

* Whether watching is currently running
* Camera connection status
* Number of screenshots captured
* Number of AI calls
* Whether an AI analysis is currently running
* Last capture time
* Last AI analysis time
* Last AI duration
* Latest camera screenshot

Directly below the camera image, the interface shows the **most recent
full AI response** and whether it was:

* **TRIGGERED**
* **NOT TRIGGERED**

The page updates automatically while it is open.

### Start / Stop Watching

Monitoring can be paused without shutting down the application.

When watching is stopped:

* The RTSP camera connection is released
* New screenshots are not captured
* New Ollama analyses are not started
* The web interface remains available
* The system tray icon remains available

When watching is started again, the application reconnects to the RTSP
stream.

The rolling screenshot buffer is cleared when monitoring stops so
screenshots from before the pause are not mixed with new screenshots.

If an Ollama request is already running when monitoring is stopped, that
request may finish, but no new analysis will begin until monitoring is
started again.

### Start Watching on Startup

The **Start watching automatically when the program starts** preference
controls whether monitoring begins automatically the next time the
application is launched.

It is disabled by default.

This preference does not immediately start or stop the current
monitoring session when saved. Use the Start/Stop Watching control for
that.

### AI Response History

Previous AI analyses are stored locally and displayed in the web
interface.

History entries can include:

* Timestamp
* Model
* Full AI response
* Trigger status
* Notification status
* Screenshot batch size
* AI duration
* Input token count
* Output token count
* Errors, if applicable

History can be cleared from the web interface.

## System Tray

Right-click the Local AI Webcam Monitor system tray icon to access
application controls.

The tray menu includes options to:

* Open Settings / History
* Start Watching or Stop Watching
* Quit the application

The Start/Stop option reflects the current monitoring state.

## Local Files

The application creates files alongside the Python script.

### `settings.json`

Stores saved preferences.

Do not commit your personal `settings.json` file to a public repository
if it contains an RTSP URL with a camera username/password or a private
ntfy topic.

### `history.json`

Stores previous AI responses.

Depending on your prompt and camera use, AI responses could contain
information about activity captured by the camera. You may not want to
commit this file either.

A recommended `.gitignore` is:

``` gitignore
settings.json
history.json
\_\_pycache\_\_/
\*.pyc
```

## Network and Security Considerations

The Flask web server listens on:

``` text
0.0.0.0
```

This is intentional so the settings page can be accessed by other
devices on the local network.

However, the included web interface does **not** provide authentication.

Anyone who can reach the configured port on your network may be able to
view the interface and change settings.

For that reason:

* Use this application only on networks you trust.
* Do not expose the Flask port directly to the public internet.
* Do not port-forward the web interface through your router.
* Be careful when using RTSP URLs containing usernames and passwords.
* Consider firewall rules if you need to restrict which devices can
access the interface.

## How the Rolling Analysis Works

Suppose the settings are:

``` text
Screenshot Interval: 5 seconds
Rolling Batch Size: 4
```

The application captures one screenshot every five seconds.

After enough screenshots have been collected, the most recent four
images are sent to Ollama together.

As new screenshots arrive, older screenshots fall out of the rolling
buffer. This gives the vision model multiple moments in time instead of
requiring it to make a decision from only one frame.

The AI's response is stored in history. If its first word matches the
configured trigger word, the application attempts to send an ntfy
notification.

## Performance

Vision models can require significant CPU, GPU, VRAM, and system memory.

Performance depends on:

* Ollama model
* GPU
* Available VRAM
* Screenshot dimensions
* Batch size
* Screenshot frequency
* Complexity of the prompt

The **Max Image Width** and **Max Image Height** settings can be reduced
to lower the amount of image data sent to the model.

Reducing image resolution and batch size will generally make analysis
faster, but may also reduce the model's ability to identify small or
subtle details.

## Troubleshooting

### No Ollama Models Appear

Make sure Ollama is installed and running.

Check installed models with:

``` bash
ollama list
```

If necessary, install the default model:

``` bash
ollama pull qwen2.5vl:7b
```

### Camera Does Not Connect

Verify the RTSP URL with another RTSP-compatible application such as
VLC.

Check:

* Camera IP address
* Username and password
* RTSP port
* Stream path
* Camera RTSP settings
* Firewall/network connectivity

### No ntfy Notifications

Verify that:

* An ntfy topic has been entered
* Your phone is subscribed to the same topic
* The AI response actually begins with the configured trigger word
* The computer can access the configured ntfy server

You can use **Send Test ntfy Notification** from the web interface to
test ntfy separately from the camera and Ollama.

### Web Interface Cannot Be Opened From Another Device

Make sure both devices are on the same network.

You may also need to allow Python through Windows Firewall for private
networks.

Use the LAN address printed by the application rather than `localhost`.
`localhost` always refers to the device on which the browser itself is
running.

## Privacy

Camera screenshots are sent to the Ollama instance configured on the
computer running this application. With a standard local Ollama
installation, model inference is performed locally.

Triggered screenshots may leave the local computer if you enable image
attachments for notifications through an external ntfy server.

Review the privacy and security requirements appropriate for your camera
location and use case before deploying the application.

## Disclaimer

AI vision models can make mistakes, miss events, or incorrectly identify
activity. This application should not be relied upon as the sole system
for security, emergency detection, life safety, or other situations
where an incorrect AI response could cause harm.

Use the application in accordance with applicable privacy, surveillance,
recording, and notification laws and regulations in your location.

## License

[MIT License](LICENSE)
