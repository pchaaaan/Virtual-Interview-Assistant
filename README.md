<h1>Real-Time Posture and Gaze Analysis</h1>

<p>This project utilizes computer vision techniques to provide real-time feedback on posture and gaze direction using Mediapipe and OpenCV. The system analyzes the user's posture and gaze during a session, offering corrective feedback when necessary.</p>

<h2>Getting Started</h2>

<h3>Prerequisites</h3>
<ul>
    <li>Python</li>
    <li>OpenCV</li>
    <li>Mediapipe</li>
    <li>NumPy</li>
</ul>

<h3>Dataset</h3>
<p>This project does not require a pre-existing dataset as it operates in real-time using webcam input.</p>

<h3>Installation</h3>
<pre><code>pip install opencv-python mediapipe numpy
</code></pre>

<h2>Implementation Details</h2>

<h3>Initial Setup</h3>
<ul>
    <li>Initialize Mediapipe Face Mesh and Pose modules.</li>
    <li>Set up drawing utilities for visual annotations.</li>
</ul>

<h3>Parameters</h3>
<ul>
    <li><strong>Posture Threshold</strong>: Threshold for detecting correct shoulder posture.</li>
    <li><strong>Posture Buffer Frames</strong>: Number of frames allowed for bad posture before alert.</li>
    <li><strong>Gaze Threshold</strong>: Threshold for detecting focused gaze.</li>
    <li><strong>Gaze Buffer Frames</strong>: Number of frames allowed for out-of-focus gaze before alert.</li>
    <li><strong>Laptop Screen Ratio</strong>: Screen range considered as 'focused'.</li>
</ul>

<h3>Real-Time Analysis</h3>
<ul>
    <li>Capture video from the webcam.</li>
    <li>Process each frame to detect face landmarks and pose landmarks.</li>
    <li>Analyze shoulder posture and gaze direction.</li>
    <li>Provide feedback on posture and gaze.</li>
    <li>Display the annotated video feed with feedback.</li>
</ul>

<h3>Session Summary</h3>
<ul>
    <li>Calculate session duration.</li>
    <li>Compute the percentage of frames with correct posture.</li>
    <li>Compute the percentage of frames with focused gaze.</li>
</ul>

<h2>Results</h2>
<ul>
    <li>The system provides real-time feedback on posture and gaze direction.</li>
    <li>Summary statistics offer insights into the user's posture and focus during the session.</li>
</ul>
