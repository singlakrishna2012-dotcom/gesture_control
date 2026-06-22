#!/usr/bin/env python3
"""
Web server to launch and control gestures.py for Mac
Run with: python3 server.py
Then open: http://localhost:5500
"""

import os
import sys
import json
import subprocess
import signal
import time
import threading
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# Global process handle
gesture_process = None
process_lock = threading.Lock()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gesture Control Launcher</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #fff;
            padding: 20px;
        }
        .container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 40px;
            max-width: 700px;
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #aaa;
            margin-bottom: 30px;
            font-size: 16px;
        }
        .status-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            min-height: 80px;
            border: 1px solid rgba(0, 210, 255, 0.2);
            text-align: left;
            line-height: 1.8;
        }
        .status-box .label {
            color: #888;
        }
        .status-box .value {
            color: #00d2ff;
            font-weight: bold;
        }
        .status-box .value.running {
            color: #00e676;
        }
        .status-box .value.stopped {
            color: #ff1744;
        }
        .btn-group {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin: 20px 0;
        }
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            color: #fff;
        }
        .btn-primary {
            background: linear-gradient(135deg, #00c853, #00e676);
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 200, 83, 0.3);
        }
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        .btn-danger {
            background: linear-gradient(135deg, #d32f2f, #ff1744);
        }
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(211, 47, 47, 0.3);
        }
        .btn-danger:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        .info-text {
            color: #888;
            font-size: 14px;
            margin-top: 20px;
            line-height: 1.6;
        }
        .info-text code {
            background: rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 4px;
            color: #00d2ff;
            font-family: 'SF Mono', Monaco, monospace;
        }
        .log-area {
            background: rgba(0, 0, 0, 0.5);
            border-radius: 8px;
            padding: 10px;
            margin-top: 15px;
            max-height: 150px;
            overflow-y: auto;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 12px;
            color: #aaa;
            text-align: left;
        }
        .log-area .log-entry {
            padding: 2px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .log-area .log-entry .time {
            color: #666;
        }
        .log-area .log-entry .msg {
            color: #00d2ff;
        }
        .log-area .log-entry .error {
            color: #ff1744;
        }
        .log-area .log-entry .success {
            color: #00e676;
        }
        @media (max-width: 600px) {
            .container { padding: 30px 20px; }
            h1 { font-size: 1.8em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖐️ Gesture Control</h1>
        <p class="subtitle">Virtual Mouse with Hand Tracking for Mac</p>

        <div class="status-box" id="statusBox">
            <div><span class="label">📌 Status:</span> <span class="value stopped" id="statusText">Not Running</span></div>
            <div><span class="label">🆔 PID:</span> <span id="pidText">-</span></div>
            <div><span class="label">📁 Script:</span> <span id="scriptText">gestures.py</span></div>
        </div>

        <div class="btn-group">
            <button id="startBtn" class="btn btn-primary">▶️ Start Gesture Control</button>
            <button id="stopBtn" class="btn btn-danger" disabled>⏹️ Stop</button>
        </div>

        <div class="log-area" id="logArea">
            <div class="log-entry"><span class="time">[System]</span> <span class="msg">Ready to start</span></div>
        </div>

        <div class="info-text">
            <p>📌 <strong>Python Version:</strong> Full mouse control with MediaPipe</p>
            <p>⚠️ Make sure <code>gestures.py</code> and <code>hand_landmarker.task</code> are in the same folder</p>
            <p style="margin-top:10px;font-size:12px;color:#666;">
                Press <kbd>Control (Ctrl) + C</kbd> in Terminal to stop the server
            </p>
        </div>
    </div>

    <script>
        let statusCheckInterval = null;

        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusText = document.getElementById('statusText');
        const pidText = document.getElementById('pidText');
        const logArea = document.getElementById('logArea');

        function addLog(message, type = 'msg') {
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="time">[${time}]</span> <span class="${type}">${message}</span>`;
            logArea.appendChild(entry);
            logArea.scrollTop = logArea.scrollHeight;
            if (logArea.children.length > 50) {
                logArea.removeChild(logArea.firstChild);
            }
        }

        async function checkStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                if (data.running) {
                    statusText.textContent = '✅ Running';
                    statusText.className = 'value running';
                    pidText.textContent = data.pid || 'N/A';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                } else {
                    statusText.textContent = '⏸️ Stopped';
                    statusText.className = 'value stopped';
                    pidText.textContent = '-';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }
            } catch (e) {
                statusText.textContent = '⚠️ Disconnected';
                statusText.className = 'value stopped';
            }
        }

        async function startGesture() {
            try {
                addLog('Starting gesture control...', 'msg');
                startBtn.disabled = true;
                startBtn.textContent = '⏳ Starting...';

                const response = await fetch('/start', { method: 'POST' });
                const data = await response.json();

                if (data.success) {
                    addLog(`✅ Started successfully! PID: ${data.pid}`, 'success');
                    await checkStatus();
                } else {
                    addLog(`❌ Failed: ${data.error}`, 'error');
                    startBtn.disabled = false;
                    startBtn.textContent = '▶️ Start Gesture Control';
                }
            } catch (e) {
                addLog(`❌ Connection error: ${e.message}`, 'error');
                startBtn.disabled = false;
                startBtn.textContent = '▶️ Start Gesture Control';
            }
        }

        async function stopGesture() {
            try {
                addLog('Stopping gesture control...', 'msg');
                stopBtn.disabled = true;
                stopBtn.textContent = '⏳ Stopping...';

                const response = await fetch('/stop', { method: 'POST' });
                const data = await response.json();

                if (data.success) {
                    addLog('✅ Stopped successfully', 'success');
                    await checkStatus();
                } else {
                    addLog(`❌ Failed: ${data.error}`, 'error');
                }
                stopBtn.textContent = '⏹️ Stop';
            } catch (e) {
                addLog(`❌ Connection error: ${e.message}`, 'error');
                stopBtn.textContent = '⏹️ Stop';
                stopBtn.disabled = false;
            }
        }

        startBtn.addEventListener('click', startGesture);
        stopBtn.addEventListener('click', stopGesture);

        // Check status every 3 seconds
        checkStatus();
        statusCheckInterval = setInterval(checkStatus, 3000);

        console.log('Gesture Control Launcher loaded');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    with process_lock:
        if gesture_process and gesture_process.poll() is None:
            return jsonify({
                'running': True,
                'pid': gesture_process.pid
            })
        else:
            return jsonify({'running': False})

@app.route('/start', methods=['POST'])
def start_gesture():
    global gesture_process
    
    with process_lock:
        # Check if already running
        if gesture_process and gesture_process.poll() is None:
            return jsonify({
                'success': False,
                'error': 'Already running'
            })
        
        try:
            script_dir = Path(__file__).parent
            gesture_script = script_dir / 'gestures.py'
            
            if not gesture_script.exists():
                return jsonify({
                    'success': False,
                    'error': f'gestures.py not found in {script_dir}'
                })
            
            # Check for model file
            model_file = script_dir / 'hand_landmarker.task'
            if not model_file.exists():
                return jsonify({
                    'success': False,
                    'error': 'hand_landmarker.task not found!'
                })
            
            # Start the process for Mac
            print(f"🚀 Starting gesture script: {gesture_script}")
            
            gesture_process = subprocess.Popen(
                [sys.executable, str(gesture_script)],
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment to see if it starts
            time.sleep(1)
            
            if gesture_process.poll() is not None:
                stdout, stderr = gesture_process.communicate()
                error_msg = stderr or stdout or 'Unknown error'
                gesture_process = None
                return jsonify({
                    'success': False,
                    'error': f'Process died: {error_msg[:200]}'
                })
            
            print(f"✅ Gesture script started with PID: {gesture_process.pid}")
            
            return jsonify({
                'success': True,
                'pid': gesture_process.pid,
                'message': 'Started successfully'
            })
            
        except Exception as e:
            print(f"❌ Failed to start: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            })

@app.route('/stop', methods=['POST'])
def stop_gesture():
    global gesture_process
    
    with process_lock:
        if not gesture_process or gesture_process.poll() is not None:
            gesture_process = None
            return jsonify({
                'success': True,
                'message': 'No process running'
            })
        
        try:
            print(f"⏹️ Stopping gesture script (PID: {gesture_process.pid})")
            
            # Try graceful termination
            gesture_process.terminate()
            
            # Wait for it to finish
            try:
                gesture_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print("⚠️ Force killing...")
                gesture_process.kill()
                gesture_process.wait()
            
            print("✅ Process stopped")
            gesture_process = None
            
            return jsonify({
                'success': True,
                'message': 'Stopped successfully'
            })
            
        except Exception as e:
            print(f"❌ Failed to stop: {e}")
            gesture_process = None
            return jsonify({
                'success': False,
                'error': str(e)
            })

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              Gesture Control Web Launcher                        ║
╠══════════════════════════════════════════════════════════════════╣
║  🌐 Server running at: http://localhost:5500                    ║
║                                                                 ║
║  📌 Click "Start Gesture Control" to run gestures.py            ║
║  ⏹️ Click "Stop" to stop the script                            ║
║                                                                 ║
║  Press Control (Ctrl) + C in this terminal to stop the server   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        app.run(host='0.0.0.0', port=5500, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        with process_lock:
            if gesture_process and gesture_process.poll() is None:
                print("⏹️ Stopping gesture process...")
                gesture_process.terminate()
                try:
                    gesture_process.wait(timeout=2)
                except:
                    pass