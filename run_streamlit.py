import subprocess
import sys
import time

# Kill any existing streamlit processes
try:
    subprocess.run(["taskkill", "/F", "/IM", "python.exe"], capture_output=True)
    time.sleep(1)
except:
    pass

# Run streamlit app and capture output
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"],
    cwd="D:\\4.+Bearings\\4. Bearings\\IMS",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait a bit for startup
time.sleep(10)

# Get output
stdout, stderr = proc.communicate(timeout=15)
print("STDOUT:")
print(stdout[:5000])
if stderr:
    print("STDERR:")
    print(stderr[:2000])