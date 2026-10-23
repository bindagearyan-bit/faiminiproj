"""
colab_tunnel.py

One-click Google Colab live backend launcher for CardioPulse AI.
Sets up a public HTTPS tunnel (Cloudflare Tunnel or pyngrok) inside Google Colab
so the website can send live patient vitals directly to Colab and receive predictions
in real time.
"""

import os
import subprocess
import sys
import time
import urllib.request
import json
import platform

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_cloudflare_tunnel(port=5000):
    """Downloads cloudflared binary if needed and starts an HTTPS tunnel without token."""
    print("[1/3] Setting up Cloudflare Public Tunnel for Google Colab...")
    
    bin_name = "cloudflared"
    if platform.system() == "Windows":
        bin_name = "cloudflared.exe"
        if not os.path.exists(bin_name):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            print(f"Downloading {bin_name}...")
            urllib.request.urlretrieve(url, bin_name)
    else:
        # Linux (Google Colab default environment)
        if not os.path.exists(bin_name):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            print(f"Downloading {bin_name} for Colab Linux environment...")
            urllib.request.urlretrieve(url, bin_name)
            os.chmod(bin_name, 0o777)

    log_file = "cloudflared.log"
    cmd = [f"./{bin_name}" if platform.system() != "Windows" else bin_name, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f)
    
    print("[2/3] Tunnel starting... Waiting for public URL...")
    tunnel_url = None
    for _ in range(30):
        time.sleep(1)
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                for line in content.split("\n"):
                    if ".trycloudflare.com" in line:
                        for part in line.split():
                            if part.startswith("https://") and ".trycloudflare.com" in part:
                                tunnel_url = part.strip()
                                break
        if tunnel_url:
            break
            
    return tunnel_url, proc

def main():
    print("=" * 70)
    print("   🫀 CARDIOPULSE AI — GOOGLE COLAB REAL-TIME BACKEND ENGINE")
    print("=" * 70)
    
    os.environ["IS_COLAB"] = "1"
    port = 5000
    
    # Check if user provided ngrok authtoken
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
    public_url = None
    
    if ngrok_token:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(ngrok_token)
            tunnel = ngrok.connect(port, bind_tls=True)
            public_url = tunnel.public_url
            print(f"[TUNNEL] Ngrok active: {public_url}")
        except Exception as e:
            print(f"[WARN] Ngrok setup failed: {e}. Falling back to Cloudflare Tunnel.")
    
    if not public_url:
        public_url, tunnel_proc = setup_cloudflare_tunnel(port)

    print("\n" + "=" * 70)
    if public_url:
        print("  🎉 GOOGLE COLAB BACKEND IS LIVE & CONNECTED TO THE CLOUD!")
        print(f"  🌐 Public API URL:  {public_url}")
        print("=" * 70)
        print("\n👉 STEP TO CONNECT WEBSITE:")
        print(f"   1. Open the website: http://127.0.0.1:5000 (or hosted site)")
        print(f"   2. Click the '☁️ Connect Google Colab' button at the top right.")
        print(f"   3. Paste this URL:  {public_url}")
        print(f"   4. Click 'Connect'. When you move sliders on the website, this")
        print(f"      Colab notebook will execute the model and display logs below!")
        print("=" * 70 + "\n")
    else:
        print("⚠️ Could not automatically detect public tunnel URL.")
        print(f"   Local backend is running on http://127.0.0.1:{port}")
        print("=" * 70 + "\n")

    # Start Flask application directly
    from app import app
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
